# 可变状态工具开发指南

## 1. 什么是有状态工具？

大多数工具是**无状态**的——每次调用独立执行，不依赖之前的调用结果。例如 `WeatherTool`：查询"北京"和查询"上海"之间没有任何关联。

但有些工具需要**记住上下文**。例如 `TerminalTool`：用户先 `cd /app`，再执行 `ls`，`ls` 必须在 `/app` 目录下执行。`current_dir` 就是一个需要跨调用保持的**可变状态**。

在多线程环境下（如 Flask 多线程模式、ThreadPoolExecutor），多个用户可能共享同一个 Tool 实例。如果直接把状态存在实例属性（`self.current_dir`）上，就会出现**状态串扰**——线程 A 的 `cd` 影响了线程 B 的 `ls`。

## 2. 核心机制：ContextVar

violet_agents 的 `Tool` 基类使用 Python 标准库的 `contextvars.ContextVar` 实现状态隔离：

```
                    TerminalTool 实例（唯一）
               ┌─────────────────────────────────┐
               │  workspace, timeout, ...          │ ← 不可变，所有线程共享
               │                                  │
               │  _state_var ─────────────────────┼──→ 线程A: {"current_dir": "/home/alice"}
               │                                  │    线程B: {"current_dir": "/home/bob"}
               └─────────────────────────────────┘
```

每个线程调用 `_state_var.get()` 时拿到的是**自己**的状态副本，互不干扰。

### 基类提供的三个入口

| 方法 | 用途 | 是否需要重写 |
|---|---|---|
| `_make_default_state()` → `Dict[str, Any]` | 定义状态的初始值 | **必须** |
| `_get_state()` → `Dict[str, Any]` | 获取当前上下文的状态（懒初始化） | 不需要 |
| `get_session_state()` → `Dict[str, Any]` | 深拷贝当前状态，用于跨 Session 持久化 | 不需要（基类已做 `copy.deepcopy`） |
| `restore_session_state(state)` | 从持久化数据恢复状态到当前上下文 | 仅在需**验证/修正**数据时重写 |
| `reset()` | 重置当前上下文的状态为默认值 | 不需要（基类直接调用 `_state_var.set(_make_default_state())`） |

### 数据流

```
run() 中读写状态         Session 切换时持久化
───────────────         ──────────────────
                        TurnEnd 钩子触发
current_dir = /app  ──→ get_session_state() ──→ sess.save_tool_state("terminal", {...})
                              │
                              │ deepcopy（基类完成）
                              ↓
                        {"current_dir": "/app"}
                              │
                        TurnStart 钩子触发
                              ↓
current_dir = /app  ←── restore_session_state({...})  ←── sess.get_tool_state("terminal")
```

## 3. 三步创建可变状态工具

### 第一步：定义默认状态

重写 `_make_default_state()`，返回一个**纯 dict**（可 JSON 序列化）：

```python
from typing import Any, Dict
from ..base import Tool

class CounterTool(Tool):
    def __init__(self):
        super().__init__(name="counter", description="一个计数工具")

    def _make_default_state(self) -> Dict[str, Any]:
        return {"count": 0}
```

这是**唯一必须重写**的方法。基类在首次访问状态时会调用它来懒初始化。

### 第二步：添加便捷访问属性

通过 `_get_state()` 读写状态 dict，封装为 property：

```python
class CounterTool(Tool):
    # ... __init__, _make_default_state ...

    @property
    def count(self) -> int:
        """当前计数，通过 ContextVar 实现每上下文隔离。"""
        return self._get_state()["count"]

    @count.setter
    def count(self, value: int) -> None:
        self._get_state()["count"] = value

    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        action = parameters.get("action", "increment")
        if action == "increment":
            self.count += 1
        elif action == "reset_count":
            self.count = 0
        return Message(
            role="tool",
            content=f"当前计数: {self.count}",
            tool_call_id=tool_call_id
        )
```

**关键点**：不要缓存 `_get_state()` 的返回值到局部变量然后修改。以下写法是**错误**的：

```python
# ❌ 错误：局部变量修改不会写回 ContextVar
state = self._get_state()
state["count"] += 1  # 修改了 dict 内容（可以工作，但不够安全）
# 更危险的错误：
state = {"count": 999}  # 只是改了局部变量，ContextVar 完全不受影响
```

正确做法是每次访问都通过 `_get_state()`，或者修改后调用 `_state_var.set()`。property 模式是最推荐的方式。

### 第三步（可选）：添加恢复验证

如果你的状态有**合法性约束**（如路径必须在沙箱内），重写 `restore_session_state` 做验证，然后**必须调用 `super()`** 委托基类执行实际的 ContextVar 写入：

```python
class CounterTool(Tool):
    # ... 其他方法 ...

    def restore_session_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        """验证计数器不为负数，然后委托基类执行原子替换。"""
        if not state:
            super().restore_session_state(None)
            return

        validated = dict(state)  # 不修改入参
        if validated.get("count", 0) < 0:
            validated["count"] = 0  # 修正非法值

        super().restore_session_state(validated)
```

**为什么必须调用 `super()`？** 因为基类执行了 `self._state_var.set(copy.deepcopy(state))`，这是**整体替换**而非逐个字段修改。如果你自己逐个字段写（如 `self.count = validated["count"]`），当状态有多个 key 时，旧 key 可能残留。

## 4. 完整示例：会话记忆工具

下面是一个记录对话摘要的工具，展示完整的最佳实践：

```python
"""SessionMemoryTool — 记录和读取会话摘要的工具有状态工具示例"""

from typing import Any, Dict, Optional
from ..base import Tool, ToolParameters, ToolProperty
from ...core.message import Message


TOOL_NAME = "session_memory"
TOOL_DESCRIPTION = "记录会话中的重要信息，支持写入、读取和清空操作"


class SessionMemoryTool(Tool):
    """会话记忆工具 — 可变状态工具的最佳实践参考实现。

    状态结构：
        {"notes": [...], "key_facts": {...}}

    线程安全：所有可变状态通过 Tool 基类的 ContextVar 隔离。
    """

    def __init__(self):
        super().__init__(name=TOOL_NAME, description=TOOL_DESCRIPTION)

    # ──── 必须重写：定义默认状态 ────
    def _make_default_state(self) -> Dict[str, Any]:
        return {
            "notes": [],
            "key_facts": {},
        }

    # ──── 便捷属性 ────
    @property
    def _state(self) -> Dict[str, Any]:
        """返回当前上下文的状态 dict 引用，仅用于修改。
        不要替换此引用，只修改其中的 key。"""
        return self._get_state()

    # ──── 核心逻辑 ────
    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        action = parameters.get("action", "read")
        if action == "write":
            return self._handle_write(parameters, tool_call_id)
        elif action == "read":
            return self._handle_read(tool_call_id)
        elif action == "clear":
            return self._handle_clear(tool_call_id)
        else:
            return Message(
                role="tool",
                content=f"❌ 不支持的操作: {action}",
                tool_call_id=tool_call_id,
            )

    def _handle_write(self, params: Dict[str, Any], tool_call_id: str) -> Message:
        note = params.get("note", "")
        fact_key = params.get("fact_key", "")
        fact_value = params.get("fact_value", "")

        if note:
            self._state["notes"].append(note)
        if fact_key:
            self._state["key_facts"][fact_key] = fact_value

        return Message(
            role="tool",
            content=f"✅ 已记录。当前笔记数: {len(self._state['notes'])}, "
                    f"关键事实数: {len(self._state['key_facts'])}",
            tool_call_id=tool_call_id,
        )

    def _handle_read(self, tool_call_id: str) -> Message:
        notes = "\n".join(f"- {n}" for n in self._state["notes"]) or "（无）"
        facts = "\n".join(f"- {k}: {v}" for k, v in self._state["key_facts"].items()) or "（无）"
        return Message(
            role="tool",
            content=f"# 会话记忆\n\n## 笔记\n{notes}\n\n## 关键事实\n{facts}",
            tool_call_id=tool_call_id,
        )

    def _handle_clear(self, tool_call_id: str) -> Message:
        self._state["notes"].clear()
        self._state["key_facts"].clear()
        return Message(role="tool", content="✅ 记忆已清空", tool_call_id=tool_call_id)

    # ──── 可选：恢复时验证 ────
    def restore_session_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        """确保恢复的状态包含必要字段。"""
        if not state:
            super().restore_session_state(None)
            return

        validated = dict(state)
        # 确保必要字段存在
        if "notes" not in validated:
            validated["notes"] = []
        if "key_facts" not in validated:
            validated["key_facts"] = {}

        super().restore_session_state(validated)

    # ──── 必须重写：工具参数定义 ────
    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={
                "action": ToolProperty(
                    type="string",
                    description="操作类型：write（写入）、read（读取）、clear（清空）",
                ),
                "note": ToolProperty(
                    type="string",
                    description="要记录的笔记内容，仅在 action=write 时使用",
                ),
                "fact_key": ToolProperty(
                    type="string",
                    description="关键事实的键名，仅在 action=write 时使用",
                ),
                "fact_value": ToolProperty(
                    type="string",
                    description="关键事实的值，仅在 action=write 时使用",
                ),
            },
            required=["action"],
        )
```

## 5. 生命周期全景

```
工具实例化
    │
    └─→ __init__()
        └─→ super().__init__(name, description)
            └─→ 创建 _state_var (ContextVar, default=None)

每次 run() 调用
    │
    ├─→ _get_state()
    │   ├─→ state = _state_var.get()
    │   ├─→ if state is None:   ← 首次访问，懒初始化
    │   │       state = _make_default_state()
    │   │       _state_var.set(state)
    │   └─→ return state
    │
    ├─→ 读写 state["xxx"]（通过 property）
    │
    └─→ return Message(...)

Session 切换时
    │
    ├─ TurnEnd: _save_all_tool_states()
    │   └─→ tool.get_session_state()           ← 基类: copy.deepcopy(_get_state())
    │       └─→ sess.save_tool_state(name, ...)
    │
    └─ TurnStart: _restore_all_tool_states()
        └─→ sess.get_tool_state(name)
            └─→ tool.restore_session_state(state)
                └─→ 子类验证 → super().restore_session_state(validated)
                    └─→ 基类: _state_var.set(copy.deepcopy(state))
```

## 6. 反模式与注意事项

### ❌ 不要重写 `get_session_state()`

```python
# ❌ 错误：绕过基类的 copy.deepcopy，丢失安全保护
def get_session_state(self) -> Dict[str, Any]:
    return {"current_dir": str(self.current_dir)}
```

基类的实现已经包含了 `copy.deepcopy`，你的手写版本可能忘记这一点，导致外部修改影响内部状态。

### ❌ 不要重写 `reset()`

```python
# ❌ 错误：绕过了基类的 ContextVar 原子替换
def reset(self) -> None:
    self.current_dir = self.workspace  # 逐个字段改，有残留风险
```

基类的 `_state_var.set(self._make_default_state())` 是一次性整体替换，确保所有字段回到默认值。

### ❌ 不要在 `__init__` 中缓存状态引用

```python
# ❌ 错误：_state 指向 __init__ 线程的 ContextVar 值，其他线程中仍是旧引用
def __init__(self):
    super().__init__(...)
    self._state = self._get_state()  # 只执行一次！
```

每次访问状态都必须通过 `_get_state()`，它会动态读取当前线程的 ContextVar。

### ❌ 不要在 `restore_session_state` 中跳过 `super()`

```python
# ❌ 错误：逐个字段赋值，如果状态新增了 key，旧 key 残留
def restore_session_state(self, state):
    self.current_dir = Path(state.get("current_dir", self.workspace))
    # 假设以后新增了 shell_env 字段...
    # self.shell_env 没有被重置！因为基类的整体替换被跳过了
```

### ✅ 正确模式速查

| 场景 | 做法 |
|---|---|
| 定义初始状态 | 重写 `_make_default_state()` → `Dict[str, Any]` |
| 读写状态 | 通过 property + `_get_state()` |
| 持久化状态 | 不重写（用基类的 `get_session_state()`） |
| 恢复状态 + 验证 | 重写 `restore_session_state()`，验证后调 `super()` |
| 重置状态 | 不重写（用基类的 `reset()`） |

## 7. 参考实现

本项目的 `TerminalTool`（[src/violet_agents/tools/builtin/terminal_tool.py](../src/violet_agents/tools/builtin/terminal_tool.py)）是可变状态工具的最佳实践参考：

- 只有 `_make_default_state()` 和 `restore_session_state()`（含验证逻辑）两个重写方法
- `current_dir` 通过 property 封装 `_get_state()` 的读写
- 不重写 `get_session_state()` 和 `reset()`，完全依赖基类实现

## 8. 线程安全原理

更多关于 ContextVar 在 Agent 框架中的线程安全原理，参见 [thread-safety.md](thread-safety.md)。
