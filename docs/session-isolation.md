# Session 会话隔离架构

## 1. 问题背景

在没有会话隔离之前，violet_agents 框架的 Agent 实例将**所有可变状态直接存储在实例字段**中：

- 消息历史（`_history`）
- 工具状态（如 `TerminalTool.current_dir`）
- 临时工具列表（`temp_tools`、`temp_tools_names`）
- 钩子函数（`hooks`）
- 轮次计数器（`current_round`）

这导致以下问题：

1. **状态泄漏**：同一个 Agent 实例服务多个用户时，前一个对话的历史和工具状态会污染后续对话。
2. **无法并发**：多个对话必须串行执行，因为状态是共享的。
3. **不可持久化**：无法单独保存某个对话的状态并在之后恢复。
4. **`_history` 类型 bug**：`ReactAgent.run()` 将 `self._history` 从 `deque` 直接赋值为 `list`，导致后续调用 `add_message()` 失败。

## 2. 设计理念

```
Agent（不变配置）          Session（可变状态）
┌─────────────────┐       ┌─────────────────┐
│ name             │       │ session_id       │
│ llm              │  1:N  │ _history         │
│ system_prompt    │──────▶│ temp_tools       │
│ tool_registry    │       │ _tool_state      │
│ max_steps        │       │ hooks            │
│ config           │       │ current_round    │
└─────────────────┘       │ metadata         │
                           └─────────────────┘
```

**核心原则**：Agent 实例持有**创建成本高、可共享**的配置（LLM 客户端、工具注册表、系统提示词），Session 持有**每个对话独有**的可变状态。一个 Agent 可创建无限个 Session。

### 为什么不用 Agent.clone()？

`clone()` 模式会复制 LLM 客户端等重量级对象，而且无法提供统一的 session 管理（列表、切换、TTL 清理）。Session 模式更轻量、更灵活。

## 3. 核心 API

### 3.1 创建和使用 Session

```python
from violet_agents import ReactAgent, VioletAgentsLLM, ToolRegistry

agent = ReactAgent(
    name="MyAgent",
    llm=VioletAgentsLLM(provider="deepseek"),
    tool_registry=ToolRegistry(),
)

# 方式一：指定 session_id 运行（自动创建 session）
response = agent.run("你好", session_id="user-123")

# 方式二：上下文管理器
with agent.session("user-123"):
    response = agent.run("继续刚才的话题")

# 方式三：手动管理
agent.create_session("user-456")
agent.switch_session("user-456")
response = agent.run("你好")
agent.switch_session("user-123")  # 切回之前的 session
agent.deactivate_session() # 清理_active_session
# 当你不需要使用session功能隔离会话时，确保你的agent实例一直是如下调用方式或在前面使用了agent.deactivate_session()，使得agent实例使用默认session
agent.run("你好")
```

### 3.2 Session 生命周期

```python
# 创建
sid = agent.create_session("my-session")
# 创建时可指定 session_id，不指定则自动生成 12 位 hex

# 查询
sess = agent.get_session(sid)
all_ids = agent.list_sessions()

# 持久化
data = agent.save_session(sid)      # 导出为 JSON 安全 dict
new_sid = agent.restore_session(data)  # 从 dict 恢复

# 销毁
agent.destroy_session(sid)

# 上下文管理器（自动切换和恢复）
with agent.session(sid) as sess:
    agent.run("你好")
```

### 3.3 Session 配置

在 `Config` 中配置 session 行为：

```python
from violet_agents import Config

config = Config(
    max_history_length=100,       # 每个 session 的最大历史消息数
    max_sessions=100,             # 最多同时存在的 session 数
    session_default_ttl=3600,     # session 默认过期时间（秒），None = 永不过期
    auto_cleanup_sessions=True,   # 是否在 create_session 时自动清理过期 session
)
```

### 3.4 向后兼容

不创建任何 session 直接调用 `agent.run()` 时，框架会自动创建一个**默认 session**。现有代码无需修改即可正常运行：

```python
# 旧代码仍然可以工作
agent = SimpleAgent(name="Bot", llm=llm)
agent.run("你好")  # 自动使用默认 session
```

## 4. 工具状态隔离

有状态的工具（如 `TerminalTool`）需要实现三个方法来支持 session 隔离：

```python
from violet_agents.tools.base import Tool

class MyStatefulTool(Tool):
    def get_session_state(self) -> Dict[str, Any]:
        """返回当前状态的 JSON 安全快照"""
        return {"my_state": self.current_value}

    def restore_session_state(self, state: Dict[str, Any]) -> None:
        """从快照恢复状态"""
        self.current_value = state.get("my_state", self.default_value)

    def reset(self) -> None:
        """重置到初始状态"""
        self.current_value = self.default_value
```

**典型实现：TerminalTool**

```python
# 在 terminal_tool.py 中
def get_session_state(self):
    return {"current_dir": str(self.current_dir)}

def restore_session_state(self, state):
    saved_dir = state.get("current_dir")
    if saved_dir:
        restored = Path(saved_dir)
        # 验证仍在 workspace 沙箱内
        try:
            restored.relative_to(self.workspace)
        except ValueError:
            restored = self.workspace
        if restored.exists() and restored.is_dir():
            self.current_dir = restored

def reset(self):
    self.current_dir = self.workspace
```

**工作原理**：每次 `agent.run()` 开始前和 `switch_session()` 时，框架自动调用 `restore_session_state()`；每次 `agent.run()` 结束后，自动调用 `get_session_state()` 保存快照。开发者无需手动管理。

## 5. 序列化与持久化

Session 支持完整的序列化/反序列化：

```python
# 保存
data = agent.save_session("user-123")
# data 是一个 JSON 安全的 dict，可存入数据库、文件、Redis 等
import json
with open("session.json", "w") as f:
    json.dump(data, f)

# 恢复（可以在另一个 Agent 实例上）
with open("session.json") as f:
    data = json.load(f)
new_sid = agent.restore_session(data)
agent.switch_session(new_sid)
response = agent.run("我们上次聊到哪了？")
```

**序列化内容**：session_id、历史消息、工具状态、临时工具、元数据、时间戳。

**不序列化**：钩子回调（函数对象无法 JSON 序列化）。恢复后默认钩子会自动重新注册，自定义钩子需手动重新注册。

## 6. 完整示例

### 6.1 多用户聊天机器人

```python
from violet_agents import ReactAgent, VioletAgentsLLM, ToolRegistry, Config

# 初始化（只需一次）
agent = ReactAgent(
    name="ChatBot",
    llm=VioletAgentsLLM(provider="deepseek"),
    system_prompt="你是一个友好的助手。",
    tool_registry=ToolRegistry(),
    config=Config(max_history_length=50, max_sessions=1000),
)

# 用户 Alice 的第一轮对话
response = agent.run("你好，我叫 Alice", session_id="alice")
print(response.content)

# 用户 Bob 的第一轮对话（完全隔离）
response = agent.run("你好，我叫 Bob", session_id="bob")
print(response.content)

# 切回 Alice，她记得自己的名字
response = agent.run("我叫什么名字？", session_id="alice")
print(response.content)  # "你叫 Alice"

# 切回 Bob
response = agent.run("我叫什么名字？", session_id="bob")
print(response.content)  # "你叫 Bob"
```

### 6.2 持久化对话

```python
import json

# 第一轮对话
agent.run("请帮我分析这份数据...", session_id="analysis-001")
agent.run("之前的分析有什么问题？", session_id="analysis-001")

# 保存
data = agent.save_session("analysis-001")
with open("analysis_001.json", "w") as f:
    json.dump(data, f, ensure_ascii=False)

# ... 第二天 ...

# 恢复并继续
with open("analysis_001.json") as f:
    data = json.load(f)
agent.restore_session(data)
agent.run("继续昨天的分析，我发现了新的问题...", session_id="analysis-001")
```

## 7. 内部实现

### 7.1 委托模式

Agent 的 `add_message`、`get_history`、`clear_history` 使用委托模式：

```python
def add_message(self, message):
    if self._active_session:
        self._active_session.add_message(message)  # 委托给 session
    else:
        self._history.append(message)  # 旧版路径（向后兼容）
```

### 7.2 工具状态保存/恢复

- `_save_tool_states_impl()`：遍历 ToolRegistry 中所有工具的 `get_session_state()`，存入 session 的 `_tool_state`。
- `_restore_tool_states_impl()`：从 session 的 `_tool_state` 读取快照，调用 `restore_session_state()` 恢复。

触发时机：
- `switch_session()` 切换前 → 保存当前 session 的工具状态
- `switch_session()` 切换后 → 恢复目标 session 的工具状态
- `run()` 开始前 → `_activate_session()` → 恢复工具状态
- `run()` 结束后 → `_deactivate_session()` → 保存工具状态
- `destroy_session()` 销毁 active session 前 → 保存工具状态

### 7.3 Hook 传播

ReactAgent 初始化时注册的钩子（`_handle_search_tools_hook`、`_on_temp_tool_called_hook`）存储在 `_default_hooks`。每个新 session 创建时，`_init_session_hooks()` 自动复制到 session 的 `hooks`。

`register_hooks()` 不指定 `session_id` 时，钩子同时写入 `_default_hooks` 和所有已存在的 session。

### 7.4 修复的 Bug

原 `ReactAgent.run()` 中的 `self._history = messages` 将 deque 替换为 list，修复后：

```python
sess._history = deque(messages, maxlen=sess.max_history_length if sess.max_history_length > 0 else None)
```

## 8. 最佳实践

1. **每个用户/对话一个 session**：使用用户 ID 或对话 ID 作为 session_id。
2. **设置合理的 TTL**：对于 Web 应用，设置 `session_default_ttl=1800`（30 分钟）自动清理闲置对话。
3. **利用序列化**：将 session 存储到 Redis/数据库，实现跨进程、跨重启的对话持久化。
4. **工具开发者**：有状态工具务必实现 `get_session_state`/`restore_session_state`/`reset` 三个方法。
5. **测试中使用 mock LLM**：参考 `tests/test_session.py` 中的 `_DummyLLM` 模式，避免测试依赖外部 API。
6. **多线程安全**：Agent 实例使用 `contextvars` 实现线程级 session 隔离，多线程可安全共享同一 Agent 实例。详见 [thread-safety.md](thread-safety.md)。
