# Agent 线程安全机制 — ContextVar 实现

## 1. 问题背景

在改造之前，`Agent` 基类将所有"当前活跃 Session"相关的状态直接存储在实例属性中：

```python
class Agent(ABC):
    def __init__(self, ...):
        self._active_session: Optional["Session"] = None   # ← 实例级属性
        self._history = deque[Message](...)                 # ← 实例级属性
        self._default_session_id: Optional[str] = None     # ← 实例级属性
        self._sessions: Dict[str, "Session"] = {}           # ← 实例级属性
```

当多个线程共享同一个 Agent 实例时，所有线程竞争同一个 `_active_session`：

```
线程 A: agent.run("你好", session_id="user-A")
        └─→ self._active_session = Session("user-A")   # 设置为 A

线程 B: agent.run("你好", session_id="user-B")
        └─→ self._active_session = Session("user-B")   # 覆盖为 B！

线程 A: # 继续执行，但 _active_session 已经变成 B 了
        └─→ sess._history.append(...)  # 写入错误的 Session！
```

这导致三类严重问题：

| 问题 | 后果 |
|---|---|
| **消息历史串话** | 线程 A 的回复写入线程 B 的 Session |
| **工具状态错乱** | 终端工作目录、临时工具状态恢复到错误的上下文 |
| **钩子触发错位** | PostToolCall 钩子将工具结果写入不对的 Session |

代码中曾有 TODO 注释（`agent.py:18`）承认此问题：

> "目前会话隔离仅在单线程下安全。未来版本可能会引入更复杂的并发控制机制以支持多线程/多进程环境下的会话隔离和状态管理。"

## 2. 解决方案：ContextVar

### 2.1 什么是 ContextVar？

`contextvars.ContextVar` 是 Python 3.7+ 标准库提供的**上下文变量**机制。它的核心特性是：**每个执行上下文（线程、asyncio Task）持有变量的独立副本**。

```
                    Agent 实例（唯一）
               ┌─────────────────────────┐
               │  name, llm, config       │ ← 不变，所有线程共享
               │  _sessions (Dict)        │ ← 共享池，通过 RLock 保护
               │                         │
               │  _active_session_var ────┼──→ 线程A: Session("user-A")
               │                         │    线程B: Session("user-B")
               │  _history_var ───────────┼──→ 线程A: deque([...])
               │                         │    线程B: deque([...])
               │  _default_session_id_var ┼──→ 线程A: "abc123"
               │                         │    线程B: "def456"
               └─────────────────────────┘
```

每个线程调用 `_active_session_var.get()` 时拿到的是**自己**的值，互不干扰。

### 2.2 为什么选择 ContextVar？

| 方案 | 优点 | 缺点 |
|---|---|---|
| `threading.local` | 天然线程隔离 | 不支持 asyncio；无法被子任务继承 |
| 全局锁 `Lock` | 简单粗暴 | 串行化，并发性能差 |
| 每个线程创建 Agent | 天然隔离 | 实例成本高（LLM 客户端、工具注册表都是重量级对象） |
| **ContextVar** | 线程+asyncio 双支持；可被子 context 继承（`copy_context`） | 需要 Python 3.7+ |

选择 ContextVar 的关键原因：**零侵入 + 零性能损失 + asyncio 兼容**。对外 API 完全不变，只是底层存储方式从实例属性换成了上下文变量。

### 2.3 核心实现

在 `Agent.__init__()` 中创建 ContextVar，每个变量用 `id(self)` 保证 key 名唯一：

```python
import contextvars
import threading

class Agent(ABC):
    def __init__(self, ...):
        # 每实例的 ContextVar（用实例 id 保证唯一 key 名）
        self._active_session_var = contextvars.ContextVar(
            f"_active_session_{id(self)}", default=None
        )
        self._history_var = contextvars.ContextVar(
            f"_history_{id(self)}",
            default=deque[Message](maxlen=...)
        )
        self._default_session_id_var = contextvars.ContextVar(
            f"_default_session_id_{id(self)}", default=None
        )

        # Session 池（全局共享，RLock 保护）
        self._sessions: Dict[str, "Session"] = {}
        self._sessions_lock = threading.RLock()
```

封装 6 个内部辅助方法，隔离 ContextVar 的读写细节：

```python
def _get_active_session(self) -> Optional["Session"]:
    """获取当前执行上下文的活跃 Session。"""
    return self._active_session_var.get()

def _set_active_session(self, sess: Optional["Session"]) -> None:
    """设置当前执行上下文的活跃 Session。"""
    self._active_session_var.set(sess)

def _get_history(self) -> deque[Message]:
    """获取当前执行上下文的默认消息历史（向后兼容）。"""
    return self._history_var.get()

def _get_default_session_id(self) -> Optional[str]:
    return self._default_session_id_var.get()

def _set_default_session_id(self, sid: Optional[str]) -> None:
    self._default_session_id_var.set(sid)
```

所有原本直接访问 `self._active_session` 的代码改为调用这些方法：

```python
# 改造前
def _save_all_tool_states(self):
    sess = self._active_session       # ← 多线程不安全
    if not sess:
        return
    ...

# 改造后
def _save_all_tool_states(self):
    sess = self._get_active_session()  # ← 每个线程拿到自己的值
    if not sess:
        return
    ...
```

### 2.4 Session 池的并发保护

`_sessions` 字典是所有线程共享的 Session 存储池，必须用锁保护：

```python
def create_session(self, session_id=None, **kwargs):
    sess = Session(...)
    self._trigger_agent_hooks("SessionInit", sess)
    with self._sessions_lock:                     # ← 临界区
        self._sessions[sess.session_id] = sess
    if self.config.auto_cleanup_sessions:
        self._cleanup_expired_sessions()
    return sess.session_id

def get_session(self, session_id: str):
    with self._sessions_lock:                     # ← 读也要加锁
        return self._sessions.get(session_id)     #   dict 并发读写会抛 RuntimeError

def destroy_session(self, session_id):
    with self._sessions_lock:
        if session_id not in self._sessions:
            return False
        active = self._get_active_session()
        if active and active.session_id == session_id:
            self._save_all_tool_states()
            self._set_active_session(None)
        del self._sessions[session_id]
    return True
```

**为什么用 `RLock` 而不是 `Lock`？**

`_cleanup_expired_sessions()` 在持有锁时可能调用 `destroy_session()`，后者也需要加锁。`RLock` 允许同一线程重入，避免死锁。

## 3. 变更范围

| 文件 | 改动 |
|---|---|
| `core/agent.py` | 3 个 ContextVar、6 个辅助方法、1 个 RLock、~30 处引用替换 |
| `agents/react_agent.py` | 5 处 `self._active_session` → `self._get_active_session()` |
| `core/session.py` | `_SessionContext` 中 2 处适配 |

公开 API **零变更**：`run()`、`create_session()`、`switch_session()`、`session()` 等方法签名和行为完全不变。

## 4. 使用示例

### 4.1 基本多线程用法

```python
import threading
from violet_agents import ReactAgent, VioletAgentsLLM, ToolRegistry

agent = ReactAgent(
    name="MultiUserBot",
    llm=VioletAgentsLLM(provider="deepseek"),
    tool_registry=ToolRegistry(),
)

def handle_user(user_id: str, message: str):
    """每个用户请求在独立线程中处理"""
    response = agent.run(message, session_id=user_id)
    print(f"[{user_id}] reply: {response.content}")

# 两个用户同时发送消息，完全隔离
t1 = threading.Thread(target=handle_user, args=("alice", "你好，我叫 Alice"))
t2 = threading.Thread(target=handle_user, args=("bob",   "你好，我叫 Bob"))
t1.start(); t2.start()
t1.join();  t2.join()

# Alice 记得自己叫什么
response = agent.run("我叫什么名字？", session_id="alice")
print(response.content)  # "你叫 Alice"

# Bob 也记得
response = agent.run("我叫什么名字？", session_id="bob")
print(response.content)  # "你叫 Bob"
```

### 4.2 Web 服务集成

```python
from flask import Flask, request
from violet_agents import ReactAgent, VioletAgentsLLM

app = Flask(__name__)

# 全局共享一个 Agent 实例
agent = ReactAgent(
    name="WebBot",
    llm=VioletAgentsLLM(provider="deepseek"),
    system_prompt="你是一个 Web 助手。",
)

@app.route("/chat", methods=["POST"])
def chat():
    user_id = request.json["user_id"]
    message = request.json["message"]

    # Flask 默认每个请求一个线程，ContextVar 自动隔离
    response = agent.run(message, session_id=user_id)
    return {"reply": response.content}

if __name__ == "__main__":
    app.run(threaded=True)  # 多线程模式，每个请求独立上下文
```

### 4.3 ThreadPoolExecutor 并发

```python
from concurrent.futures import ThreadPoolExecutor
from violet_agents import ReactAgent, VioletAgentsLLM

agent = ReactAgent(
    name="BatchBot",
    llm=VioletAgentsLLM(provider="deepseek"),
)

# 100 个用户并发提问，线程池限制 10 个并发
user_messages = {f"user-{i}": f"帮我分析数据第 {i} 号" for i in range(100)}

def process(user_id, message):
    return agent.run(message, session_id=user_id)

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {
        executor.submit(process, uid, msg): uid
        for uid, msg in user_messages.items()
    }
    for future in futures:
        uid = futures[future]
        result = future.result()
        print(f"[{uid}] done")
```

### 4.4 asyncio 协程（未来兼容）

ContextVar 天然支持 asyncio——每个 Task 有独立的 context：

```python
import asyncio

async def async_user(agent, user_id, message):
    # asyncio.to_thread 会将 ContextVar 上下文带入子线程
    return await asyncio.to_thread(
        agent.run, message, session_id=user_id
    )

async def main():
    agent = ReactAgent(...)

    # 3 个协程并发
    tasks = [
        async_user(agent, "alice", "你好"),
        async_user(agent, "bob", "你好"),
        async_user(agent, "carol", "你好"),
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r.content)

asyncio.run(main())
```

### 4.5 线程间通过 Session 池共享状态

虽然每个线程的 active session 是隔离的，但可以通过 session_id 在不同线程间访问同一 Session：

```python
import threading, time

def init_conversation():
    """线程 A：初始化对话"""
    agent.run("我们来讨论 Python 的 GIL 问题", session_id="shared-topic")
    agent.run("请列出 GIL 的三个主要影响", session_id="shared-topic")

def continue_conversation():
    """线程 B：接着线程 A 的话题继续"""
    time.sleep(2)  # 等待线程 A 初始化完成
    # 直接指定 session_id 即可读取共享 Session 的历史
    history = agent.get_history(session_id="shared-topic")
    print(f"之前的消息数: {len(history)}")
    # 在另一个线程中继续对话
    response = agent.run("基于之前的讨论，给出优化建议", session_id="shared-topic")

ta = threading.Thread(target=init_conversation)
tb = threading.Thread(target=continue_conversation)
ta.start(); tb.start()
ta.join();  tb.join()
```

> **注意**：多线程操作同一 Session 时，Session 内部的 dataclass 字段（如 `_history`）没有锁保护。如果多个线程同时写同一个 Session，需要调用方加锁。推荐用法是**每个线程操作自己的 Session**，只在需要共享历史时通过 `session_id` 读取。

## 5. 内部原理

### 5.1 ContextVar 的生命周期

```python
# ContextVar 绑定在 Agent 实例上，但值是 per-context 的
var = contextvars.ContextVar("example", default="default")

def worker():
    print(var.get())   # "default"
    var.set("worker-1")
    print(var.get())   # "worker-1"
    # 线程结束时，值自动被 GC 回收

# 主线程
var.set("main")
print(var.get())       # "main"

t = threading.Thread(target=worker)
t.start()
t.join()

print(var.get())       # 仍然是 "main"，线程变更不影响
```

### 5.2 `run()` 方法的完整执行流

```
agent.run("你好", session_id="user-123")
│
├─ 1. _resolve_session("user-123")
│     └─ _sessions_lock 检查 → 存在则返回 Session 对象
│
├─ 2. switch_session("user-123")
│     ├─ _get_active_session()        ← ContextVar，当前线程拿到的值
│     ├─ 如果 old_session != None:
│     │     _save_all_tool_states()    ← 保存工具状态到 old_session
│     ├─ _set_active_session(sess)     ← ContextVar，设为当前线程的 active
│     └─ _restore_all_tool_states()    ← 从新 session 恢复工具状态
│
├─ 3. _trigger_session_hooks("TurnStart", input_text, sess=sess)
├─ 4. do_run(input_text, session=sess)  ← 子类实现
├─ 5. _trigger_session_hooks("TurnEnd", response, sess=sess)
│
└─ 6. return response
```

在步骤 2 中，`_get_active_session()` 和 `_set_active_session()` 操作用的是当前线程的 ContextVar 副本。线程 B 的并发调用完全不会影响线程 A 的这套流程。

### 5.3 为什么不使用 threading.local？

```python
# threading.local 的行为：
mydata = threading.local()
mydata.x = 1  # 线程 A 中

def worker():
    print(mydata.x)  # AttributeError: 'thread._local' object has no attribute 'x'
    # threading.local 不会继承默认值，且不支持 asyncio Task 之间的上下文继承
```

ContextVar 相比之下：
- 支持 `default=` 参数，不存在的上下文返回默认值
- 支持 `contextvars.copy_context()` —— 可以显式地将当前 context 传递给子线程
- 支持 asyncio —— 每个 `asyncio.Task` 有独立的 context，且子 Task 可以继承父 Task 的 context

## 6. 线程安全边界

| 组件 | 线程安全性 | 说明 |
|---|---|---|
| `Agent._active_session` | ✅ ContextVar 隔离 | 每个线程独立 |
| `Agent._history` (默认) | ✅ ContextVar 隔离 | 每个线程独立 |
| `Agent._default_session_id` | ✅ ContextVar 隔离 | 每个线程独立 |
| `Agent._sessions` Dict | ✅ RLock 保护 | 全局共享池，并发安全 |
| `Tool._state_var` | ✅ ContextVar 隔离 | 每个线程独立持有工具状态副本 |
| `Session._history` | ⚠️ 调用方负责 | 多线程操作同一 Session 需自行加锁 |
| `Session._tool_state` | ⚠️ 调用方负责 | 同上 |
| `ToolRegistry` | ✅ 只读安全 | 工具注册后不变；`execute_tool` 每次创建新 Message |
| `Agent._agent_hooks` | ⚠️ 不保护 | 假设在初始化阶段注册，运行时不变 |

### 6.1 Tool 状态的 ContextVar 机制

工具的可变状态同样通过 ContextVar 隔离。`Tool` 基类在 `__init__` 中创建一个 `_state_var`，每个执行上下文持有独立的工具状态 dict：

```python
class Tool(ABC):
    def __init__(self, name, description):
        self.name = name
        self.description = description
        self._state_var = contextvars.ContextVar(
            f"tool_state_{name}_{id(self)}", default=None
        )

    def _get_state(self) -> Dict[str, Any]:
        """懒初始化当前上下文的状态。"""
        state = self._state_var.get()
        if state is None:
            state = self._make_default_state()
            self._state_var.set(state)
        return state
```

有状态工具（如 `TerminalTool`）通过 `_make_default_state()` 定义默认状态，通过 property 读写 `_get_state()` 中的字段：

```python
class TerminalTool(Tool):
    def _make_default_state(self):
        return {"current_dir": str(self.workspace)}

    @property
    def current_dir(self) -> Path:
        return Path(self._get_state()["current_dir"])  # ← 每线程独立
```

**数据流**：tool.get_session_state() → copy.deepcopy(ContextVar) → sess.save_tool_state() → (数据库/Redis)

```
线程 A                          Session A                    线程 B
───────                         ─────────                    ───────
current_dir = /home/alice       _tool_state: {alice状态}      current_dir = /home/bob
  ↓                                                          ↓
get_session_state() → {"current_dir": "/home/alice"}   get_session_state() → {"current_dir": "/home/bob"}
  ↓                                                          ↓
sess.save_tool_state("terminal", ...)                  sess.save_tool_state("terminal", ...)
```

### 6.2 为什么 ContextVar 默认值用 None 而非可变 dict？

```python
# ❌ 错误：所有上下文共享同一个 dict 对象
var = contextvars.ContextVar("x", default={})

# ✅ 正确：懒初始化，每个上下文首次访问时创建独立 dict
var = contextvars.ContextVar("x", default=None)
state = var.get()
if state is None:
    state = {"fresh": True}
    var.set(state)
```

ContextVar 的 `default` 参数是**在模块加载时求值一次**的。如果用 `{}`，所有线程/协程的初始值指向同一个 dict 对象，mutate 操作互相影响。`None` + 懒初始化避免了这个问题。

## 7. 最佳实践

1. **Agent 实例全局共享**：创建一个 Agent 实例，所有线程/协程共用，通过 `session_id` 区分用户。
2. **每个用户一个 Session**：不要在多线程间共享同一个 `session_id` 进行**写操作**。如果需要共享历史，仅在读写分离的场景下操作。
3. **长连接场景**：使用 `session_default_ttl` 自动清理过期会话，防止内存泄漏。
4. **测试中使用 mock LLM**：避免测试依赖外部 API，参考：
   ```python
   llm = VioletAgentsLLM(
       api_key="sk-test",
       base_url="https://test.example.com",
       model="test-model"
   )
   ```
5. **asyncio 场景**：ContextVar 原生支持 asyncio Task 隔离，`asyncio.to_thread()` 会保留 context，`asyncio.create_task()` 会继承父 context（可通过 `contextvars.copy_context()` 显式控制）。

## 8. 相关文档

- [可变状态工具开发指南](mutable-state-tool-guide.md) —— 如何创建线程安全的有状态工具，含完整示例和反模式说明。
