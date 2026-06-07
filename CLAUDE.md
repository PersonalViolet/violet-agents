# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

violet-agents 是一个基于 OpenAI 兼容 API 的轻量级 Python Agent 框架，默认对接 DeepSeek API。

## 开发命令

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行所有测试
pytest

# 运行单个测试文件
pytest tests/test_agents.py

# 运行单个测试函数
pytest tests/test_agents.py::TestAgent::test_simple_agent_response

# 运行 react_agent 示例（含所有内置工具）
python -m src.violet_agents.agents.react_agent
```

## 架构

### 核心层 (`src/violet_agents/core/`)

- **Agent** (`agent.py`): 抽象基类，定义 `run(input_text) -> Message` 接口，管理消息历史（`deque[Message]`，长度受 `Config.max_history_length` 限制）。子类：`SubAgent` —— 自动从 `SUB_AGENT_LLM_*` 环境变量读取独立的 LLM 配置。
- **VioletAgentsLLM** (`llm.py`): 封装 OpenAI SDK 的 `OpenAI` 客户端。通过 `provider` 参数自动推断 API key / base URL 环境变量名。支持 `deepseek` 和 `modelscope` 两种 provider。`chat()` 方法接受 `list[dict]` 或 `deque[Message]`，自动转换为 OpenAI 格式。
- **Message** (`message.py`): Pydantic 模型，包含 `reasoning_content` 字段（适配 DeepSeek 的思考过程）。`to_openai_dict()` 输出时排除 `timestamp`、`metadata` 等内部字段。`from_chat_completion_message()` 可从 OpenAI 响应构造。
- **Config** (`config.py`): Pydantic 模型，当前仅 `max_history_length`（默认 100）。

### Agent 层 (`src/violet_agents/agents/`)

- **SimpleAgent**: 单轮对话，不调用工具，直接返回 LLM 响应。
- **ReactAgent**: ReAct 循环 —— 轮询 LLM → 执行工具调用 → 回到 LLM，最多 `max_steps` 轮。内部维护 `temp_tools`（运行时发现的临时工具，3 轮未调用自动过期）。

**ReactAgent 钩子系统**：支持 `UserPromptSubmit`、`PreToolCall`、`PostToolCall` 三种事件。内置两个钩子：
- `_handle_search_tools_hook`（PostToolCall）：当 `SearchToolsTool.get` 返回工具 schema 时，自动将其添加到 `temp_tools`。
- `_on_temp_tool_called_hook`（PreToolCall）：更新临时工具的最后调用轮次，清理过期工具。

### 工具系统 (`src/violet_agents/tools/`)

- **Tool** (`base.py`): 抽象基类，子类必须实现 `run(parameters, tool_call_id) -> Message` 和 `get_parameters() -> ToolParameters`。
- **ToolRegistry** (`registry.py`): 管理 `_tools`（普通工具，对外暴露给 LLM）和 `_defer_tools`（延迟工具，不暴露给 LLM，需通过 SearchToolsTool 发现后调用）。执行工具前通过 `ApprovalTool` 做人工审批。
- **ApprovalTool** (`approval_tool.py`): 审批规则 —— `require_approval_tools` 优先于 `auto_approve_tools`，都不匹配时依 `auto_approve_if_no_rules` 决定。
- **DefaultApprovalTool** (`infrastructure/`): 终端交互式审批，`input()` 确认，最多重试 `max_attempts` 次。

**内置工具**（`builtin/`）：
- `WeatherTool`: 模拟天气查询（返回固定数据）。
- `TerminalTool`: 执行 shell 命令，沙箱限制在 `workspace` 目录内，禁止危险命令（rm、shutdown 等）。
- `SkillsTool`: 从多个路径发现 `SKILL.md` 文件，支持 `list`（列出）和 `load`（加载内容）操作。
- `SearchToolsTool`: 运行时工具搜索，支持 `embedding`（TF-IDF 向量化）和 `subAgent`（子 Agent 判断）两种策略。`get` 操作从 `_defer_tools` 中获取完整工具 schema。
- `TodoWriteTool`: 待办事项工具（尚未完成 `run` 方法实现）。

### 协议层 (`src/violet_agents/protocols/mcp/`)

- MCP 客户端（早期开发阶段），基于 `fastmcp`，当前仅有框架性代码。

## 环境变量

```
DEEPSEEK_API_KEY=        # DeepSeek API 密钥
DEEPSEEK_BASE_URL=       # DeepSeek API 地址
SUB_AGENT_LLM_API_KEY=   # 子 Agent 的 LLM API 密钥（独立配置）
SUB_AGENT_LLM_BASE_URL=  # 子 Agent 的 LLM API 地址
SUB_AGENT_LLM_MODEL=     # 子 Agent 的模型名称（如 deepseek-v4-flash）
```

通用回退变量：`LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL_NAME` —— 当 provider 专用变量不存在时使用。

## 关键设计约定

- 所有工具的 `run()` 必须返回 `Message` 对象（`role="tool"`，包含 `tool_call_id`）。
- `ToolRegistry.get_openai_tools()` 只返回 `_tools` 中的工具，`_defer_tools` 中的工具不暴露给 LLM。
- ReactAgent 的 `run()` 在达到 `max_steps` 时返回错误提示而非抛异常。
- `_defer_tools` 中的"懒加载"工具不会被 LLM 直接看到，只能通过 `SearchToolsTool` 搜索发现并获取后再调用。
