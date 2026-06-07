"""TerminalTool - 命令行工具


"""

from typing import Any, Dict, List, Optional
import subprocess
import os
from pathlib import Path
import shlex
from ..base import Tool, ToolParameters, ToolProperty
from ...core.message import Message
import platform

# 禁止的危险命令黑名单 - 按操作系统分类
FORBIDDEN_DANGEROUS_COMMANDS = {
    'windows': {
        # 文件删除命令
        'del', 'erase', 'rd', 'rmdir', 'deltree',
        # 批量删除或格式化
        'format', 'diskpart',
        # 系统关机命令
        'shutdown', 'restart', 'poweroff', 'logoff'
    },
    'linux': {
        # 危险的删除命令
        'rm', 
        # 格式化命令
        'mkfs', 'dd',
        # 系统关机命令
        'shutdown', 'halt', 'reboot', 'poweroff'
    },
    'macos': {
        # 危险的删除命令
        'rm',
        # 系统关机命令
        'shutdown', 'reboot', 'halt'
    }
}

TOOL_NAME = "terminal"
TOOL_DESCRIPTION = "执行命令行指令，获取输出结果"
class TerminalTool(Tool):
    """TerminalTool - 命令行工具
    绕过人工审批，直接执行命令行指令，获取输出结果
    """

    def __init__(
            self,
            workspace: str = ".",
            timeout: int = 30,
            max_output_size: int = 10 * 1024 * 1024,  # 10 MB
            allow_cd: bool = True):
        super().__init__(
            name=TOOL_NAME,
            description=TOOL_DESCRIPTION
        )

        self.workspace = Path(workspace).resolve()
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.allow_cd = allow_cd
        self.os_type = platform.system().lower()
        # 当前工作目录（相对于workspace）
        self.current_dir = self.workspace

        # 确保工作目录存在
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        """执行命令行指令，获取输出结果"""
        command = parameters.get("command", "")
        if not command:
            return Message(role="tool", content="❌ 未提供要执行的命令", tool_call_id=tool_call_id)

        # 检查是否为 cd 命令
        parts = shlex.split(command)
        if parts[0] == "cd":
            return Message(role="tool", content=self._handle_cd(parts), tool_call_id=tool_call_id)

        # 检查是否为危险命令
        if parts[0] in FORBIDDEN_DANGEROUS_COMMANDS.get(self.os_type, []):
            return Message(role="tool", content=f"❌ 禁止使用危险命令: {parts[0]}", tool_call_id=tool_call_id)

        # 执行命令
        output = self._execute_command(command)
        return Message(role="tool", content=output, tool_call_id=tool_call_id)

    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={
                "command": ToolProperty(
                    type="string",
                    description=f"该主机系统：{self.os_type}，要执行的命令，禁止使用以下危险命令：{', '.join(FORBIDDEN_DANGEROUS_COMMANDS.get(self.os_type, []))}"
                )
            },
            required=["command"]
        )

    def _handle_cd(self, parts: List[str]) -> str:
        """处理 cd 命令"""
        if not self.allow_cd:
            return "❌ cd 命令已禁用"
        
        if len(parts) < 2:
            # cd 无参数，返回当前目录
            return f"当前目录: {self.current_dir}"
        
        target_dir = parts[1]
        
        # 处理相对路径
        if target_dir == "..":
            new_dir = self.current_dir.parent
        elif target_dir == ".":
            new_dir = self.current_dir
        elif target_dir == "~":
            new_dir = self.workspace
        else:
            new_dir = (self.current_dir / target_dir).resolve()
        
        # 检查是否在工作目录内
        try:
            new_dir.relative_to(self.workspace)
        except ValueError:
            return f"❌ 不允许访问工作目录外的路径: {new_dir}"
        
        # 检查目录是否存在
        if not new_dir.exists():
            return f"❌ 目录不存在: {new_dir}"
        
        if not new_dir.is_dir():
            return f"❌ 不是目录: {new_dir}"
        
        # 更新当前目录
        self.current_dir = new_dir
        return f"✅ 切换到目录: {self.current_dir}"

    def _execute_command(self, command: str) -> str:
        """执行命令"""
        try:
            # 在当前目录下执行命令
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.current_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=os.environ.copy()
            )
            
            # 合并标准输出和标准错误
            output = f"{self.current_dir}> {command}\n\n"
            output += result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            
            # 检查输出大小
            if len(output) > self.max_output_size:
                output = output[:self.max_output_size]
                output += f"\n\n⚠️ 输出被截断（超过 {self.max_output_size} 字节）"
            
            # 添加返回码信息
            if result.returncode != 0:
                output = f"⚠️ 命令返回码: {result.returncode}\n\n{output}"
            
            return output if output else "✅ 命令执行成功（无输出）"
            
        except subprocess.TimeoutExpired:
            return f"❌ 命令执行超时（超过 {self.timeout} 秒）"
        except Exception as e:
            return f"❌ 命令执行失败: {e}"

    def get_session_state(self) -> Dict[str, Any]:
        return {"current_dir": str(self.current_dir)}

    def restore_session_state(self, state: Optional[Dict[str, Any]] = None) -> None:
        if not state:
            self.current_dir = self.workspace
            return
        saved_dir = state.get("current_dir")
        if saved_dir:
            restored = Path(saved_dir)
            try:
                restored.relative_to(self.workspace)
            except ValueError:
                restored = self.workspace
            if restored.exists() and restored.is_dir():
                self.current_dir = restored
            else:
                self.current_dir = self.workspace
        else:
            self.current_dir = self.workspace

    def reset(self) -> None:
        self.current_dir = self.workspace