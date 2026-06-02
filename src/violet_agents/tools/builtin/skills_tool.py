from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel
from ..base import Tool, ToolParameters, ToolProperty
from ...core.message import Message, MessageRole
from pathlib import Path
import re


def _parse_skill_frontmatter(content: str) -> Dict[str, str]:
    """Extract YAML frontmatter from SKILL.md."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        return {}

    frontmatter = {}
    for line in match.group(1).split("\n"):
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()

    return frontmatter

def _get_skill_body(content: str) -> str:
    """Extract the body (after frontmatter) from a SKILL.md file."""
    match = re.match(r"^---\s*\n.*?\n---\s*\n", content, re.DOTALL)
    if match:
        return content[match.end():]
    return content

class SkillsTool(Tool):
    def __init__(self,
                 builtin_path: Optional[Path] = None,
                 global_path: Optional[Path] = None,
                 project_path: Optional[Path] = None,
                 extra_path: Optional[Path] = None):
        """
        初始化技能工具，接受多个路径用于发现技能文件。
        目前仅支持utf-8编码的SKILL.md文件

        优先级：project_path > extra_path > global_path > builtin_path
        Args:
            builtin_path (Optional[Path]): 内置技能路径
            global_path (Optional[Path]): 全局技能路径
            project_path (Optional[Path]): 项目技能路径
            extra_path (Optional[Path]): 额外技能路径

        """
        super().__init__(
            name="skills",
            description="获取当前可用的技能列表，以及加载指定技能的内容。"
        )
        self.skill_paths: List[Path] = []
        if builtin_path:
            self.skill_paths.append(builtin_path)
        if global_path:
            self.skill_paths.append(global_path)
        else:
            # 默认全局路径为 ~/.violet_agents/skills
            default_global_path = Path.home() / ".violet_agents" / "skills"
            if default_global_path.exists():
                self.skill_paths.append(default_global_path)
        if extra_path:
            self.skill_paths.append(extra_path)
        if extra_path:
            self.skill_paths.append(extra_path)
        if project_path:
            self.skill_paths.append(project_path)

    def _discover_skills(self) -> Dict[str, Tuple]:
        """
        发现skill_paths中所有技能文件，并解析其SKILL.md获取技能信息。
        采用utf-8编码读取SKILL.md文件，解析前置数据（如技能名称、描述等），并返回一个技能字典。

        返回数据示例：
        {
            "skill名称": ("/path/to/SKILL.md", {"name": "skill名称", "description": "技能描述"}),
            ...
        }
        """

        skills: Dict[str, Tuple] = {}

        for skills_path in self.skill_paths:
            if not skills_path.exists():
                continue
            for item in skills_path.iterdir():
                if item.is_dir():
                    skill_md = item / "SKILL.md"
                    if skill_md.exists():
                        try:
                            content = skill_md.read_text(encoding="utf-8")
                            frontmatter = _parse_skill_frontmatter(content)
                            skill_name = frontmatter.get("name", item.name)
                            skills[skill_name] = (skill_md, frontmatter)
                        except Exception as e:
                            skills[item.name] = (
                                skill_md,
                                {"name": item.name, "description": "读取skill时发生错误" }
                            )
        return skills

    def get_system_prompt_section(self) -> str:

        discovered_skills = self._discover_skills()
        if not discovered_skills:
            return ""
        lines = [
            "\n## 可用技能列表",
            "可通过调用skills工具来查看和加载技能，以下是当前可用的技能："
        ]
        for name, (_, frontmatter) in sorted(discovered_skills.items()):
            description = frontmatter.get("description", "无描述")
            lines.append(f"- **{name}**: {description}")
        return "\n".join(lines)



    def run(self, parameters: Dict[str, Any], tool_call_id: str) -> Message:
        action = parameters.get("action", "list")
        name = parameters.get("name", "")

        discovered_skills = self._discover_skills()
        if action == "list":
            if not discovered_skills:
                return Message(role="tool", content="当前没有可用的技能", tool_call_id=tool_call_id)
            lines = ["# 当前可用的SKILL列表："]
            
            for skill_name, (_, frontmatter) in sorted(discovered_skills.items()):
                description = frontmatter.get("description", "无描述")
                triggers = frontmatter.get("triggers", "无触发条件")
                lines.append(f"## {skill_name}")
                lines.append(f"description: {description}")
                if triggers:
                    lines.append(f"_Triggers: {triggers}_")
                lines.append("")  # 添加空行分隔技能
            
            return Message(role="tool", content="\n".join(lines), tool_call_id=tool_call_id)
        elif action == "load":
            if not name:
                return Message(role="tool", content="❌ 请提供要加载的技能名称", tool_call_id=tool_call_id)
            if name not in discovered_skills:
                return Message(role="tool", content=f"❌ 未找到技能: {name}", tool_call_id=tool_call_id)
            
            skill_md, frontmatter = discovered_skills[name]
            try:
                content = skill_md.read_text(encoding="utf-8")
                skill_name = frontmatter.get("name", name)
                body = _get_skill_body(content)
                msg = f"# Skill: {skill_name}\n\n{body}"
                return Message(role="tool", content=msg, tool_call_id=tool_call_id)
            except Exception as e:
                return Message(role="tool", content=f"❌ 读取技能时发生错误: {str(e)}", tool_call_id=tool_call_id)

    def get_parameters(self) -> ToolParameters:
        return ToolParameters(
            type="object",
            properties={
                "action": ToolProperty(
                    type="string",
                    description="要执行的操作（list 或 load）"
                ),
                "name": ToolProperty(
                    type="string",
                    description="要加载的技能名称"
                )
            },
            required=["action"]
        )