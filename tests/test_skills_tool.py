import pytest
from violet_agents.tools.builtin.skills_tool import SkillsTool

def test_skill_discovery():
    """测试技能发现功能"""
    tool = SkillsTool()
    
    # 测试技能列表
    result = tool.run(action="list")
    assert "available skills" in result.lower()
    
    # 测试加载技能
    load_result = tool.run(action="load", name="calculator")
    assert "loaded" in load_result.lower()