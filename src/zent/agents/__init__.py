"""Agent implementations for Zent.

This module provides concrete agent implementations:
- ToolCallingAgent: Agent that uses LLM function calling
- CodeAgent: Agent that generates and executes Python code
"""

from zent.agents.code import CodeAgent
from zent.agents.tool_calling import ToolCallingAgent

__all__ = [
    "CodeAgent",
    "ToolCallingAgent",
]
