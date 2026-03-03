"""Zent - A lightweight Python Agent framework.

Zent provides a simple, extensible framework for building LLM-powered agents.

Basic usage:
    ```python
    from zent import create_agent, tool

    @tool
    def search(query: str) -> str:
        \"\"\"Search for information\"\"\"
        return f"Results for: {query}"

    agent = create_agent("openai:gpt-4", tools=[search])
    result = await agent.run("Search for Python tutorials")
    print(result.output)
    ```

Advanced usage:
    ```python
    from zent import AgentConfig
    from zent.agents.tool_calling import ToolCallingAgent
    from zent.integrations.models.openai import OpenAIModel

    config = AgentConfig(
        model=OpenAIModel(model="gpt-4"),
        tools=[search, calculator],
        system_prompt="You are a helpful assistant.",
    )
    agent = ToolCallingAgent(config)
    ```
"""

__version__ = "0.4.0"

# Core
from zent.core.agent import Agent, AgentConfig
from zent.core.messages import Message, ToolCall, ToolResult
from zent.core.model import BaseModel
from zent.core.steps import (
    ActionStep,
    AgentResult,
    FinalAnswerStep,
    MemoryStep,
    TaskStep,
)
from zent.core.tool import BaseTool, ToolRegistry, tool

# Agents
from zent.agents.code import CodeAgent
from zent.agents.tool_calling import ToolCallingAgent

# Factory
from zent.app.factory import create_agent

__all__ = [
    # Version
    "__version__",
    # Core
    "Agent",
    "AgentConfig",
    "BaseModel",
    "BaseTool",
    "ToolRegistry",
    "tool",
    # Messages
    "Message",
    "ToolCall",
    "ToolResult",
    # Steps
    "MemoryStep",
    "TaskStep",
    "ActionStep",
    "FinalAnswerStep",
    "AgentResult",
    # Agents
    "ToolCallingAgent",
    "CodeAgent",
    # Factory
    "create_agent",
]
