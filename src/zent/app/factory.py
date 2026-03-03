"""Factory functions for creating agents.

Provides convenient factory functions for common agent configurations.
Similar to LangChain's create_* functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zent.agents.code import CodeAgent
from zent.agents.tool_calling import ToolCallingAgent
from zent.core.agent import AgentConfig

if TYPE_CHECKING:
    from zent.core.agent import Agent
    from zent.core.model import BaseModel
    from zent.core.tool import BaseTool
    from zent.integrations.memory.in_memory import InMemoryMemory


def create_agent(
    model: BaseModel | str,
    tools: list[BaseTool] | None = None,
    memory: InMemoryMemory | None = None,
    system_prompt: str | None = None,
    agent_type: str = "tool_calling",
    **kwargs,
) -> Agent:
    """Create an agent with the specified configuration.

    This is the main entry point for creating agents.
    Supports both model instances and string shortcuts.

    Args:
        model: Model instance or string (e.g., "openai:gpt-4").
        tools: List of available tools.
        memory: Memory implementation (creates InMemoryMemory if None).
        system_prompt: System prompt for the agent.
        agent_type: Type of agent ("tool_calling" or "code").
        **kwargs: Additional configuration options.

    Returns:
        Configured agent instance.

    Raises:
        ValueError: If agent_type is unknown or model string is invalid.

    Example:
        ```python
        # Using string shortcut
        agent = create_agent("openai:gpt-4", tools=[search, calculator])

        # Using model instance
        from zent.integrations.models.openai import OpenAIModel
        model = OpenAIModel(model="gpt-4")
        agent = create_agent(model, tools=[search])

        # Full configuration
        agent = create_agent(
            "openai:gpt-4",
            tools=[search, calculator],
            memory=InMemoryMemory(),
            system_prompt="You are a helpful assistant.",
            max_iterations=15,
        )
        ```
    """
    # Parse model string if needed
    if isinstance(model, str):
        model = _resolve_model(model)

    # Create default memory if not provided
    if memory is None:
        from zent.integrations.memory.in_memory import InMemoryMemory

        memory = InMemoryMemory()

    # Create configuration
    config = AgentConfig(
        model=model,
        tools=tools or [],
        memory=memory,
        system_prompt=system_prompt,
        **kwargs,
    )

    # Create agent based on type
    if agent_type == "tool_calling":
        return ToolCallingAgent(config)
    elif agent_type == "code":
        return CodeAgent(config)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")


def _resolve_model(model_str: str) -> BaseModel:
    """Resolve a model string to a model instance.

    Supports formats:
    - "openai:gpt-4" -> OpenAIModel
    - "openai:gpt-3.5-turbo" -> OpenAIModel
    - "azure:deployment-name" -> AzureOpenAIModel (requires endpoint)

    Args:
        model_str: Model string in format "provider:model".

    Returns:
        Model instance.

    Raises:
        ValueError: If format is invalid or provider is unknown.
    """
    if ":" not in model_str:
        raise ValueError(f"Model string must be in format 'provider:model', got: {model_str}")

    provider, model_name = model_str.split(":", 1)

    if provider == "openai":
        from zent.integrations.models.openai import OpenAIModel

        return OpenAIModel(model=model_name)
    elif provider == "anthropic":
        from zent.integrations.models.anthropic import AnthropicModel

        return AnthropicModel(model=model_name)
    elif provider == "azure":
        raise ValueError(
            "Azure OpenAI requires endpoint configuration. "
            "Use AzureOpenAIModel directly or provide azure_endpoint in kwargs."
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic, azure")
