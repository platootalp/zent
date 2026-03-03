"""ToolCallingAgent - Uses LLM native function calling.

Reference:
- smolagents: ToolCallingAgent
- LangChain: create_tool_calling_agent

This agent uses the LLM's native function calling capability
to decide which tools to use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zent.core.agent import ActionStep, Agent, AgentConfig, FinalAnswerStep
from zent.core.messages import Message, MessageRole
from zent.core.steps import PlanningStep

if TYPE_CHECKING:
    from zent.core.steps import ActionStep, FinalAnswerStep


class ToolCallingAgent(Agent):
    """Agent that uses LLM native function calling.

    Similar to smolagents' ToolCallingAgent.

    This agent is suitable for:
    - Models that support function calling (GPT-4, Claude, etc.)
    - Production use cases requiring precise tool control
    - Scenarios where safety and control are important

    Example:
        ```python
        agent = ToolCallingAgent(AgentConfig(
            model=OpenAIModel(model="gpt-4"),
            tools=[search_tool, calculator_tool],
            system_prompt="You are a helpful assistant."
        ))

        result = await agent.run("What is the weather in Paris?")
        ```
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the ToolCallingAgent.

        Args:
            config: The agent configuration.
        """
        super().__init__(config)
        self._messages: list[Message] = []

        # Add system prompt if provided
        if config.system_prompt:
            self._messages.append(Message.system(config.system_prompt))

    async def _initialize(self, task: str) -> None:
        """Initialize with user task."""
        await super()._initialize(task)
        self._messages.append(Message.user(task))

    async def _do_planning(self) -> None:
        """Perform planning step."""
        # Simple planning: ask model to create a plan
        # This can be enhanced with more sophisticated planning
        pass

    async def _step(self) -> "ActionStep | FinalAnswerStep":
        """Execute one step using function calling.

        Returns:
            ActionStep if tools need to be called.
            FinalAnswerStep if task is complete.
        """
        # Get available tools
        available_tools = self.tools.get_tools()

        # Call model
        response = await self.config.model.generate(
            messages=self._messages,
            tools=available_tools if available_tools else None,
        )

        # Handle tool calls
        if response.has_tool_calls:
            # Add assistant message with tool calls
            self._messages.append(
                Message.assistant(
                    content=response.content or "",
                    tool_calls=[call.to_dict() for call in response.tool_calls],
                )
            )

            # Execute tools
            observations = []
            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                observations.append(f"[{tool_call.name}] {result.output}")

                # Add tool result message
                self._messages.append(result.to_message())

            return ActionStep(
                tool_calls=response.tool_calls,
                observations="\n".join(observations),
            )

        # Handle final answer
        elif response.content:
            self._messages.append(Message.assistant(response.content))
            return FinalAnswerStep(answer=response.content)

        # Empty response
        else:
            return ActionStep(
                observations="",
                error="Empty response from model",
            )

    async def _build_messages(self) -> list[Message]:
        """Build message history from memory or use current messages.

        Returns:
            List of messages for the LLM.
        """
        if self.config.memory:
            return await self.config.memory.get_messages(limit=20)
        return self._messages
