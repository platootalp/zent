"""Agent base class using Template Method pattern.

Reference:
- smolagents: MultiStepAgent with _run() template and _step() abstract method
- LangChain: Agent concepts but simplified

This implements the Template Method design pattern where:
- Base class defines the ReAct loop structure (_run)
- Subclasses implement specific step logic (_step)
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from zent.core.types import (
    ActionStep,
    AgentResult,
    FinalAnswerStep,
    MemoryStep,
    TaskStep,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    from zent.core.memory import Memory
    from zent.core.model import BaseModel
    from zent.core.tool import BaseTool, ToolRegistry


@dataclass
class AgentConfig:
    """Configuration for an Agent.

    Attributes:
        model: The language model to use.
        tools: List of available tools.
        memory: Optional memory implementation.
        max_iterations: Maximum number of ReAct iterations.
        system_prompt: System prompt defining agent behavior.
        planning_interval: Re-plan every N steps (optional).
        on_step: Callback for each step (for observability).
        on_error: Callback for errors.
    """

    model: "BaseModel"
    tools: list["BaseTool"] = field(default_factory=list)
    memory: "Memory | None" = None
    max_iterations: int = 10
    system_prompt: str | None = None
    planning_interval: int | None = None
    on_step: Callable[[MemoryStep], None] | None = None
    on_error: Callable[[Exception], None] | None = None


class Agent(ABC):
    """Abstract base class for agents using Template Method pattern.

    Similar to smolagents' MultiStepAgent.

    Subclasses must implement:
        - _step(): Execute a single step

    The base class handles:
        - ReAct loop management
        - Memory storage
        - Callback triggering
        - Error handling
        - Result assembly

    Example:
        ```python
        class MyAgent(Agent):
            async def _step(self):
                # Implement single step logic
                return ActionStep(...) or FinalAnswerStep(...)
        ```
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the agent.

        Args:
            config: The agent configuration.
        """
        self.config = config
        self.tools = ToolRegistry(config.tools)
        self.step_number = 0
        self.completed = False
        self._task_id: str | None = None

    async def run(self, task: str, **context: Any) -> AgentResult:
        """Execute a task using the ReAct framework.

        This is the Template Method that defines the execution flow.
        Subclasses should NOT override this method.

        Args:
            task: The task description from the user.
            **context: Additional context for the task.

        Returns:
            The result of the task execution.
        """
        self._task_id = str(uuid.uuid4())[:8]
        self.step_number = 0
        self.completed = False

        try:
            # 1. Initialize
            await self._initialize(task)

            # 2. ReAct Loop
            while not self.completed and self.step_number < self.config.max_iterations:
                self.step_number += 1

                # 2.1 Planning (optional)
                if self._should_plan():
                    await self._do_planning()

                # 2.2 Execute step (subclass implements)
                step_result = await self._step()

                # 2.3 Store in memory
                if self.config.memory:
                    await self.config.memory.add(step_result)

                # 2.4 Trigger callback
                if self.config.on_step:
                    self.config.on_step(step_result)

                # 2.5 Check completion
                if isinstance(step_result, FinalAnswerStep):
                    self.completed = True
                    return AgentResult(
                        output=step_result.answer,
                        steps=await self._get_steps(),
                        success=True,
                        task_id=self._task_id,
                    )

            # 3. Max iterations reached
            return AgentResult(
                output="Maximum iterations reached without completion",
                steps=await self._get_steps(),
                success=False,
                task_id=self._task_id,
            )

        except Exception as e:
            if self.config.on_error:
                self.config.on_error(e)
            return AgentResult(
                output=str(e),
                steps=await self._get_steps(),
                success=False,
                error=e,
                task_id=self._task_id or "",
            )

    async def _initialize(self, task: str) -> None:
        """Initialize the task.

        Can be overridden by subclasses for custom initialization.

        Args:
            task: The task description.
        """
        task_step = TaskStep(task=task)
        if self.config.memory:
            await self.config.memory.add(task_step)

    def _should_plan(self) -> bool:
        """Check if planning should occur this step.

        Returns:
            True if planning interval is set and reached.
        """
        return (
            self.config.planning_interval is not None
            and self.step_number % self.config.planning_interval == 0
            and self.step_number > 0
        )

    async def _do_planning(self) -> None:
        """Perform planning step.

        Can be overridden by subclasses. Default does nothing.
        """
        pass

    @abstractmethod
    async def _step(self) -> ActionStep | FinalAnswerStep:
        """Execute a single step.

        This is the abstract method subclasses must implement.

        Returns:
            ActionStep: To continue the loop.
            FinalAnswerStep: To complete the task.
        """
        pass

    async def _get_steps(self) -> list[MemoryStep]:
        """Get all steps from memory.

        Returns:
            List of memory steps.
        """
        if self.config.memory:
            return await self.config.memory.get_steps(limit=1000)
        return []

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call.

        Helper method for subclasses.

        Args:
            call: The tool call to execute.

        Returns:
            The result of the tool execution.
        """
        tool = self.tools.get(call.name)

        if not tool:
            return ToolResult(
                call_id=call.id,
                output=f"Tool '{call.name}' not found",
                is_error=True,
            )

        try:
            output = await tool.run(**call.arguments)
            return ToolResult(call_id=call.id, output=output)
        except Exception as e:
            return ToolResult(
                call_id=call.id,
                output=f"Error: {e}",
                is_error=True,
            )


# Avoid circular import
from zent.core.tool import ToolRegistry
