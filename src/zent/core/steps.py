"""Strongly typed memory steps for Zent.

Reference:
- smolagents: MemoryStep, TaskStep, ActionStep, PlanningStep, FinalAnswerStep
- LangChain: Similar concepts but less formalized

These steps provide observability and debugging capabilities by tracking
each action the agent takes with type safety.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from zent.core.messages import ToolCall


class StepType(str, Enum):
    """Type of memory step."""

    SYSTEM = "system"
    TASK = "task"
    PLANNING = "planning"
    ACTION = "action"
    OBSERVATION = "observation"
    FINAL_ANSWER = "final_answer"


@dataclass
class MemoryStep:
    """Base class for all memory steps.

    Similar to smolagents' MemoryStep base class.

    Attributes:
        step_type: The type of step (for serialization/deserialization).
        timestamp: When the step was created.
        metadata: Additional metadata for observability.
    """

    step_type: StepType = field(init=False)
    timestamp: datetime = field(default_factory=datetime.now, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class SystemPromptStep(MemoryStep):
    """Initial system prompt step.

    Stores the system prompt that defines agent behavior.
    """

    system_prompt: str = ""

    def __post_init__(self) -> None:
        self.step_type = StepType.SYSTEM


@dataclass
class TaskStep(MemoryStep):
    """User task input step.

    Records what task the user asked the agent to perform.

    Attributes:
        task: The user's task description.
    """

    task: str = ""

    def __post_init__(self) -> None:
        self.step_type = StepType.TASK


@dataclass
class PlanningStep(MemoryStep):
    """Planning step where agent thinks about approach.

    Similar to smolagents' PlanningStep.

    Attributes:
        plan: The agent's plan for solving the task.
        facts: Relevant facts discovered so far.
    """

    plan: str = ""
    facts: str = ""

    def __post_init__(self) -> None:
        self.step_type = StepType.PLANNING


@dataclass
class ActionStep(MemoryStep):
    """Action step where agent calls tools or executes code.

    Similar to smolagents' ActionStep.

    Attributes:
        tool_calls: List of tool calls made in this step.
        observations: The results/observations from tool execution.
        error: Any error that occurred.
        duration: Time taken for execution in seconds.
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    observations: str = ""
    error: str | None = None
    duration: float = 0.0

    def __post_init__(self) -> None:
        self.step_type = StepType.ACTION


@dataclass
class ObservationStep(MemoryStep):
    """Separate observation step for explicit tracking.

    Alternative to including observations in ActionStep.
    """

    content: str = ""
    tool_call_id: str | None = None
    is_error: bool = False

    def __post_init__(self) -> None:
        self.step_type = StepType.OBSERVATION


@dataclass
class FinalAnswerStep(MemoryStep):
    """Final answer step when task is complete.

    Similar to smolagents' FinalAnswerStep.

    Attributes:
        answer: The final answer provided to the user.
    """

    answer: str = ""

    def __post_init__(self) -> None:
        self.step_type = StepType.FINAL_ANSWER


@dataclass
class AgentResult:
    """Result of an agent run.

    Contains the complete execution history and final output.

    Attributes:
        output: The final text output.
        steps: Complete list of memory steps.
        success: Whether the run succeeded.
        error: Any error that occurred.
        task_id: Unique identifier for this task run.
    """

    output: str = ""
    steps: list[MemoryStep] = field(default_factory=list)
    success: bool = True
    error: Exception | None = None
    task_id: str = ""

    @property
    def final_answer(self) -> str:
        """Alias for output property (consistent naming)."""
        return self.output

    @property
    def step_count(self) -> int:
        """Count of non-system/task steps (actual work steps)."""
        return len(
            [s for s in self.steps if isinstance(s, (ActionStep, PlanningStep, FinalAnswerStep))]
        )
