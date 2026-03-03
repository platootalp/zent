"""Core data types for Zent.

This module contains all data models used throughout the framework.
It has zero internal dependencies to avoid circular imports.

Reference:
- LangChain: BaseMessage, HumanMessage, AIMessage, ToolMessage
- smolagents: Message, ChatMessage, MemoryStep variants
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Message role enumeration."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """A single message in a conversation.

    Similar to LangChain's BaseMessage but simplified.
    Used for LLM communication context.

    Attributes:
        role: The role of the message sender.
        content: The text content of the message.
        metadata: Additional metadata (tool_calls, etc.).
    """

    role: MessageRole
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for API calls."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        result.update(self.metadata)
        return result

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str, **metadata: Any) -> Message:
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, metadata=metadata)

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> Message:
        """Create a tool result message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            metadata={"tool_call_id": tool_call_id},
        )


@dataclass
class ToolCall:
    """A request to call a tool.

    Similar to LangChain's ToolCall and smolagents' ToolCall.

    Attributes:
        id: Unique identifier for this tool call.
        name: The name of the tool to call.
        arguments: The arguments to pass to the tool (as dict).
    """

    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to OpenAI-style tool call format."""
        import json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


@dataclass
class ToolResult:
    """The result of a tool execution.

    Attributes:
        call_id: The ID of the corresponding ToolCall.
        output: The output from the tool.
        is_error: Whether the tool execution resulted in an error.
    """

    call_id: str
    output: str
    is_error: bool = False

    def to_message(self) -> Message:
        """Convert to a tool message."""
        return Message.tool(self.output, self.call_id)


@dataclass
class ModelResponse:
    """Response from a language model.

    Similar to LangChain's AIMessage with tool_calls.

    Attributes:
        content: The text content of the response.
        tool_calls: List of tool calls requested by the model.
        model: The name of the model that generated the response.
        finish_reason: Why the model stopped generating.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    finish_reason: str | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0


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
