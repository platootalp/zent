"""Message data models for Zent.

Reference:
- LangChain: BaseMessage, HumanMessage, AIMessage, ToolMessage
- smolagents: Message, ChatMessage
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
