"""Tests for message data models."""

import pytest

from zent.core.messages import (
    Message,
    MessageRole,
    ModelResponse,
    ToolCall,
    ToolResult,
)


class TestMessage:
    """Test Message class."""

    def test_create_user_message(self) -> None:
        """Test creating a user message."""
        msg = Message.user("Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_create_assistant_message(self) -> None:
        """Test creating an assistant message."""
        msg = Message.assistant("Hi there", tool_calls=[{"id": "1"}])
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there"
        assert msg.metadata["tool_calls"] == [{"id": "1"}]

    def test_create_system_message(self) -> None:
        """Test creating a system message."""
        msg = Message.system("You are helpful")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are helpful"

    def test_create_tool_message(self) -> None:
        """Test creating a tool message."""
        msg = Message.tool("Result", "call_123")
        assert msg.role == MessageRole.TOOL
        assert msg.content == "Result"
        assert msg.metadata["tool_call_id"] == "call_123"

    def test_message_to_dict(self) -> None:
        """Test converting message to dictionary."""
        msg = Message.user("Test")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Test"


class TestToolCall:
    """Test ToolCall class."""

    def test_create_tool_call(self) -> None:
        """Test creating a tool call."""
        call = ToolCall(
            id="call_1",
            name="search",
            arguments={"query": "python"},
        )
        assert call.id == "call_1"
        assert call.name == "search"
        assert call.arguments == {"query": "python"}

    def test_tool_call_to_dict(self) -> None:
        """Test converting tool call to OpenAI format."""
        call = ToolCall(
            id="call_1",
            name="search",
            arguments={"query": "python"},
        )
        d = call.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "search"


class TestToolResult:
    """Test ToolResult class."""

    def test_create_tool_result(self) -> None:
        """Test creating a tool result."""
        result = ToolResult(call_id="call_1", output="Found it")
        assert result.call_id == "call_1"
        assert result.output == "Found it"
        assert not result.is_error

    def test_error_result(self) -> None:
        """Test creating an error result."""
        result = ToolResult(call_id="call_1", output="Error", is_error=True)
        assert result.is_error

    def test_to_message(self) -> None:
        """Test converting result to message."""
        result = ToolResult(call_id="call_1", output="Found it")
        msg = result.to_message()
        assert msg.role == MessageRole.TOOL
        assert msg.content == "Found it"


class TestModelResponse:
    """Test ModelResponse class."""

    def test_response_with_content(self) -> None:
        """Test response with text content."""
        response = ModelResponse(
            content="Hello",
            model="gpt-4",
            tool_calls=[],
        )
        assert response.content == "Hello"
        assert not response.has_tool_calls

    def test_response_with_tool_calls(self) -> None:
        """Test response with tool calls."""
        response = ModelResponse(
            content=None,
            model="gpt-4",
            tool_calls=[ToolCall(id="1", name="search", arguments={})],
        )
        assert response.has_tool_calls

    def test_response_with_both(self) -> None:
        """Test response with both content and tool calls."""
        response = ModelResponse(
            content="I'll search for that",
            model="gpt-4",
            tool_calls=[ToolCall(id="1", name="search", arguments={})],
        )
        assert response.content is not None
        assert response.has_tool_calls
