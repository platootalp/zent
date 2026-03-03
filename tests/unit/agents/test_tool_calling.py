"""Tests for ToolCallingAgent."""

from __future__ import annotations

from typing import Any

import pytest

from zent.agents.tool_calling import ToolCallingAgent
from zent.core.agent import AgentConfig
from zent.core.messages import Message, MessageRole, ModelResponse, ToolCall
from zent.core.model import BaseModel
from zent.core.steps import ActionStep, AgentResult, FinalAnswerStep
from zent.core.tool import BaseTool


class MockTool(BaseTool):
    """Mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }

    async def run(self, query: str) -> str:
        return f"Result for: {query}"


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        super().__init__(model="mock")
        self.responses = responses or []
        self.call_count = 0
        self.last_messages: list[Message] = []
        self.last_tools: list[BaseTool] | None = None

    async def generate(
        self,
        messages: list[Message],
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        self.last_messages = messages
        self.last_tools = tools

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return ModelResponse(content="Default response", model="mock")

    async def stream(self, messages: list[Message], **kwargs: Any) -> Any:
        yield "Mock response"


class TestToolCallingAgent:
    """Test ToolCallingAgent."""

    @pytest.fixture
    def mock_tool(self) -> MockTool:
        return MockTool()

    async def test_agent_with_direct_answer(self, mock_tool: MockTool) -> None:
        """Test agent completes with direct answer (no tool calls)."""
        model = MockModel(
            [
                ModelResponse(content="The answer is 42", model="mock", tool_calls=[]),
            ]
        )

        config = AgentConfig(
            model=model,
            tools=[mock_tool],
            max_iterations=5,
        )

        agent = ToolCallingAgent(config)
        result = await agent.run(task="What is the answer?")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.output == "The answer is 42"
        assert model.call_count == 1

    async def test_agent_with_tool_call(self, mock_tool: MockTool) -> None:
        """Test agent uses tool when needed."""
        model = MockModel(
            [
                ModelResponse(
                    content=None,
                    model="mock",
                    tool_calls=[
                        ToolCall(id="call_1", name="mock_tool", arguments={"query": "test"}),
                    ],
                ),
                ModelResponse(content="Final answer", model="mock", tool_calls=[]),
            ]
        )

        config = AgentConfig(
            model=model,
            tools=[mock_tool],
            max_iterations=5,
        )

        agent = ToolCallingAgent(config)
        result = await agent.run(task="Search for test")

        assert result.success is True
        assert result.output == "Final answer"
        assert model.call_count == 2

    async def test_agent_system_prompt(self, mock_tool: MockTool) -> None:
        """Test agent includes system prompt."""
        model = MockModel(
            [
                ModelResponse(content="Done", model="mock", tool_calls=[]),
            ]
        )

        config = AgentConfig(
            model=model,
            tools=[mock_tool],
            system_prompt="You are a helpful assistant",
            max_iterations=5,
        )

        agent = ToolCallingAgent(config)
        await agent.run(task="Hello")

        # Check system prompt was added
        assert len(model.last_messages) >= 1
        assert model.last_messages[0].role == MessageRole.SYSTEM
        assert model.last_messages[0].content == "You are a helpful assistant"

    async def test_agent_tool_execution(self, mock_tool: MockTool) -> None:
        """Test agent correctly executes tool and passes result back."""
        model = MockModel(
            [
                ModelResponse(
                    content=None,
                    model="mock",
                    tool_calls=[
                        ToolCall(id="call_1", name="mock_tool", arguments={"query": "hello"}),
                    ],
                ),
                ModelResponse(content="Done", model="mock", tool_calls=[]),
            ]
        )

        config = AgentConfig(
            model=model,
            tools=[mock_tool],
            max_iterations=5,
        )

        agent = ToolCallingAgent(config)
        result = await agent.run(task="Test")

        # Check tool result is in messages
        tool_result_message = None
        for msg in model.last_messages:
            if msg.role == MessageRole.TOOL:
                tool_result_message = msg
                break

        assert tool_result_message is not None
        assert "Result for: hello" in tool_result_message.content

    async def test_agent_multiple_tool_calls(self, mock_tool: MockTool) -> None:
        """Test agent handles multiple tool calls in one step."""
        model = MockModel(
            [
                ModelResponse(
                    content=None,
                    model="mock",
                    tool_calls=[
                        ToolCall(id="call_1", name="mock_tool", arguments={"query": "first"}),
                        ToolCall(id="call_2", name="mock_tool", arguments={"query": "second"}),
                    ],
                ),
                ModelResponse(content="Done", model="mock", tool_calls=[]),
            ]
        )

        config = AgentConfig(
            model=model,
            tools=[mock_tool],
            max_iterations=5,
        )

        agent = ToolCallingAgent(config)
        result = await agent.run(task="Test")

        assert result.success is True
        # Check both tool results are present
        tool_results = [msg for msg in model.last_messages if msg.role == MessageRole.TOOL]
        assert len(tool_results) == 2

    async def test_agent_tools_passed_to_model(self, mock_tool: MockTool) -> None:
        """Test that tools are passed to the model."""
        model = MockModel(
            [
                ModelResponse(content="Done", model="mock", tool_calls=[]),
            ]
        )

        config = AgentConfig(
            model=model,
            tools=[mock_tool],
            max_iterations=5,
        )

        agent = ToolCallingAgent(config)
        await agent.run(task="Test")

        assert model.last_tools is not None
        assert len(model.last_tools) == 1
        assert model.last_tools[0].name == "mock_tool"

    async def test_agent_no_tools(self) -> None:
        """Test agent works without tools."""
        model = MockModel(
            [
                ModelResponse(content="Direct answer", model="mock", tool_calls=[]),
            ]
        )

        config = AgentConfig(
            model=model,
            tools=[],
            max_iterations=5,
        )

        agent = ToolCallingAgent(config)
        result = await agent.run(task="Hello")

        assert result.success is True
        assert result.output == "Direct answer"
        assert model.last_tools is None or model.last_tools == []

    async def test_agent_empty_response(self, mock_tool: MockTool) -> None:
        """Test agent handles empty response from model."""
        model = MockModel(
            [
                ModelResponse(content=None, model="mock", tool_calls=[]),
                ModelResponse(content="Recovered", model="mock", tool_calls=[]),
            ]
        )

        config = AgentConfig(
            model=model,
            tools=[mock_tool],
            max_iterations=5,
        )

        agent = ToolCallingAgent(config)

        # Initialize and run one step to get the empty response behavior
        await agent._initialize("Test")
        step = await agent._step()

        # Should return ActionStep with error for empty response
        from zent.core.steps import ActionStep

        assert isinstance(step, ActionStep)
        assert step.error == "Empty response from model"

    async def test_agent_step_returns_action_step(self, mock_tool: MockTool) -> None:
        """Test that _step returns ActionStep for tool calls."""
        model = MockModel(
            [
                ModelResponse(
                    content="I'll search",
                    model="mock",
                    tool_calls=[
                        ToolCall(id="call_1", name="mock_tool", arguments={"query": "test"}),
                    ],
                ),
            ]
        )

        config = AgentConfig(
            model=model,
            tools=[mock_tool],
            max_iterations=5,
        )

        agent = ToolCallingAgent(config)
        # Initialize first
        await agent._initialize("Test task")

        step = await agent._step()

        assert isinstance(step, ActionStep)
        assert len(step.tool_calls) == 1
        assert "Result for: test" in step.observations

    async def test_agent_step_returns_final_answer(self, mock_tool: MockTool) -> None:
        """Test that _step returns FinalAnswerStep for direct answers."""
        model = MockModel(
            [
                ModelResponse(content="Final answer", model="mock", tool_calls=[]),
            ]
        )

        config = AgentConfig(
            model=model,
            tools=[mock_tool],
            max_iterations=5,
        )

        agent = ToolCallingAgent(config)
        # Initialize first
        await agent._initialize("Test task")

        step = await agent._step()

        assert isinstance(step, FinalAnswerStep)
        assert step.answer == "Final answer"


class TestToolCallingAgentMessages:
    """Test message management in ToolCallingAgent."""

    async def test_messages_accumulate(self) -> None:
        """Test that messages accumulate across steps."""

        class MockTool(BaseTool):
            name = "tool"
            description = "Tool"

            @property
            def parameters(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            async def run(self, **kwargs: Any) -> str:
                return "result"

        model = MockModel(
            [
                ModelResponse(
                    content=None,
                    model="mock",
                    tool_calls=[ToolCall(id="1", name="tool", arguments={})],
                ),
                ModelResponse(content="Answer", model="mock", tool_calls=[]),
            ]
        )

        config = AgentConfig(
            model=model,
            tools=[MockTool()],
            max_iterations=5,
        )

        agent = ToolCallingAgent(config)
        await agent.run(task="Test")

        # Should have: system (optional), user, assistant (tool call), tool result, assistant (final)
        assert len(agent._messages) >= 4
