"""Tests for agent base class."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from zent.core.agent import Agent, AgentConfig
from zent.core.memory import Memory
from zent.core.messages import Message, ModelResponse, ToolCall, ToolResult
from zent.core.model import BaseModel
from zent.core.steps import (
    ActionStep,
    AgentResult,
    FinalAnswerStep,
    MemoryStep,
    TaskStep,
)
from zent.core.tool import BaseTool, ToolRegistry


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self) -> None:
        super().__init__(model="mock")

    async def generate(
        self,
        messages: list[Message],
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        return ModelResponse(content="mock", model="mock")

    async def stream(self, messages: list[Message], **kwargs: Any) -> AsyncIterator[str]:
        yield "mock"


class MockMemory(Memory):
    """Mock memory implementation for testing."""

    def __init__(self) -> None:
        self._steps: list[MemoryStep] = []

    async def add(self, step: MemoryStep) -> None:
        self._steps.append(step)

    async def get_messages(self, limit: int = 10) -> list[Message]:
        return []

    async def get_steps(self, limit: int = 10) -> list[MemoryStep]:
        return self._steps[-limit:]

    async def clear(self) -> None:
        self._steps = []


class SimpleTestAgent(Agent):
    """Simple test agent for testing base class."""

    def __init__(
        self, config: AgentConfig, steps_to_return: list[ActionStep | FinalAnswerStep]
    ) -> None:
        super().__init__(config)
        self._steps_to_return = iter(steps_to_return)
        self.step_calls = 0

    async def _step(self) -> ActionStep | FinalAnswerStep:
        self.step_calls += 1
        return next(self._steps_to_return)


class TestAgentBase:
    """Test Agent base class."""

    @pytest.fixture
    def mock_config(self) -> AgentConfig:
        """Create a mock config."""
        return AgentConfig(
            model=MockModel(),
            tools=[],
            memory=None,
            max_iterations=5,
        )

    async def test_agent_run_completes_on_final_answer(self, mock_config: AgentConfig) -> None:
        """Test agent completes when FinalAnswerStep is returned."""
        agent = SimpleTestAgent(
            mock_config,
            steps_to_return=[FinalAnswerStep(answer="Done")],
        )

        result = await agent.run(task="Test task")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.output == "Done"
        assert agent.step_calls == 1

    async def test_agent_run_multiple_steps(self, mock_config: AgentConfig) -> None:
        """Test agent runs multiple steps before completion."""
        agent = SimpleTestAgent(
            mock_config,
            steps_to_return=[
                ActionStep(tool_calls=[], observations="Step 1"),
                ActionStep(tool_calls=[], observations="Step 2"),
                FinalAnswerStep(answer="Final"),
            ],
        )

        result = await agent.run(task="Test task")

        assert result.success is True
        assert result.output == "Final"
        assert agent.step_calls == 3

    async def test_agent_run_max_iterations(self, mock_config: AgentConfig) -> None:
        """Test agent stops at max iterations."""
        mock_config.max_iterations = 3
        agent = SimpleTestAgent(
            mock_config,
            steps_to_return=[
                ActionStep(tool_calls=[], observations="Step 1"),
                ActionStep(tool_calls=[], observations="Step 2"),
                ActionStep(tool_calls=[], observations="Step 3"),
                ActionStep(tool_calls=[], observations="Step 4"),
            ],
        )

        result = await agent.run(task="Test task")

        assert result.success is False
        assert "Maximum iterations" in result.output
        assert agent.step_calls == 3  # Stops at max_iterations

    async def test_agent_run_with_memory(self, mock_config: AgentConfig) -> None:
        """Test agent stores steps in memory."""
        memory = MockMemory()
        mock_config.memory = memory

        agent = SimpleTestAgent(
            mock_config,
            steps_to_return=[FinalAnswerStep(answer="Done")],
        )

        await agent.run(task="Test task")

        steps = await memory.get_steps()
        assert len(steps) == 2  # TaskStep + FinalAnswerStep
        assert isinstance(steps[0], TaskStep)
        assert isinstance(steps[1], FinalAnswerStep)

    async def test_agent_run_with_callback(self, mock_config: AgentConfig) -> None:
        """Test agent triggers step callback."""
        callback_steps: list[MemoryStep] = []

        def on_step(step: MemoryStep) -> None:
            callback_steps.append(step)

        mock_config.on_step = on_step

        agent = SimpleTestAgent(
            mock_config,
            steps_to_return=[FinalAnswerStep(answer="Done")],
        )

        await agent.run(task="Test task")

        assert len(callback_steps) == 1
        assert isinstance(callback_steps[0], FinalAnswerStep)

    async def test_agent_run_with_error(self, mock_config: AgentConfig) -> None:
        """Test agent handles errors."""

        class ErrorAgent(Agent):
            async def _step(self) -> ActionStep | FinalAnswerStep:
                raise ValueError("Test error")

        agent = ErrorAgent(mock_config)
        result = await agent.run(task="Test task")

        assert result.success is False
        assert "Test error" in result.output

    async def test_agent_run_with_error_callback(self, mock_config: AgentConfig) -> None:
        """Test error callback is triggered."""
        error_caught: Exception | None = None

        def on_error(e: Exception) -> None:
            nonlocal error_caught
            error_caught = e

        mock_config.on_error = on_error

        class ErrorAgent(Agent):
            async def _step(self) -> ActionStep | FinalAnswerStep:
                raise ValueError("Test error")

        agent = ErrorAgent(mock_config)
        await agent.run(task="Test task")

        assert isinstance(error_caught, ValueError)
        assert str(error_caught) == "Test error"

    async def test_agent_task_id_generation(self, mock_config: AgentConfig) -> None:
        """Test task ID is generated."""
        agent = SimpleTestAgent(
            mock_config,
            steps_to_return=[FinalAnswerStep(answer="Done")],
        )

        result = await agent.run(task="Test task")

        assert result.task_id
        assert len(result.task_id) == 8  # First 8 chars of UUID

    async def test_agent_tool_registry_initialized(self, mock_config: AgentConfig) -> None:
        """Test tool registry is initialized with tools."""

        class MockTool(BaseTool):
            name = "mock"
            description = "Mock tool"

            @property
            def parameters(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            async def run(self, **kwargs: Any) -> str:
                return "mock result"

        mock_config.tools = [MockTool()]
        agent = SimpleTestAgent(mock_config, steps_to_return=[FinalAnswerStep(answer="Done")])

        assert isinstance(agent.tools, ToolRegistry)
        assert "mock" in agent.tools


class TestAgentToolExecution:
    """Test agent tool execution helper."""

    @pytest.fixture
    def mock_config_with_tool(self) -> AgentConfig:
        """Create config with a mock tool."""

        class MockTool(BaseTool):
            name = "test_tool"
            description = "Test tool"

            @property
            def parameters(self) -> dict[str, Any]:
                return {"type": "object", "properties": {"query": {"type": "string"}}}

            async def run(self, query: str) -> str:
                return f"Result for {query}"

        return AgentConfig(
            model=MockModel(),
            tools=[MockTool()],
            max_iterations=5,
        )

    async def test_execute_existing_tool(self, mock_config_with_tool: AgentConfig) -> None:
        """Test executing an existing tool."""
        agent = SimpleTestAgent(
            mock_config_with_tool,
            steps_to_return=[FinalAnswerStep(answer="Done")],
        )

        call = ToolCall(id="call_1", name="test_tool", arguments={"query": "test"})
        result = await agent._execute_tool(call)

        assert isinstance(result, ToolResult)
        assert result.call_id == "call_1"
        assert result.output == "Result for test"
        assert not result.is_error

    async def test_execute_nonexistent_tool(self, mock_config_with_tool: AgentConfig) -> None:
        """Test executing a non-existent tool."""
        agent = SimpleTestAgent(
            mock_config_with_tool,
            steps_to_return=[FinalAnswerStep(answer="Done")],
        )

        call = ToolCall(id="call_1", name="nonexistent", arguments={})
        result = await agent._execute_tool(call)

        assert result.is_error
        assert "not found" in result.output

    async def test_execute_tool_with_error(self, mock_config_with_tool: AgentConfig) -> None:
        """Test tool execution that raises an error."""

        class ErrorTool(BaseTool):
            name = "error_tool"
            description = "Tool that errors"

            @property
            def parameters(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            async def run(self, **kwargs: Any) -> str:
                raise RuntimeError("Tool error")

        mock_config_with_tool.tools = [ErrorTool()]
        agent = SimpleTestAgent(
            mock_config_with_tool,
            steps_to_return=[FinalAnswerStep(answer="Done")],
        )

        call = ToolCall(id="call_1", name="error_tool", arguments={})
        result = await agent._execute_tool(call)

        assert result.is_error
        assert "Tool error" in result.output


class TestAgentConfig:
    """Test AgentConfig."""

    def test_config_defaults(self) -> None:
        """Test config has sensible defaults."""
        config = AgentConfig(model=MockModel())

        assert config.tools == []
        assert config.memory is None
        assert config.max_iterations == 10
        assert config.system_prompt is None
        assert config.planning_interval is None
        assert config.on_step is None
        assert config.on_error is None

    def test_config_custom_values(self) -> None:
        """Test config with custom values."""
        config = AgentConfig(
            model=MockModel(),
            tools=[],
            max_iterations=20,
            system_prompt="You are helpful",
            planning_interval=3,
        )

        assert config.max_iterations == 20
        assert config.system_prompt == "You are helpful"
        assert config.planning_interval == 3
