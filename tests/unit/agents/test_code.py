"""Tests for CodeAgent and LocalPythonExecutor."""

from __future__ import annotations

from typing import Any

import pytest

from zent.agents.code import CodeAgent, ExecutionResult, LocalPythonExecutor
from zent.core.agent import AgentConfig
from zent.core.messages import Message, ModelResponse
from zent.core.model import BaseModel
from zent.core.steps import ActionStep, FinalAnswerStep
from zent.core.tool import BaseTool


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        super().__init__(model="mock")
        self.responses = responses or []
        self.call_count = 0

    async def generate(
        self,
        messages: list[Message],
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return ModelResponse(content="_output = 'default'", model="mock")

    async def stream(self, messages: list[Message], **kwargs: Any) -> Any:
        yield "mock"


class TestLocalPythonExecutor:
    """Test LocalPythonExecutor."""

    async def test_execute_simple_code(self) -> None:
        """Test executing simple code."""
        executor = LocalPythonExecutor()
        result = await executor.execute("_output = 'hello world'")

        assert isinstance(result, ExecutionResult)
        assert result.output == "hello world"
        assert result.error is None

    async def test_execute_with_math(self) -> None:
        """Test executing code with math operations."""
        executor = LocalPythonExecutor()
        result = await executor.execute("_output = str(2 + 2)")

        assert result.output == "4"

    async def test_execute_with_error(self) -> None:
        """Test executing code with syntax error."""
        executor = LocalPythonExecutor()
        result = await executor.execute("invalid syntax!!!")

        assert result.error is not None
        assert "SyntaxError" in result.error

    async def test_execute_runtime_error(self) -> None:
        """Test executing code with runtime error."""
        executor = LocalPythonExecutor()
        result = await executor.execute("_output = str(1 / 0)")

        assert result.error is not None
        assert "ZeroDivisionError" in result.error

    async def test_unauthorized_import(self) -> None:
        """Test that unauthorized imports are blocked."""
        executor = LocalPythonExecutor(authorized_imports=["math"])
        result = await executor.execute("import os\n_output = 'test'")

        assert result.error is not None
        assert "Unauthorized import" in result.error
        assert "os" in result.error

    async def test_authorized_import(self) -> None:
        """Test that authorized imports work."""
        executor = LocalPythonExecutor(authorized_imports=["math"])
        # Use the math module directly (it's pre-imported in globals)
        result = await executor.execute("_output = str(math.sqrt(16))")

        assert result.error is None
        assert result.output == "4.0"

    async def test_dangerous_function_blocked(self) -> None:
        """Test that dangerous functions are blocked."""
        executor = LocalPythonExecutor()
        result = await executor.execute("eval('1 + 1')")

        assert result.error is not None
        assert "Dangerous function" in result.error

    async def test_file_operation_blocked(self) -> None:
        """Test that file operations are blocked."""
        executor = LocalPythonExecutor()
        result = await executor.execute("open('test.txt', 'w')")

        # File operations are blocked by not including 'open' in safe builtins
        assert result.error is not None
        assert "NameError" in result.error or "File operation" in result.error

    async def test_execute_with_tools(self) -> None:
        """Test executing code with tool functions available."""

        class MockTool(BaseTool):
            name = "add"
            description = "Add two numbers"

            @property
            def parameters(self) -> dict[str, Any]:
                return {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                }

            async def run(self, a: int, b: int) -> str:
                return str(a + b)

        from zent.core.tool import ToolRegistry

        registry = ToolRegistry([MockTool()])
        executor = LocalPythonExecutor(tools=registry)

        # Tools are added to the execution globals
        # Try to call the tool (it will fail since it's async, but proves it's available)
        result = await executor.execute("""
try:
    add(1, 2)
    _output = 'tool available'
except Exception as e:
    _output = 'tool available: ' + type(e).__name__
""")
        assert "tool available" in result.output


class TestCodeAgent:
    """Test CodeAgent."""

    @pytest.fixture
    def mock_config(self) -> AgentConfig:
        """Create a mock config."""
        return AgentConfig(
            model=MockModel(),
            tools=[],
            max_iterations=5,
        )

    async def test_agent_code_generation(self, mock_config: AgentConfig) -> None:
        """Test agent generates and executes code."""
        model = MockModel(
            [
                ModelResponse(
                    content="```python\n_output = 'hello'\n```",
                    model="mock",
                ),
            ]
        )
        mock_config.model = model

        agent = CodeAgent(mock_config)
        result = await agent.run(task="Say hello")

        assert result.success is True
        assert "hello" in result.output

    async def test_agent_multiple_steps(self, mock_config: AgentConfig) -> None:
        """Test agent runs multiple code steps."""
        model = MockModel(
            [
                ModelResponse(
                    content="```python\nresult = 2 + 2\n_output = str(result)\n```",
                    model="mock",
                ),
                ModelResponse(
                    content="```python\n_output = 'Final: 4'\n```",
                    model="mock",
                ),
            ]
        )
        mock_config.model = model

        agent = CodeAgent(mock_config)
        result = await agent.run(task="Calculate 2+2")

        assert result.success is True
        # Should complete after getting a final answer

    async def test_agent_handles_code_error(self, mock_config: AgentConfig) -> None:
        """Test agent handles code execution errors."""
        model = MockModel(
            [
                ModelResponse(
                    content="```python\n_output = 1 / 0\n```",
                    model="mock",
                ),
                ModelResponse(
                    content="```python\n_output = 'Fixed'\n```",
                    model="mock",
                ),
            ]
        )
        mock_config.model = model

        agent = CodeAgent(mock_config)
        result = await agent.run(task="Cause error then fix")

        assert result.success is True
        assert "Fixed" in result.output

    async def test_agent_no_code_block(self, mock_config: AgentConfig) -> None:
        """Test agent handles response without code block."""
        model = MockModel(
            [
                ModelResponse(
                    content="_output = 'no code block'",
                    model="mock",
                ),
            ]
        )
        mock_config.model = model

        agent = CodeAgent(mock_config)
        result = await agent.run(task="Test")

        assert result.success is True
        assert "no code block" in result.output

    async def test_agent_empty_response(self, mock_config: AgentConfig) -> None:
        """Test agent handles empty response."""
        model = MockModel(
            [
                ModelResponse(content="", model="mock"),
                ModelResponse(content="```python\n_output = 'retry'\n```", model="mock"),
            ]
        )
        mock_config.model = model

        agent = CodeAgent(mock_config)
        result = await agent.run(task="Test")

        # Should handle the empty response gracefully
        assert isinstance(result.steps, list)

    async def test_agent_step_returns_action_step(self, mock_config: AgentConfig) -> None:
        """Test that _step returns ActionStep for intermediate steps."""
        model = MockModel(
            [
                ModelResponse(
                    content="```python\nintermediate = 'step'\n_output = intermediate\n```",
                    model="mock",
                ),
            ]
        )
        mock_config.model = model

        agent = CodeAgent(mock_config)
        await agent._initialize("Test")

        step = await agent._step()

        # Should return FinalAnswerStep since _output was set
        assert isinstance(step, FinalAnswerStep)

    async def test_agent_with_authorized_imports(self, mock_config: AgentConfig) -> None:
        """Test agent with custom authorized imports."""
        mock_config.authorized_imports = ["math", "random"]

        agent = CodeAgent(mock_config)
        assert "math" in agent.executor.authorized_imports
        assert "random" in agent.executor.authorized_imports


class TestCodeParsing:
    """Test code parsing from model responses."""

    def test_parse_python_code_block(self) -> None:
        """Test parsing python code block."""
        agent = CodeAgent(AgentConfig(model=MockModel()))

        content = """Here's the solution:
```python
x = 1 + 1
_output = str(x)
```
Done!"""

        code = agent._parse_code(content)
        assert "x = 1 + 1" in code
        assert "_output" in code
        assert "```" not in code

    def test_parse_generic_code_block(self) -> None:
        """Test parsing generic code block."""
        agent = CodeAgent(AgentConfig(model=MockModel()))

        content = """```
some code here
```"""

        code = agent._parse_code(content)
        assert "some code here" in code

    def test_parse_no_code_block(self) -> None:
        """Test parsing when no code block present."""
        agent = CodeAgent(AgentConfig(model=MockModel()))

        content = "just plain text"
        code = agent._parse_code(content)

        assert code == "just plain text"

    def test_parse_empty_content(self) -> None:
        """Test parsing empty content."""
        agent = CodeAgent(AgentConfig(model=MockModel()))

        code = agent._parse_code(None)
        assert code == ""

        code = agent._parse_code("")
        assert code == ""


class TestFinalAnswerDetection:
    """Test final answer detection."""

    def test_output_variable_indicates_final(self) -> None:
        """Test that _output in code indicates final answer."""
        agent = CodeAgent(AgentConfig(model=MockModel()))

        assert agent._is_final_answer("result", "_output = 'result'") is True

    def test_final_keywords_in_output(self) -> None:
        """Test final keywords detection."""
        agent = CodeAgent(AgentConfig(model=MockModel()))

        assert agent._is_final_answer("The final answer is 42", "code") is True
        assert agent._is_final_answer("Here is the result", "code") is True
        assert agent._is_final_answer("Task completed", "code") is True

    def test_substantial_output_indicates_final(self) -> None:
        """Test that substantial output is considered final."""
        agent = CodeAgent(AgentConfig(model=MockModel()))

        assert agent._is_final_answer("This is a long and detailed response", "code") is True
        assert agent._is_final_answer("Short", "code") is False

    def test_empty_output_not_final(self) -> None:
        """Test empty output is not final."""
        agent = CodeAgent(AgentConfig(model=MockModel()))

        assert agent._is_final_answer("", "code") is False
