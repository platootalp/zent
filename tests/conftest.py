"""Test configuration and fixtures."""

from __future__ import annotations

from typing import Any, AsyncIterator

import pytest

from zent.core.messages import Message, ModelResponse, ToolCall
from zent.core.model import BaseModel
from zent.core.steps import FinalAnswerStep
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
        """Return pre-configured responses in sequence."""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return ModelResponse(content="Default response", model="mock")

    async def stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream mock content."""
        yield "Mock "
        yield "response"


class MockTool(BaseTool):
    """Mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing"

    def __init__(self, return_value: str = "mock result") -> None:
        self.return_value = return_value
        self._params = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        }

    @property
    def parameters(self) -> dict[str, Any]:
        return self._params

    async def run(self, **kwargs: Any) -> str:
        return self.return_value


@pytest.fixture
def mock_model() -> MockModel:
    """Fixture providing a mock model."""
    return MockModel()


@pytest.fixture
def mock_tool() -> MockTool:
    """Fixture providing a mock tool."""
    return MockTool()


@pytest.fixture
def final_answer_response() -> ModelResponse:
    """Fixture providing a final answer response."""
    return ModelResponse(
        content="This is the final answer",
        model="mock",
        tool_calls=[],
    )


@pytest.fixture
def tool_call_response() -> ModelResponse:
    """Fixture providing a tool call response."""
    return ModelResponse(
        content=None,
        model="mock",
        tool_calls=[
            ToolCall(
                id="call_1",
                name="mock_tool",
                arguments={"query": "test"},
            )
        ],
    )
