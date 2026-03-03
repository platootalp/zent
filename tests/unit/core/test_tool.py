"""Tests for tool classes and registry."""

from __future__ import annotations

from typing import Any

import pytest

from zent.core.tool import BaseTool, FunctionTool, ToolRegistry, tool


class CalculatorTool(BaseTool):
    """Test tool implementation."""

    name = "calculator"
    description = "Perform calculations"

    def __init__(self) -> None:
        self._params = {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        }

    @property
    def parameters(self) -> dict[str, Any]:
        return self._params

    async def run(self, expression: str) -> str:
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class TestBaseTool:
    """Test BaseTool ABC."""

    def test_tool_name_and_description(self) -> None:
        """Test tool has name and description."""
        calc = CalculatorTool()
        assert calc.name == "calculator"
        assert calc.description == "Perform calculations"

    def test_tool_to_openai_format(self) -> None:
        """Test conversion to OpenAI format."""
        calc = CalculatorTool()
        openai_format = calc.to_openai_format()

        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "calculator"
        assert openai_format["function"]["description"] == "Perform calculations"
        assert "parameters" in openai_format["function"]

    async def test_tool_run(self) -> None:
        """Test tool execution."""
        calc = CalculatorTool()
        result = await calc.run(expression="2 + 2")
        assert result == "4"

    async def test_tool_run_error(self) -> None:
        """Test tool execution with error."""
        calc = CalculatorTool()
        result = await calc.run(expression="invalid")
        assert "Error" in result


class TestToolDecorator:
    """Test @tool decorator."""

    def test_decorator_basic(self) -> None:
        """Test basic decorator usage."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert isinstance(add, FunctionTool)
        assert add.name == "add"
        assert add.description == "Add two numbers."

    def test_decorator_with_name(self) -> None:
        """Test decorator with custom name."""

        @tool("multiply")
        def mult(x: int, y: int) -> int:
            """Multiply two numbers."""
            return x * y

        assert mult.name == "multiply"
        assert "Multiply" in mult.description

    def test_decorator_with_description(self) -> None:
        """Test decorator with custom description."""

        @tool(description="Custom description")
        def noop() -> None:
            """This should be ignored."""
            pass

        assert noop.description == "Custom description"

    def test_decorator_parameters_schema(self) -> None:
        """Test parameter schema generation."""

        @tool
        def search(query: str, limit: int = 10) -> list[str]:
            """Search items."""
            return []

        params = search.parameters
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert "limit" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["limit"]["type"] == "integer"
        assert "query" in params["required"]
        assert "limit" not in params["required"]  # Has default value

    async def test_decorator_execution_sync(self) -> None:
        """Test execution of sync function."""

        @tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        result = await greet.run(name="World")
        assert result == "Hello, World!"

    async def test_decorator_execution_async(self) -> None:
        """Test execution of async function."""

        @tool
        async def fetch(query: str) -> str:
            """Fetch data."""
            return f"Results for {query}"

        result = await fetch.run(query="test")
        assert result == "Results for test"

    async def test_decorator_result_serialization(self) -> None:
        """Test result serialization for non-string returns."""

        @tool
        def get_data() -> dict[str, Any]:
            """Get data."""
            return {"key": "value", "number": 42}

        result = await get_data.run()
        import json

        assert json.loads(result) == {"key": "value", "number": 42}


class TestToolRegistry:
    """Test ToolRegistry."""

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        registry = ToolRegistry()
        calc = CalculatorTool()

        registry.register(calc)

        assert "calculator" in registry
        assert registry.get("calculator") == calc

    def test_register_duplicate_raises(self) -> None:
        """Test registering duplicate tool raises error."""
        registry = ToolRegistry()
        calc = CalculatorTool()

        registry.register(calc)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(calc)

    def test_register_empty_name_raises(self) -> None:
        """Test registering tool with empty name raises error."""
        registry = ToolRegistry()

        class BadTool(BaseTool):
            name = ""
            description = "Bad tool"

            @property
            def parameters(self) -> dict[str, Any]:
                return {"type": "object", "properties": {}}

            async def run(self, **kwargs: Any) -> str:
                return ""

        with pytest.raises(ValueError, match="must have a name"):
            registry.register(BadTool())

    def test_get_nonexistent_returns_none(self) -> None:
        """Test getting non-existent tool returns None."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_get_all_tools(self) -> None:
        """Test getting all registered tools."""
        registry = ToolRegistry()
        calc = CalculatorTool()

        @tool
        def search(query: str) -> str:
            """Search."""
            return query

        registry.register(calc)
        registry.register(search)

        tools = registry.get_tools()
        assert len(tools) == 2
        assert calc in tools
        assert search in tools

    def test_get_openai_tools(self) -> None:
        """Test getting tools in OpenAI format."""
        registry = ToolRegistry()
        calc = CalculatorTool()

        registry.register(calc)

        openai_tools = registry.get_openai_tools()
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "calculator"

    def test_initialize_with_tools(self) -> None:
        """Test initializing registry with tools."""
        calc = CalculatorTool()
        registry = ToolRegistry([calc])

        assert "calculator" in registry
        assert len(registry) == 1

    def test_len_and_iter(self) -> None:
        """Test __len__ and __iter__."""
        registry = ToolRegistry()
        assert len(registry) == 0

        registry.register(CalculatorTool())
        assert len(registry) == 1

        items = list(registry)
        assert len(items) == 1
        assert items[0][0] == "calculator"


class TestFunctionTool:
    """Test FunctionTool directly."""

    def test_function_tool_creation(self) -> None:
        """Test creating FunctionTool."""

        def sample_func(x: int) -> str:
            """Sample function."""
            return str(x)

        ft = FunctionTool(func=sample_func, name="sample", description="A sample")

        assert ft.func == sample_func
        assert ft.name == "sample"
        assert ft.description == "A sample"

    def test_function_tool_auto_name(self) -> None:
        """Test FunctionTool auto-generates name from function."""

        def my_function():
            """Docstring."""
            pass

        ft = FunctionTool(func=my_function)
        assert ft.name == "my_function"
        assert ft.description == "Docstring."

    def test_function_tool_no_docstring(self) -> None:
        """Test FunctionTool with no docstring."""

        def no_docs():
            pass

        ft = FunctionTool(func=no_docs)
        assert ft.description == ""
