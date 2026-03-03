"""Base tool class for Zent.

Reference:
- LangChain: BaseTool, StructuredTool
- smolagents: Tool base class

Uses ABC pattern with both class-based and decorator-based creation.
"""

from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints


class BaseTool(ABC):
    """Abstract base class for tools.

    Similar to LangChain's BaseTool and smolagents' Tool.

    Tools can be created by:
    1. Inheriting from BaseTool
    2. Using the @tool decorator on a function

    Example:
        ```python
        # Method 1: Class inheritance
        class Calculator(BaseTool):
            name = "calculator"
            description = "Perform calculations"

            @property
            def parameters(self) -> dict:
                return {...}

            async def run(self, expression: str) -> str:
                return str(eval(expression))

        # Method 2: Decorator
        @tool
        def search(query: str) -> str:
            \"\"\"Search the web\"\"\"
            return f"Results for {query}"
        ```
    """

    name: str = ""
    description: str = ""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for the tool's parameters.

        Returns:
            Dict following JSON Schema format.
        """
        pass

    @abstractmethod
    async def run(self, **kwargs: Any) -> str:
        """Execute the tool.

        Args:
            **kwargs: The arguments for the tool.

        Returns:
            The result as a string.
        """
        pass

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class FunctionTool(BaseTool):
    """Tool created from a function.

    Similar to LangChain's StructuredTool.from_function().

    Attributes:
        func: The function to wrap.
        name: Tool name (defaults to function name).
        description: Tool description (defaults to docstring).
    """

    func: Callable = field(repr=False)
    name: str = ""
    description: str = ""
    _parameters: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.func.__name__
        if not self.description:
            self.description = inspect.getdoc(self.func) or ""

    @property
    def parameters(self) -> dict[str, Any]:
        """Generate JSON Schema from function signature."""
        if self._parameters is not None:
            return self._parameters

        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)

        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = type_hints.get(param_name, str)
            json_type = _python_type_to_json_schema(param_type)

            properties[param_name] = json_type

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        self._parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return self._parameters

    async def run(self, **kwargs: Any) -> str:
        """Execute the wrapped function."""
        if inspect.iscoroutinefunction(self.func):
            result = await self.func(**kwargs)
        else:
            result = self.func(**kwargs)

        # Convert result to string
        if isinstance(result, str):
            return result
        return json.dumps(result)


def tool(
    name_or_func: str | Callable | None = None,
    *,
    description: str | None = None,
) -> Callable | FunctionTool:
    """Decorator to convert a function into a Tool.

    Similar to LangChain's @tool decorator and smolagents' @tool.

    Supports three usage patterns:
    1. @tool - No arguments, uses function name and docstring
    2. @tool() - Empty parentheses, same as @tool
    3. @tool("custom_name") or @tool(description="...") - With custom metadata

    Args:
        name_or_func: Either the tool name, the function, or None.
        description: Custom description (overrides docstring).

    Returns:
        A Tool instance or decorator.
    """

    def decorator(func: Callable, name: str | None = None) -> FunctionTool:
        tool_name = name if name else func.__name__
        tool_desc = description or inspect.getdoc(func) or ""

        return FunctionTool(
            func=func,
            name=tool_name,
            description=tool_desc,
        )

    # Handle @tool without parentheses: @tool
    if callable(name_or_func):
        func = name_or_func
        return decorator(func)

    # Handle @tool() or @tool("name") with description
    tool_name = name_or_func if isinstance(name_or_func, str) else None

    def inner_decorator(func: Callable) -> FunctionTool:
        return decorator(func, name=tool_name)

    return inner_decorator


def _python_type_to_json_schema(py_type: type) -> dict[str, Any]:
    """Convert Python type to JSON Schema type definition.

    Args:
        py_type: Python type annotation.

    Returns:
        JSON Schema type definition.
    """
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    return type_map.get(py_type, {"type": "string"})


class ToolRegistry:
    """Registry for managing multiple tools.

    Similar to LangChain's tool registry concept.
    """

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        """Initialize registry.

        Args:
            tools: Optional list of tools to register.
        """
        self._tools: dict[str, BaseTool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Register a tool.

        Args:
            tool: The tool to register.

        Raises:
            ValueError: If tool name is empty or duplicate.
        """
        if not tool.name:
            raise ValueError("Tool must have a name")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name.

        Args:
            name: The tool name.

        Returns:
            The tool or None if not found.
        """
        return self._tools.get(name)

    def get_tools(self) -> list[BaseTool]:
        """Get all registered tools.

        Returns:
            List of all tools.
        """
        return list(self._tools.values())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get all tools in OpenAI format.

        Returns:
            List of tool definitions for OpenAI API.
        """
        return [tool.to_openai_format() for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __len__(self) -> int:
        """Count of registered tools."""
        return len(self._tools)

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.items())
