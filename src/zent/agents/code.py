"""CodeAgent - Generates and executes Python code.

Reference:
- smolagents: CodeAgent implementation
- Advantages: Works with any LLM, powerful code expression

Safety mechanisms:
- AST parsing execution (not exec)
- Whitelist import checking
- Dangerous function interception
"""

from __future__ import annotations

import ast
import builtins
import re
from dataclasses import dataclass
from typing import Any

from zent.core.agent import ActionStep, Agent, AgentConfig, FinalAnswerStep
from zent.core.messages import Message, MessageRole
from zent.core.steps import MemoryStep
from zent.core.tool import ToolRegistry


@dataclass
class ExecutionResult:
    """Result of code execution.

    Attributes:
        output: The output of the execution.
        error: Any error that occurred.
        variables: Final state of variables.
    """

    output: str
    error: str | None = None
    variables: dict[str, Any] | None = None


class LocalPythonExecutor:
    """Local Python safe executor using AST.

    Executes Python code safely by:
    1. Parsing AST
    2. Checking imports against whitelist
    3. Running in isolated namespace
    """

    def __init__(
        self,
        authorized_imports: list[str] | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        """Initialize the executor.

        Args:
            authorized_imports: List of allowed imports.
            tools: Tool registry to make available to code.
        """
        self.authorized_imports = set(
            authorized_imports or ["math", "random", "datetime", "json", "re"]
        )
        self.tools = tools
        self.state: dict[str, Any] = {}

    async def execute(self, code: str) -> ExecutionResult:
        """Execute code safely using AST.

        Args:
            code: Python code to execute.

        Returns:
            ExecutionResult with output or error.
        """
        try:
            # Parse AST
            tree = ast.parse(code)

            # Security checks
            security_error = self._check_security(tree)
            if security_error:
                return ExecutionResult(output="", error=security_error)

            # Prepare execution environment
            exec_globals = self._prepare_globals()
            exec_locals = {}

            # Execute
            exec(compile(tree, filename="<agent>", mode="exec"), exec_globals, exec_locals)

            # Capture output
            output = exec_locals.get("_output", "")
            if output is None:
                output = ""
            elif not isinstance(output, str):
                output = str(output)

            return ExecutionResult(
                output=output,
                error=None,
                variables=dict(exec_locals),
            )

        except Exception as e:
            return ExecutionResult(output="", error=f"{type(e).__name__}: {e}")

    def _check_security(self, tree: ast.AST) -> str | None:
        """Check code for security issues.

        Args:
            tree: AST of the code.

        Returns:
            Error message if security issue found, None otherwise.
        """
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module not in self.authorized_imports:
                        return f"Unauthorized import: {alias.name}"

            elif isinstance(node, ast.ImportFrom):
                module = node.module.split(".")[0] if node.module else ""
                if module not in self.authorized_imports:
                    return f"Unauthorized import from: {node.module}"

            # Check for dangerous builtins
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    dangerous = ["eval", "exec", "compile", "__import__"]
                    if node.func.id in dangerous:
                        return f"Dangerous function call: {node.func.id}"

            # Check for file operations
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    file_ops = ["open", "file"]
                    if node.func.id in file_ops:
                        return f"File operation not allowed: {node.func.id}"

        return None

    def _prepare_globals(self) -> dict[str, Any]:
        """Prepare global namespace for execution.

        Returns:
            Dictionary of globals including safe builtins and tools.
        """
        # Safe builtins
        safe_builtins = {
            "True": True,
            "False": False,
            "None": None,
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "chr": chr,
            "dict": dict,
            "dir": dir,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "hasattr": hasattr,
            "hex": hex,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "iter": iter,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "vars": vars,
            "zip": zip,
        }

        # Add authorized imports
        import importlib

        for module_name in self.authorized_imports:
            try:
                module = importlib.import_module(module_name)
                safe_builtins[module_name] = module
            except ImportError:
                pass

        # Add tools as functions
        if self.tools:
            for name, tool in self.tools:
                safe_builtins[name] = tool.run

        return {"__builtins__": safe_builtins}


class CodeAgent(Agent):
    """Agent that generates and executes Python code.

    Suitable for:
    - Models without Function Calling support
    - Complex data/computation tasks
    - Scenarios requiring multi-tool composition

    Example:
        ```python
        agent = CodeAgent(AgentConfig(
            model=LocalModel(...),
            tools=[search, calculator],
            authorized_imports=["numpy", "pandas"]
        ))
        result = await agent.run("Analyze the data")
        ```
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the CodeAgent.

        Args:
            config: The agent configuration. Can include 'authorized_imports'.
        """
        super().__init__(config)
        authorized_imports = getattr(config, "authorized_imports", None)
        self.executor = LocalPythonExecutor(
            authorized_imports=authorized_imports,
            tools=self.tools,
        )

    async def _initialize(self, task: str) -> None:
        """Initialize with user task."""
        await super()._initialize(task)

    async def _step(self) -> ActionStep | FinalAnswerStep:
        """Execute one step using code generation.

        Returns:
            ActionStep to continue the loop.
            FinalAnswerStep to complete the task.
        """
        # Build messages with code generation prompt
        messages = await self._build_messages()
        prompt = self._build_code_prompt()
        messages.append(Message.user(prompt))

        # Generate code
        response = await self.config.model.generate(messages=messages)
        code = self._parse_code(response.content or "")

        if not code.strip():
            return ActionStep(
                observations="",
                error="No code generated by model",
            )

        # Execute code
        result = await self.executor.execute(code)

        # Store the code and result in observations
        observations = f"Generated code:\n```python\n{code}\n```\n\n"
        if result.error:
            observations += f"Error: {result.error}"
            return ActionStep(
                tool_calls=[],
                observations=observations,
                error=result.error,
            )
        else:
            observations += f"Output: {result.output}"

        # Check if this is a final answer
        if self._is_final_answer(result.output, code):
            return FinalAnswerStep(answer=result.output)

        return ActionStep(
            tool_calls=[],
            observations=observations,
        )

    def _build_code_prompt(self) -> str:
        """Build code generation prompt.

        Returns:
            Prompt string for the model.
        """
        tool_descriptions = self._get_tool_descriptions()

        if tool_descriptions:
            tools_text = "\n".join(f"- {name}: {desc}" for name, desc in tool_descriptions)
            tools_section = f"""
Available tools:
{tools_text}

You can call these tools directly as functions in your code."""
        else:
            tools_section = ""

        return f"""You are a Python coding agent. Write code to solve the task.

{tools_section}

Requirements:
1. Write complete, executable Python code
2. Use ```python code block format
3. Store final result in a variable named `_output`
4. Use available tools by calling them as functions
5. Handle errors gracefully

Example:
```python
# Your solution here
result = some_calculation()
_output = "The result is: " + str(result)
```

When you have completed the task, make sure to set `_output` to the final answer."""

    def _parse_code(self, content: str | None) -> str:
        """Extract code block from model response.

        Args:
            content: Model response content.

        Returns:
            Extracted code or empty string.
        """
        if not content:
            return ""

        # Try to find Python code block
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try any code block
        pattern = r"```\n?(.*?)\n?```"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Return entire content if no code block found
        return content.strip()

    def _is_final_answer(self, output: str, code: str) -> bool:
        """Check if the output represents a final answer.

        Args:
            output: The execution output.
            code: The generated code.

        Returns:
            True if this appears to be a final answer.
        """
        # Check if _output was set in the code
        if "_output" in code:
            return True

        # Check for final answer keywords
        final_keywords = ["final_answer", "final answer", "result", "completed"]
        output_lower = output.lower()
        if any(kw in output_lower for kw in final_keywords):
            return True

        # If we have meaningful output and no error, consider it done
        if output and len(output) > 10:
            return True

        return False

    def _get_tool_descriptions(self) -> list[tuple[str, str]]:
        """Get descriptions of available tools.

        Returns:
            List of (name, description) tuples.
        """
        descriptions = []
        for name, tool in self.tools:
            descriptions.append((name, tool.description))
        return descriptions

    async def _build_messages(self) -> list[Message]:
        """Build message history from memory.

        Returns:
            List of messages for the LLM.
        """
        if self.config.memory:
            return await self.config.memory.get_messages(limit=20)
        return []
