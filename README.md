# Zent v4

A lightweight, extensible Python framework for building LLM-powered agents.

## Features

- **Simple API**: Get started in minutes with the `@tool` decorator and `create_agent()` factory
- **Extensible Architecture**: ABC-based design for custom models, tools, and agents
- **Multi-Provider Support**: OpenAI, Anthropic, and custom model integrations
- **Tool Calling**: First-class support for LLM function calling
- **Code Execution**: Built-in CodeAgent for safe Python code execution
- **Streaming**: Real-time response streaming
- **Type Safe**: Full type hints and async/await support

## Installation

```bash
# Basic installation (core only)
pip install zent

# With OpenAI support
pip install zent[openai]

# With Anthropic support
pip install zent[anthropic]

# With all integrations
pip install zent[openai,anthropic]

# Development
pip install zent[dev]
```

## Quick Start

### Basic Chat

```python
import asyncio
from zent import create_agent
from zent.integrations.models.openai import OpenAIModel

async def main():
    agent = create_agent(
        model=OpenAIModel(model="gpt-4"),
    )

    result = await agent.run("What is the capital of France?")
    print(result.final_answer)

asyncio.run(main())
```

### Tools

```python
from zent import tool, create_agent
from zent.integrations.models.openai import OpenAIModel

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

async def main():
    agent = create_agent(
        model=OpenAIModel(model="gpt-4"),
        tools=[calculate, get_weather],
    )

    result = await agent.run("What's 234 * 567? And what's the weather in Tokyo?")
    print(result.final_answer)

asyncio.run(main())
```

### Code Agent

```python
from zent.agents.code import CodeAgent
from zent.integrations.models.openai import OpenAIModel

async def main():
    agent = CodeAgent(
        model=OpenAIModel(model="gpt-4"),
    )

    result = await agent.run("Calculate the factorial of 20")
    print(result.final_answer)

asyncio.run(main())
```

### Streaming

```python
from zent import Message
from zent.integrations.models.openai import OpenAIModel

async def main():
    model = OpenAIModel(model="gpt-4")

    async for chunk in model.stream([Message.user("Write a haiku about coding")]):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

## Architecture

Zent v4 uses a layered architecture with ABC-based abstractions:

```
┌─────────────────────────────────────┐
│  App Layer (Factory, Config)        │
├─────────────────────────────────────┤
│  Agent Layer (ToolCalling, Code)    │
├─────────────────────────────────────┤
│  Core Layer (Model, Tool, Memory)   │
├─────────────────────────────────────┤
│  Integration Layer (OpenAI, etc.)   │
└─────────────────────────────────────┘
```

### Core Abstractions

- **BaseModel**: Abstract base for LLM providers (OpenAI, Anthropic, etc.)
- **BaseTool**: Abstract base for tools with `@tool` decorator support
- **Agent**: Abstract base for agent implementations using Template Method pattern
- **Memory**: Interface for conversation state management

## Examples

See the `examples/` directory for more:

- `01-basic-chat.py` - Simple conversation
- `02-tool-calling.py` - Using tools with agents
- `03-code-agent.py` - Code generation and execution
- `04-anthropic-model.py` - Using Claude models

## Custom Model

```python
from zent.core.model import BaseModel
from zent.core.messages import Message, ModelResponse

class MyModel(BaseModel):
    async def generate(self, messages, tools=None, **kwargs):
        # Your implementation
        return ModelResponse(content="Hello!")

    async def stream(self, messages, **kwargs):
        yield "Hello"
        yield " World!"
```

## Custom Tool

```python
from zent.core.tool import BaseTool

class MyTool(BaseTool):
    @property
    def name(self):
        return "my_tool"

    @property
    def description(self):
        return "Does something useful"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"]
        }

    async def run(self, input: str):
        return f"Processed: {input}"
```

## Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=zent

# Type check
mypy src/zent

# Lint
ruff check src/zent
```

## License

MIT License

## Credits

Zent v4 design inspired by:
- [LangChain](https://github.com/langchain-ai/langchain) - Runnable protocol, LCEL
- [smolagents](https://github.com/huggingface/smolagents) - Template method, typed memory
