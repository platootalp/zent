"""Example: Tool calling with Zent.

This example demonstrates how to use tools with agents.
"""

import asyncio

from zent import Message, create_agent, tool
from zent.integrations.models.openai import OpenAIModel


# Define tools using the @tool decorator
@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query.

    Returns:
        Search results.
    """
    # In a real implementation, this would call a search API
    return f"Search results for '{query}': Found 10 relevant pages."


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: The mathematical expression to evaluate.

    Returns:
        The result of the calculation.
    """
    try:
        # Safe evaluation - in production, use a proper math parser
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location.

    Args:
        location: The city or location.

    Returns:
        Current weather information.
    """
    # In a real implementation, this would call a weather API
    return f"Weather in {location}: Sunny, 72°F (22°C)"


async def main():
    """Run the tool calling example."""
    print("=" * 60)
    print("Zent Tool Calling Example")
    print("=" * 60)

    # Create an agent with tools
    agent = create_agent(
        model=OpenAIModel(model="gpt-4"),
        tools=[search_web, calculate, get_weather],
    )

    # Example 1: Simple calculation
    print("\n1. Mathematical calculation:")
    print("-" * 40)
    task = "What is 234 * 567?"
    print(f"User: {task}")

    result = await agent.run(task)
    print(f"Agent: {result.final_answer}")

    # Example 2: Multi-step reasoning
    print("\n2. Multi-step reasoning with tools:")
    print("-" * 40)
    task = "What's the weather in Tokyo and what's 25% of the temperature in Celsius?"
    print(f"User: {task}")

    result = await agent.run(task)
    print(f"Agent: {result.final_answer}")

    # Example 3: Search and calculate
    print("\n3. Search then calculate:")
    print("-" * 40)
    task = (
        "Search for Python programming and tell me how many characters are in 'Python programming'"
    )
    print(f"User: {task}")

    result = await agent.run(task)
    print(f"Agent: {result.final_answer}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
