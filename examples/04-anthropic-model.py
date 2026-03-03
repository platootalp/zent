"""Example: Using Anthropic models with Zent.

This example demonstrates how to use Claude models.
"""

import asyncio

from zent import Message, create_agent, tool
from zent.integrations.models.anthropic import AnthropicModel


@tool
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def count_words(text: str) -> str:
    """Count words in text.

    Args:
        text: The text to count words in.

    Returns:
        Word count.
    """
    words = text.split()
    return f"Word count: {len(words)}"


async def main():
    """Run the Anthropic model example."""
    print("=" * 60)
    print("Zent Anthropic Model Example")
    print("=" * 60)

    # Create agent with Anthropic model
    agent = create_agent(
        model=AnthropicModel(model="claude-3-sonnet-20240229"),
        tools=[get_current_time, count_words],
    )

    # Example 1: Simple conversation
    print("\n1. Simple conversation:")
    print("-" * 40)
    task = "What are three interesting facts about Python programming?"
    print(f"User: {task}")

    result = await agent.run(task)
    print(f"Claude: {result.final_answer}")

    # Example 2: Tool usage
    print("\n2. Tool usage with Claude:")
    print("-" * 40)
    task = "What time is it now? Also count the words in this sentence."
    print(f"User: {task}")

    result = await agent.run(task)
    print(f"Claude: {result.final_answer}")

    # Example 3: Streaming
    print("\n3. Streaming response:")
    print("-" * 40)
    task = "Write a haiku about coding"
    print(f"User: {task}")
    print("Claude: ", end="", flush=True)

    model = AnthropicModel(model="claude-3-haiku-20240307")
    async for chunk in model.stream([Message.user(task)]):
        print(chunk, end="", flush=True)
    print()

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
