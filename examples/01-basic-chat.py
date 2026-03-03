"""Basic chat example with Zent.

This example demonstrates the simplest way to create an agent
and have a conversation.
"""

from __future__ import annotations

import asyncio

from zent import create_agent


async def main() -> None:
    """Run a basic chat with the agent."""
    # Create a simple agent
    # Note: Requires OPENAI_API_KEY environment variable
    agent = create_agent(
        "openai:gpt-3.5-turbo",
        system_prompt="You are a helpful assistant.",
    )

    # Have a conversation
    result = await agent.run("What is the capital of France?")

    print(f"Response: {result.output}")
    print(f"Success: {result.success}")
    print(f"Steps taken: {result.step_count}")


if __name__ == "__main__":
    asyncio.run(main())
