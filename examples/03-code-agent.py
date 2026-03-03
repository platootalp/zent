"""Example: Code Agent with Zent.

This example demonstrates how to use the CodeAgent for Python code generation.
"""

import asyncio

from zent import Message
from zent.agents.code import CodeAgent
from zent.integrations.models.openai import OpenAIModel


async def main():
    """Run the code agent example."""
    print("=" * 60)
    print("Zent Code Agent Example")
    print("=" * 60)

    # Create a code agent
    agent = CodeAgent(
        model=OpenAIModel(model="gpt-4"),
    )

    # Example 1: Simple calculation
    print("\n1. Simple calculation:")
    print("-" * 40)
    task = "Calculate the factorial of 10"
    print(f"User: {task}")

    result = await agent.run(task)
    print(f"Agent: {result.final_answer}")
    print(f"Steps taken: {len(result.steps)}")

    # Example 2: Data processing
    print("\n2. Data processing with Python:")
    print("-" * 40)
    task = "Create a list of the first 10 prime numbers"
    print(f"User: {task}")

    result = await agent.run(task)
    print(f"Agent: {result.final_answer}")

    # Example 3: String manipulation
    print("\n3. String manipulation:")
    print("-" * 40)
    task = "Reverse the string 'Hello World' and convert to uppercase"
    print(f"User: {task}")

    result = await agent.run(task)
    print(f"Agent: {result.final_answer}")

    # Example 4: Complex calculation
    print("\n4. Complex calculation:")
    print("-" * 40)
    task = "Calculate the sum of squares from 1 to 100"
    print(f"User: {task}")

    result = await agent.run(task)
    print(f"Agent: {result.final_answer}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
