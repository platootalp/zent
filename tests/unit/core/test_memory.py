"""Tests for memory implementations."""

from __future__ import annotations

import pytest

from zent.core.messages import MessageRole, ToolCall
from zent.core.steps import ActionStep, FinalAnswerStep, MemoryStep, TaskStep
from zent.core.memory import InMemoryMemory


class TestInMemoryMemory:
    """Test InMemoryMemory implementation."""

    async def test_add_step(self) -> None:
        """Test adding a step to memory."""
        memory = InMemoryMemory()
        step = TaskStep(task="Test task")

        await memory.add(step)

        steps = await memory.get_steps()
        assert len(steps) == 1
        assert steps[0] == step

    async def test_get_steps_limit(self) -> None:
        """Test get_steps respects limit."""
        memory = InMemoryMemory()

        for i in range(5):
            await memory.add(TaskStep(task=f"Task {i}"))

        steps = await memory.get_steps(limit=3)
        assert len(steps) == 3
        # Should get the most recent 3
        assert steps[0].task == "Task 2"
        assert steps[2].task == "Task 4"

    async def test_clear_memory(self) -> None:
        """Test clearing memory."""
        memory = InMemoryMemory()
        await memory.add(TaskStep(task="Test"))

        await memory.clear()

        steps = await memory.get_steps()
        assert len(steps) == 0

    async def test_max_steps_limit(self) -> None:
        """Test max_steps enforcement."""
        memory = InMemoryMemory(max_steps=3)

        for i in range(5):
            await memory.add(TaskStep(task=f"Task {i}"))

        steps = await memory.get_steps(limit=100)
        assert len(steps) == 3
        # Should only have the most recent 3
        assert steps[0].task == "Task 2"
        assert steps[2].task == "Task 4"

    async def test_len_method(self) -> None:
        """Test __len__ method."""
        memory = InMemoryMemory()
        assert len(memory) == 0

        await memory.add(TaskStep(task="Test"))
        assert len(memory) == 1

    async def test_get_messages_task_step(self) -> None:
        """Test converting TaskStep to message."""
        memory = InMemoryMemory()
        await memory.add(TaskStep(task="Hello"))

        messages = await memory.get_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "Hello"

    async def test_get_messages_action_step_with_tools(self) -> None:
        """Test converting ActionStep with tools to message."""
        memory = InMemoryMemory()
        step = ActionStep(
            tool_calls=[ToolCall(id="call_1", name="search", arguments={"query": "test"})],
            observations="Results",
        )
        await memory.add(step)

        messages = await memory.get_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.ASSISTANT
        assert messages[0].metadata.get("tool_calls")

    async def test_get_messages_final_answer(self) -> None:
        """Test converting FinalAnswerStep to message."""
        memory = InMemoryMemory()
        await memory.add(FinalAnswerStep(answer="The answer is 42"))

        messages = await memory.get_messages()
        assert len(messages) == 1
        assert messages[0].role == MessageRole.ASSISTANT
        assert messages[0].content == "The answer is 42"

    async def test_get_messages_mixed_steps(self) -> None:
        """Test converting mixed steps to messages."""
        memory = InMemoryMemory()

        await memory.add(TaskStep(task="What is the weather?"))
        await memory.add(
            ActionStep(
                tool_calls=[ToolCall(id="1", name="weather", arguments={"city": "Paris"})],
                observations="Sunny",
            )
        )
        await memory.add(FinalAnswerStep(answer="It's sunny in Paris"))

        messages = await memory.get_messages()
        assert len(messages) == 3
        assert messages[0].role == MessageRole.USER
        assert messages[1].role == MessageRole.ASSISTANT
        assert messages[2].role == MessageRole.ASSISTANT

    async def test_get_messages_limit(self) -> None:
        """Test get_messages respects limit."""
        memory = InMemoryMemory()

        for i in range(5):
            await memory.add(TaskStep(task=f"Task {i}"))

        messages = await memory.get_messages(limit=2)
        assert len(messages) == 2
        # Should get the most recent 2
        assert messages[0].content == "Task 3"
        assert messages[1].content == "Task 4"

    async def test_empty_memory(self) -> None:
        """Test operations on empty memory."""
        memory = InMemoryMemory()

        steps = await memory.get_steps()
        assert steps == []

        messages = await memory.get_messages()
        assert messages == []

        assert len(memory) == 0
