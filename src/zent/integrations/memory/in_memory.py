"""In-memory memory implementation.

Reference:
- smolagents: AgentMemory
- LangChain: ConversationBufferMemory

Simple in-memory storage for development and testing.
For production, consider persistent storage implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from zent.core.agent import Memory
from zent.core.messages import Message, MessageRole
from zent.core.steps import (
    ActionStep,
    FinalAnswerStep,
    MemoryStep,
    StepType,
    TaskStep,
)


@dataclass
class InMemoryMemory(Memory):
    """Simple in-memory memory storage.

    Attributes:
        max_steps: Maximum number of steps to store (oldest are dropped).
    """

    max_steps: int = 100
    _steps: list[MemoryStep] = field(default_factory=list, repr=False)

    async def add(self, step: MemoryStep) -> None:
        """Add a step to memory.

        Args:
            step: The step to add.
        """
        self._steps.append(step)

        # Enforce max steps limit
        if len(self._steps) > self.max_steps:
            self._steps = self._steps[-self.max_steps :]

    async def get_messages(self, limit: int = 10) -> list[Message]:
        """Convert steps to messages format.

        This conversion allows steps to be used as LLM context.

        Args:
            limit: Maximum number of steps to convert.

        Returns:
            List of messages.
        """
        messages = []

        for step in self._steps[-limit:]:
            if isinstance(step, TaskStep):
                messages.append(Message.user(step.task))

            elif isinstance(step, ActionStep):
                # Tool calls become assistant message
                if step.tool_calls:
                    messages.append(
                        Message.assistant(
                            content="",
                            tool_calls=[call.to_dict() for call in step.tool_calls],
                        )
                    )

            elif isinstance(step, FinalAnswerStep):
                messages.append(Message.assistant(step.answer))

        return messages

    async def get_steps(self, limit: int = 10) -> list[MemoryStep]:
        """Get raw steps.

        Args:
            limit: Maximum number of steps to return.

        Returns:
            List of memory steps.
        """
        return self._steps[-limit:] if self._steps else []

    async def clear(self) -> None:
        """Clear all stored steps."""
        self._steps.clear()

    def __len__(self) -> int:
        """Return the number of stored steps."""
        return len(self._steps)
