"""Base model class for Zent.

Reference:
- LangChain: BaseChatModel, BaseLanguageModel
- smolagents: Model ABC

Uses ABC pattern instead of Protocol for better extensibility
and shared implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator

from zent.core.messages import Message, ModelResponse

if TYPE_CHECKING:
    from zent.core.tool import BaseTool


class BaseModel(ABC):
    """Abstract base class for language models.

    Similar to LangChain's BaseChatModel but simplified.
    Any LLM provider should inherit from this class.

    Example:
        ```python
        class OpenAIModel(BaseModel):
            async def generate(self, messages, tools=None, **kwargs):
                # Implementation
                pass

            async def stream(self, messages, **kwargs):
                # Implementation
                pass
        ```
    """

    def __init__(self, model: str = "", **kwargs: Any) -> None:
        """Initialize the model.

        Args:
            model: The model name/identifier.
            **kwargs: Additional provider-specific parameters.
        """
        self.model = model
        self.params = kwargs

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list["BaseTool"] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response from the model.

        Args:
            messages: The conversation history.
            tools: Optional list of available tools.
            **kwargs: Additional provider-specific parameters.

        Returns:
            The model's response.
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from the model.

        Args:
            messages: The conversation history.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Chunks of the generated text.
        """
        pass

    def bind_tools(self, tools: list["BaseTool"]) -> BoundModel:
        """Bind tools to this model for automatic tool calling.

        Similar to LangChain's bind_tools() method.

        Args:
            tools: List of tools to bind.

        Returns:
            A BoundModel with tools pre-configured.
        """
        return BoundModel(self, tools)


class BoundModel:
    """A model with pre-bound tools.

    Similar to LangChain's RunnableBinding.
    """

    def __init__(self, model: BaseModel, tools: list["BaseTool"]) -> None:
        self._model = model
        self._tools = tools

    async def generate(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate with bound tools."""
        return await self._model.generate(messages, self._tools, **kwargs)

    async def stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream with bound tools."""
        async for chunk in self._model.stream(messages, **kwargs):
            yield chunk
