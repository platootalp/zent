"""OpenAI model implementation.

Reference:
- LangChain: ChatOpenAI
- smolagents: OpenAIServerModel
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

from zent.core.messages import Message, ModelResponse, ToolCall
from zent.core.model import BaseModel


try:
    from openai import AsyncOpenAI

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAIModel(BaseModel):
    """OpenAI model implementation.

    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.

    Example:
        ```python
        model = OpenAIModel(model="gpt-4", temperature=0.7)
        response = await model.generate(messages=[Message.user("Hello")])
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI model.

        Args:
            model: Model name (e.g., "gpt-4", "gpt-3.5-turbo").
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
            base_url: Optional custom base URL for API.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Raises:
            ImportError: If openai package is not installed.
        """
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI support requires the 'openai' package. Install with: pip install openai"
            )

        super().__init__(model=model, **kwargs)

        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Zent messages to OpenAI format.

        Args:
            messages: List of Zent messages.

        Returns:
            List of OpenAI message dictionaries.
        """
        return [msg.to_dict() for msg in messages]

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response using OpenAI API.

        Args:
            messages: Conversation history.
            tools: Available tools.
            **kwargs: Additional parameters.

        Returns:
            ModelResponse with content and/or tool calls.
        """
        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            **self.params,
            **kwargs,
        }

        # Add tools if provided
        if tools:
            params["tools"] = [tool.to_openai_format() for tool in tools]

        # Make API call
        response = await self.client.chat.completions.create(**params)

        choice = response.choices[0]
        message = choice.message

        # Extract tool calls
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return ModelResponse(
            content=message.content,
            tool_calls=tool_calls,
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    async def stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from OpenAI.

        Args:
            messages: Conversation history.
            **kwargs: Additional parameters.

        Yields:
            Text chunks as they're generated.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "stream": True,
            **self.params,
            **kwargs,
        }

        stream = await self.client.chat.completions.create(**params)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AzureOpenAIModel(OpenAIModel):
    """Azure OpenAI model implementation.

    Inherits from OpenAIModel with Azure-specific configuration.
    """

    def __init__(
        self,
        model: str,
        azure_endpoint: str,
        api_key: str | None = None,
        api_version: str = "2024-02-01",
        **kwargs: Any,
    ) -> None:
        """Initialize Azure OpenAI model.

        Args:
            model: Deployment name (not model name).
            azure_endpoint: Azure OpenAI endpoint URL.
            api_key: Azure API key.
            api_version: API version.
            **kwargs: Additional parameters.
        """
        if not HAS_OPENAI:
            raise ImportError(
                "Azure OpenAI support requires the 'openai' package. "
                "Install with: pip install openai"
            )

        # Initialize BaseModel directly to avoid double client creation
        BaseModel.__init__(self, model=model, **kwargs)

        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=f"{azure_endpoint}/openai/deployments/{model}",
            default_query={"api-version": api_version},
        )
