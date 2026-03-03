"""Anthropic model implementation.

Reference:
- LangChain: ChatAnthropic
- smolagents: AnthropicModel
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

from zent.core.messages import Message, MessageRole, ModelResponse, ToolCall
from zent.core.model import BaseModel


try:
    from anthropic import AsyncAnthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class AnthropicModel(BaseModel):
    """Anthropic model implementation.

    Supports Claude 3 models (Opus, Sonnet, Haiku) and other Anthropic models.

    Example:
        ```python
        model = AnthropicModel(model="claude-3-sonnet-20240229", max_tokens=1024)
        response = await model.generate(messages=[Message.user("Hello")])
        ```
    """

    def __init__(
        self,
        model: str = "claude-3-sonnet-20240229",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Anthropic model.

        Args:
            model: Model name (e.g., "claude-3-opus-20240229").
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
            base_url: Optional custom base URL for API.
            **kwargs: Additional parameters (max_tokens, temperature, etc.).

        Raises:
            ImportError: If anthropic package is not installed.
        """
        if not HAS_ANTHROPIC:
            raise ImportError(
                "Anthropic support requires the 'anthropic' package. "
                "Install with: pip install anthropic"
            )

        super().__init__(model=model, **kwargs)

        self.client = AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url,
        )

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert Zent messages to Anthropic format.

        Anthropic uses a different format where:
        - System prompt is a separate top-level parameter
        - Messages are a list with role and content
        - Content is a list of content blocks

        Args:
            messages: List of Zent messages.

        Returns:
            Tuple of (system_prompt, anthropic_messages).
        """
        system_prompt: str | None = None
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Anthropic expects system as a separate parameter
                # If multiple system messages, concatenate them
                if system_prompt is None:
                    system_prompt = msg.content
                else:
                    system_prompt += "\n\n" + msg.content
            elif msg.role == MessageRole.TOOL:
                # Tool results are sent as user messages with tool_result content
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.metadata.get("tool_call_id", ""),
                                "content": msg.content,
                            }
                        ],
                    }
                )
            else:
                # User and assistant messages
                # For assistant messages with tool calls, we need to handle specially
                if msg.role == MessageRole.ASSISTANT and msg.metadata.get("tool_calls"):
                    # Convert tool calls to Anthropic format
                    content_blocks: list[dict[str, Any]] = []

                    # Add text content if present
                    if msg.content:
                        content_blocks.append(
                            {
                                "type": "text",
                                "text": msg.content,
                            }
                        )

                    # Add tool_use blocks
                    for tc in msg.metadata["tool_calls"]:
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc["id"],
                                "name": tc["function"]["name"],
                                "input": json.loads(tc["function"]["arguments"]),
                            }
                        )

                    anthropic_messages.append(
                        {
                            "role": "assistant",
                            "content": content_blocks,
                        }
                    )
                else:
                    # Standard text message
                    anthropic_messages.append(
                        {
                            "role": msg.role.value,
                            "content": msg.content,
                        }
                    )

        return system_prompt, anthropic_messages

    def _convert_tools(self, tools: list) -> list[dict[str, Any]]:
        """Convert tools to Anthropic format.

        Args:
            tools: List of BaseTool instances.

        Returns:
            List of tool definitions in Anthropic format.
        """
        anthropic_tools = []
        for tool in tools:
            openai_format = tool.to_openai_format()
            # Convert from OpenAI format to Anthropic format
            anthropic_tools.append(
                {
                    "name": openai_format["function"]["name"],
                    "description": openai_format["function"]["description"],
                    "input_schema": openai_format["function"]["parameters"],
                }
            )
        return anthropic_tools

    def _parse_response(self, response: Any) -> ModelResponse:
        """Parse Anthropic response into ModelResponse.

        Args:
            response: Anthropic API response.

        Returns:
            ModelResponse with content and/or tool calls.
        """
        content_text = ""
        tool_calls: list[ToolCall] = []

        # Anthropic returns content as a list of blocks
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        return ModelResponse(
            content=content_text if content_text else None,
            tool_calls=tool_calls,
            model=response.model,
            finish_reason=response.stop_reason,
        )

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a response using Anthropic API.

        Args:
            messages: Conversation history.
            tools: Available tools.
            **kwargs: Additional parameters.

        Returns:
            ModelResponse with content and/or tool calls.

        Raises:
            Exception: If API call fails.
        """
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.params.get("max_tokens", 1024),
            **{k: v for k, v in self.params.items() if k != "max_tokens"},
            **kwargs,
        }

        # Add system prompt if present
        if system_prompt:
            params["system"] = system_prompt

        # Add tools if provided
        if tools:
            params["tools"] = self._convert_tools(tools)

        # Make API call
        try:
            response = await self.client.messages.create(**params)
        except Exception as e:
            raise Exception(f"Anthropic API error: {e}") from e

        return self._parse_response(response)

    async def stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from Anthropic.

        Args:
            messages: Conversation history.
            **kwargs: Additional parameters.

        Yields:
            Text chunks as they're generated.

        Raises:
            Exception: If API call fails.
        """
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Build request parameters
        params: dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": self.params.get("max_tokens", 1024),
            "stream": True,
            **{k: v for k, v in self.params.items() if k != "max_tokens"},
            **kwargs,
        }

        # Add system prompt if present
        if system_prompt:
            params["system"] = system_prompt

        # Make streaming API call
        try:
            async with self.client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise Exception(f"Anthropic streaming error: {e}") from e
