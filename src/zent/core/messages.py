"""Message data models for Zent.

This module re-exports from types for backward compatibility.
New code should import from zent.core.types directly.
"""

# Re-export from types for backward compatibility
from zent.core.types import (
    Message,
    MessageRole,
    ModelResponse,
    ToolCall,
    ToolResult,
)

__all__ = [
    "Message",
    "MessageRole",
    "ModelResponse",
    "ToolCall",
    "ToolResult",
]
