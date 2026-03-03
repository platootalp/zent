"""Core abstractions for Zent.

This module contains the foundational ABC-based abstractions:
- BaseModel: Abstract base for LLM providers
- BaseTool: Abstract base for tools
- Agent: Abstract base for agents (Template Method pattern)
- Memory: Interface for conversation state
- Types: Core data models (Message, ToolCall, Step types, etc.)
"""

from zent.core.agent import Agent, AgentConfig
from zent.core.memory import InMemoryMemory, Memory
from zent.core.model import BaseModel
from zent.core.tool import BaseTool, ToolRegistry, tool
from zent.core.types import (
    ActionStep,
    AgentResult,
    FinalAnswerStep,
    MemoryStep,
    Message,
    MessageRole,
    ModelResponse,
    PlanningStep,
    StepType,
    SystemPromptStep,
    TaskStep,
    ToolCall,
    ToolResult,
)

__all__ = [
    # Agent
    "Agent",
    "AgentConfig",
    # Memory
    "Memory",
    "InMemoryMemory",
    # Model
    "BaseModel",
    # Tool
    "BaseTool",
    "ToolRegistry",
    "tool",
    # Types
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolResult",
    "ModelResponse",
    "MemoryStep",
    "SystemPromptStep",
    "TaskStep",
    "PlanningStep",
    "ActionStep",
    "FinalAnswerStep",
    "AgentResult",
    "StepType",
]
