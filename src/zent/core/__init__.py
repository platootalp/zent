"""Core abstractions for Zent.

This module contains the foundational ABC-based abstractions:
- BaseModel: Abstract base for LLM providers
- BaseTool: Abstract base for tools
- Agent: Abstract base for agents (Template Method pattern)
- Memory: Interface for conversation state
- Messages: Core message types
- Steps: Typed memory steps for observability
"""

from zent.core.agent import Agent, AgentConfig, Memory
from zent.core.memory import InMemoryMemory
from zent.core.messages import Message, MessageRole, ToolCall, ToolResult
from zent.core.model import BaseModel
from zent.core.steps import (
    ActionStep,
    AgentResult,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)
from zent.core.tool import BaseTool, ToolRegistry, tool

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
    # Messages
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolResult",
    # Steps
    "MemoryStep",
    "SystemPromptStep",
    "TaskStep",
    "PlanningStep",
    "ActionStep",
    "FinalAnswerStep",
    "AgentResult",
]
