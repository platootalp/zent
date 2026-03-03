# Zent Agent 框架设计文档 V3

> **核心理念**: 极简主义，聚焦核心，渐进增强

## 1. 设计原则

### 1.1 核心原则
- **极简主义**: 只保留真正必要的抽象
- **显式优于隐式**: 配置和依赖显式声明
- **渐进增强**: 从简单开始，按需叠加复杂度
- **协议优先**: Python Protocol 而非抽象基类

### 1.2 架构决策
- **3层架构**: Core → Integrations → App
- **4个核心抽象**: Model, Tool, Memory, Agent
- **核心层零依赖**: 仅 Python 标准库 + typing
- **异步优先**: 所有 IO 操作都是 async
- **MCP 原生**: 工具协议与 MCP 对齐

---

## 2. 系统架构

### 2.1 三层架构

```
┌─────────────────────────────────────────────────────────────┐
│                     应用层 (App)                             │
│  @tool, @agent, create_agent(), 配置驱动                     │
├─────────────────────────────────────────────────────────────┤
│                     集成层 (Integrations)                    │
│  OpenAI, Anthropic, MCP, Memory 实现                         │
├─────────────────────────────────────────────────────────────┤
│                     核心层 (Core)                            │
│  Model, Tool, Memory, Agent (Protocols + 实现)               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心概念

```python
# 核心关系图
Agent
├── model: Model        # 语言模型 (原 LLM)
├── tools: List[Tool]   # 工具集
├── memory: Memory      # 记忆（可选）
└── config: AgentConfig # 配置

# Agent 是协调器，直接管理工作流循环
# 无需单独的 Workflow/Runtime 层
```

---

## 3. 项目结构

```
src/zent/
├── __init__.py              # 公开 API
├── __version__.py           # 版本信息
│
├── core/                    # 核心层（零外部依赖）
│   ├── __init__.py
│   ├── protocols.py         # Model, Tool, Memory Protocols
│   ├── agent.py             # Agent 实现（含 ReAct 循环）
│   ├── messages.py          # Message, MessageRole
│   └── result.py            # AgentResult, ToolResult
│
├── integrations/            # 集成层
│   ├── __init__.py
│   ├── openai.py            # OpenAI 模型适配
│   ├── anthropic.py         # Claude 模型适配
│   ├── mcp.py               # MCP 工具适配
│   └── memory.py            # 内存实现
│
└── app/                     # 应用层
    ├── __init__.py
    ├── decorators.py        # @tool, @agent
    └── factory.py           # create_agent()

tests/
├── unit/                    # 单元测试
├── integration/             # 集成测试
└── e2e/                     # 端到端测试

examples/
├── 01-basic-chat.py
├── 02-tools.py
├── 03-memory.py
├── 04-mcp-tools.py
└── 05-multi-agent.py
```

**v2 → v3 简化点**:
- ❌ workflow/ → ✅ 合并到 core/agent.py
- ❌ capabilities/ → ✅ 合并到 core/protocols.py
- ❌ runtime/ → ✅ 删除（EventBus, Checkpoint 作为可选中间件）
- ❌ adapters/ → ✅ 重命名为 integrations/
- ❌ ext/ → ✅ 删除（使用标准 Python 导入）
- ❌ utils/ → ✅ 删除（YAGNI）

---

## 4. 核心 API 设计

### 4.1 Protocol 定义（zent/core/protocols.py）

```python
from typing import Protocol, runtime_checkable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum

# ==================== 基础类型 ====================

class MessageRole(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class Message:
    """对话消息"""
    role: MessageRole
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ToolCall:
    """工具调用请求"""
    id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    """工具执行结果"""
    call_id: str
    output: str
    is_error: bool = False


@dataclass
class ModelResponse:
    """模型响应"""
    content: str | None
    tool_calls: list[ToolCall]
    model: str
    finish_reason: str | None = None


# ==================== 核心 Protocols ====================

@runtime_checkable
class Model(Protocol):
    """语言模型协议 - 简化为最简接口"""
    
    async def generate(
        self,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        **kwargs
    ) -> ModelResponse: ...
    
    async def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class Tool(Protocol):
    """工具协议 - MCP 对齐"""
    
    @property
    def name(self) -> str: ...
    
    @property
    def description(self) -> str: ...
    
    @property
    def parameters(self) -> dict: ...  # JSON Schema
    
    async def run(self, **kwargs) -> str: ...


@runtime_checkable
class Memory(Protocol):
    """记忆协议 - 最简接口"""
    
    async def add(self, message: Message) -> None: ...
    
    async def get(self, limit: int = 10) -> list[Message]: ...
    
    async def clear(self) -> None: ...
```

### 4.2 Agent 实现（zent/core/agent.py）

```python
"""Agent 实现 - 包含 ReAct 循环"""
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import asyncio

from zent.core.protocols import (
    Model, Tool, Memory, Message, MessageRole,
    ToolCall, ToolResult, ModelResponse
)
from zent.core.result import AgentResult


@dataclass
class AgentConfig:
    """Agent 配置"""
    model: Model
    tools: List[Tool] = field(default_factory=list)
    memory: Optional[Memory] = None
    max_iterations: int = 10
    system_prompt: Optional[str] = None
    
    # 可选回调（中间件模式）
    on_step: Optional[Callable[[str, dict], None]] = None


class Agent:
    """Agent 协调器 - 简化的 ReAct 实现
    
    Usage:
        agent = Agent(AgentConfig(model=OpenAIModel(...)))
        result = await agent.run("查询天气")
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools_map = {t.name: t for t in config.tools}
        self.messages: list[Message] = []
        
        if config.system_prompt:
            self.messages.append(
                Message(role=MessageRole.SYSTEM, content=config.system_prompt)
            )
    
    async def run(self, query: str, **context) -> AgentResult:
        """执行单次任务 - ReAct 循环"""
        # 添加用户查询
        self.messages.append(
            Message(role=MessageRole.USER, content=query)
        )
        
        try:
            for iteration in range(self.config.max_iterations):
                # 1. 调用模型
                response = await self.config.model.generate(
                    messages=self.messages,
                    tools=self.config.tools if self.config.tools else None
                )
                
                # 2. 检查是否需要调用工具
                if response.tool_calls:
                    # 添加助手消息（包含 tool_calls）
                    self.messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content="",
                            metadata={"tool_calls": response.tool_calls}
                        )
                    )
                    
                    # 执行工具
                    for tool_call in response.tool_calls:
                        result = await self._execute_tool(tool_call)
                        
                        # 添加工具结果
                        self.messages.append(
                            Message(
                                role=MessageRole.TOOL,
                                content=result.output,
                                metadata={"tool_call_id": tool_call.id}
                            )
                        )
                        
                        if self.config.on_step:
                            self.config.on_step("tool", {
                                "call": tool_call,
                                "result": result
                            })
                
                # 3. 返回最终答案
                elif response.content:
                    self.messages.append(
                        Message(role=MessageRole.ASSISTANT, content=response.content)
                    )
                    
                    # 保存到记忆
                    if self.config.memory:
                        for msg in self.messages[-3:]:  # 最近 3 条
                            await self.config.memory.add(msg)
                    
                    return AgentResult(
                        output=response.content,
                        messages=self.messages.copy(),
                        success=True
                    )
            
            return AgentResult(
                output="达到最大迭代次数",
                messages=self.messages.copy(),
                success=False
            )
            
        except Exception as e:
            return AgentResult(
                output=str(e),
                messages=self.messages.copy(),
                success=False,
                error=e
            )
    
    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """执行工具调用"""
        tool = self.tools_map.get(call.name)
        
        if not tool:
            return ToolResult(
                call_id=call.id,
                output=f"工具 '{call.name}' 不存在",
                is_error=True
            )
        
        try:
            output = await tool.run(**call.arguments)
            return ToolResult(call_id=call.id, output=output)
        except Exception as e:
            return ToolResult(
                call_id=call.id,
                output=f"错误: {e}",
                is_error=True
            )
    
    async def stream(self, query: str, **context):
        """流式执行（简化版）"""
        result = await self.run(query, **context)
        yield result.output
```

### 4.3 应用层（zent/app/decorators.py）

```python
"""装饰器 - 简化工具创建"""
import inspect
from typing import Callable

from zent.core.protocols import Tool
from zent.integrations.mcp import FunctionTool  # MCP 格式的工具实现


def tool(name: str | None = None, description: str | None = None):
    """工具装饰器 - 将函数转换为 Tool
    
    Usage:
        @tool
        def search(query: str) -> str:
            \"\"\"搜索知识库\"\"\"
            return f"Results: {query}"
        
        @tool(name="custom_search")
        def my_search(query: str) -> str:
            return f"Results: {query}"
    """
    def decorator(func: Callable) -> Tool:
        return FunctionTool(
            func=func,
            name=name or func.__name__,
            description=description or inspect.getdoc(func) or ""
        )
    
    # 支持 @tool 无括号
    if callable(name):
        func = name
        name = None
        return decorator(func)
    
    return decorator
```

`zent/app/factory.py`:

```python
"""工厂函数 - 简化 Agent 创建"""
from zent.core.agent import Agent, AgentConfig
from zent.core.protocols import Model, Tool, Memory


def create_agent(
    model: Model | str,
    tools: list[Tool] | None = None,
    memory: Memory | None = None,
    system_prompt: str | None = None,
    **kwargs
) -> Agent:
    """创建 Agent 的工厂函数
    
    Usage:
        # 使用字符串快速创建
        agent = create_agent("openai:gpt-4", tools=[search])
        
        # 使用模型实例
        agent = create_agent(OpenAIModel(...), tools=[search])
    """
    # 解析模型字符串（如 "openai:gpt-4"）
    if isinstance(model, str):
        model = _resolve_model(model)
    
    config = AgentConfig(
        model=model,
        tools=tools or [],
        memory=memory,
        system_prompt=system_prompt,
        **kwargs
    )
    
    return Agent(config)


def _resolve_model(model_str: str) -> Model:
    """解析模型字符串"""
    provider, model_name = model_str.split(":", 1)
    
    if provider == "openai":
        from zent.integrations.openai import OpenAIModel
        return OpenAIModel(model=model_name)
    elif provider == "anthropic":
        from zent.integrations.anthropic import AnthropicModel
        return AnthropicModel(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

---

## 5. 集成层设计

### 5.1 OpenAI 适配（zent/integrations/openai.py）

```python
"""OpenAI 模型适配"""
import os
from typing import AsyncIterator

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from zent.core.protocols import Model, Message, ModelResponse, ToolCall


class OpenAIModel:
    """OpenAI 模型实现"""
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str | None = None,
        **kwargs
    ):
        if not HAS_OPENAI:
            raise ImportError("pip install openai")
        
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.default_params = kwargs
    
    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """转换为 OpenAI 格式"""
        return [
            {"role": m.role.value, "content": m.content, **m.metadata}
            for m in messages
        ]
    
    def _convert_tools(self, tools: list) -> list[dict]:
        """转换为 OpenAI 工具格式"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters
                }
            }
            for t in tools
        ]
    
    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        **kwargs
    ) -> ModelResponse:
        """生成响应"""
        params = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            **({"tools": self._convert_tools(tools)} if tools else {}),
            **self.default_params,
            **kwargs
        }
        
        response = await self.client.chat.completions.create(**params)
        choice = response.choices[0]
        message = choice.message
        
        # 提取工具调用
        tool_calls = []
        if message.tool_calls:
            import json
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))
        
        return ModelResponse(
            content=message.content,
            tool_calls=tool_calls,
            model=response.model,
            finish_reason=choice.finish_reason
        )
    
    async def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        """流式生成"""
        params = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "stream": True,
            **self.default_params,
            **kwargs
        }
        
        async for chunk in await self.client.chat.completions.create(**params):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

### 5.2 MCP 工具适配（zent/integrations/mcp.py）

```python
"""MCP 工具适配"""
import inspect
import json
from dataclasses import dataclass
from typing import Callable, get_type_hints

from zent.core.protocols import Tool


@dataclass
class FunctionTool:
    """函数工具 - MCP 格式
    
    自动从函数签名生成 JSON Schema
    """
    func: Callable
    name: str
    description: str
    _parameters: dict | None = None
    
    @property
    def parameters(self) -> dict:
        """生成 JSON Schema"""
        if self._parameters is not None:
            return self._parameters
        
        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)
        
        properties = {}
        required = []
        
        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, str)
            
            # Python 类型 -> JSON Schema 类型
            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object"
            }
            json_type = type_map.get(param_type, "string")
            
            properties[param_name] = {"type": json_type}
            
            if param.default == inspect.Parameter.empty:
                required.append(param_name)
        
        self._parameters = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        
        return self._parameters
    
    async def run(self, **kwargs) -> str:
        """执行函数"""
        if inspect.iscoroutinefunction(self.func):
            result = await self.func(**kwargs)
        else:
            result = self.func(**kwargs)
        
        return result if isinstance(result, str) else json.dumps(result)


class MCPToolAdapter:
    """MCP 工具适配器 - 将外部 MCP 工具转换为 Zent Tool"""
    
    def __init__(self, mcp_client):
        self.client = mcp_client
    
    async def list_tools(self) -> list[Tool]:
        """列出 MCP 服务器上的工具"""
        # 调用 MCP 协议获取工具列表
        # 转换为 FunctionTool 实例
        pass
```

### 5.3 内存实现（zent/integrations/memory.py）

```python
"""内存实现"""
from dataclasses import dataclass, field
from zent.core.protocols import Memory, Message


@dataclass
class InMemoryMemory:
    """内存记忆实现"""
    _messages: list[Message] = field(default_factory=list)
    
    async def add(self, message: Message) -> None:
        self._messages.append(message)
    
    async def get(self, limit: int = 10) -> list[Message]:
        return self._messages[-limit:] if self._messages else []
    
    async def clear(self) -> None:
        self._messages.clear()
```

---

## 6. 使用示例

### 6.1 基础用法

```python
import asyncio
from zent import create_agent, tool
from zent.integrations.openai import OpenAIModel
from zent.integrations.memory import InMemoryMemory


@tool
def search(query: str) -> str:
    """搜索知识库"""
    return f"Results for: {query}"


async def main():
    # 方式1: 使用工厂函数（推荐）
    agent = create_agent(
        "openai:gpt-4",
        tools=[search],
        memory=InMemoryMemory(),
        system_prompt="你是一个 helpful 助手。"
    )
    
    result = await agent.run("搜索 Python 教程")
    print(result.output)
    
    # 方式2: 显式配置
    model = OpenAIModel(model="gpt-4")
    agent = create_agent(model, tools=[search])
    
    result = await agent.run("搜索 Python 教程")
    print(result.output)


asyncio.run(main())
```

### 6.2 流式输出

```python
async def stream_demo():
    agent = create_agent("openai:gpt-4")
    
    async for chunk in agent.stream("讲个故事"):
        print(chunk, end="")
```

### 6.3 使用 MCP 工具

```python
from zent.integrations.mcp import MCPToolAdapter

async def mcp_demo():
    # 连接到 MCP 服务器
    adapter = MCPToolAdapter(mcp_client)
    tools = await adapter.list_tools()
    
    agent = create_agent("openai:gpt-4", tools=tools)
    result = await agent.run("使用工具计算 123 * 456")
```

---

## 7. 路线图

### Phase 1: 核心（已完成设计）
- [x] 简化的 3 层架构
- [x] 4 个核心 Protocol（Model, Tool, Memory, Agent）
- [x] Agent 内置 ReAct 循环
- [x] @tool 装饰器
- [x] create_agent() 工厂函数

### Phase 2: 集成（Week 1-2）
- [ ] OpenAI 适配
- [ ] Anthropic 适配
- [ ] MCP 工具适配
- [ ] InMemoryMemory 实现
- [ ] 测试套件

### Phase 3: 扩展（Week 3-4）
- [ ] 流式输出完善
- [ ] 错误处理 & 重试
- [ ] 更多工具类型
- [ ] 文档 & 示例

### Phase 4: 生态（Week 5+）
- [ ] Chroma 记忆扩展
- [ ] RAG 支持
- [ ] 可观测性集成

---

## 8. v2 → v3 变更总结

| 变更项 | v2 (旧) | v3 (新) | 理由 |
|:---|:---|:---|:---|
| **架构层数** | 5 层 | 3 层 | 减少概念负担 |
| **核心抽象** | LLM, Tool, Memory, Planner, Agent | **Model, Tool, Memory, Agent** | 删除冗余 Planner |
| **Workflow 层** | 独立层 | **合并到 Agent** | Agent 就是协调器 |
| **Runtime 层** | EventBus, Checkpoint | **删除** | 作为可选中间件 |
| **项目结构** | 8+ 目录 | **3 个目录** | 极简主义 |
| **扩展机制** | ExtensionRegistry | **标准 Python 导入** | 减少魔法 |
| **创建方式** | Agent(AgentConfig(...)) | **create_agent()** | LangChain 风格 |
| **流式 API** | AsyncIterator[StreamChunk] | **AsyncIterator[str]** | 简化 |

---

## 9. 关键决策记录

| 决策 | 选择 | 理由 |
|:---|:---|:---|
| 架构层数 | 3 层 | LangChain 1.0 经验：删除 80% 命名空间后更清晰 |
| 删除 Planner | Agent 直接实现 ReAct | 简单场景不需要独立规划器 |
| 删除 Runtime | 回调函数替代 EventBus | 减少概念，保持核心精简 |
| 重命名 LLM → Model | Model | 更通用，不仅限于 LLM |
| create_agent 工厂 | 函数式 API | LangChain 风格，更直观 |

---

*版本: 3.0*  
*更新: 2025-03-03*
