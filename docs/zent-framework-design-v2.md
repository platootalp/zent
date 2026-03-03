# Zent Agent 框架设计文档 V2

## 1. 设计原则

### 1.1 核心原则
- **单一职责**：每个模块只做一件事，做好一件事
- **显式优于隐式**：配置和依赖显式声明，不隐藏魔法
- **渐进增强**：从简单开始，按需叠加复杂度
- **协议优先**：Python Protocol 而非抽象基类，零继承成本

### 1.2 架构决策
- **核心层零依赖**：仅 Python 标准库 + typing
- **异步优先**：所有 IO 操作都是 async
- **类型安全**：完整类型注解，支持静态检查
- **MCP 原生**：工具协议与 MCP 对齐

---

## 2. 系统架构

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│                     应用层 (Application)                      │
│  AgentApp, @agent, @tool, 配置驱动启动                         │
├────────────────────────────────────────────────────────────────┤
│                     工作流层 (Workflow)                       │
│  Agent (协调器), Session (状态), Step (步骤)                   │
├─────────────────────────────────────────────────────────────┤
│                     能力层 (Capabilities)                     │
│  LLM (模型), Tool (工具), Memory (记忆), Planner (规划)        │
├─────────────────────────────────────────────────────────────┤
│                     运行时层 (Runtime)                        │
│  EventBus (事件), Context (上下文), Checkpoint (检查点)        │
├─────────────────────────────────────────────────────────────┤
│                     适配层 (Adapters)                         │
│  OpenAI, Anthropic, MCP, Chroma, Redis...                   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心概念

```python
# 核心关系图
Agent
├── llm: LLM              # 语言模型
├── tools: List[Tool]     # 工具集
├── memory: Memory        # 记忆（可选）
├── planner: Planner      # 规划器（默认 ReAct）
└── session: Session      # 运行时状态

Session
├── messages: List[Message]    # 消息历史
├── steps: List[Step]          # 执行步骤
├── context: Dict              # 上下文数据
└── checkpoint: Checkpoint     # 检查点（可选）

Step
├── type: StepType      # THINK / TOOL_CALL / ANSWER
├── content: str        # 内容
├── tool_call: ToolCall # 工具调用（可选）
└── timestamp: datetime
```

---

## 3. 项目结构

```
zent/
├── src/zent/
│   ├── __init__.py              # 公开 API
│   ├── __version__.py           # 版本信息
│   │
│   ├── core/                    # 核心层（零外部依赖）
│   │   ├── __init__.py
│   │   ├── protocols.py         # 所有 Protocol 定义
│   │   ├── message.py           # Message, MessageRole
│   │   ├── session.py           # Session, Step, StepType
│   │   └── result.py            # AgentResult, ToolResult
│   │
│   ├── workflow/                # 工作流层
│   │   ├── __init__.py
│   │   ├── agent.py             # Agent 协调器
│   │   ├── loop.py              # ReAct 循环实现
│   │   └── context.py           # ContextVar 管理
│   │
│   ├── capabilities/            # 能力层
│   │   ├── __init__.py
│   │   ├── llm.py               # LLM Protocol
│   │   ├── tool.py              # Tool Protocol + 基类
│   │   ├── memory.py            # Memory Protocol
│   │   └── planner.py           # Planner Protocol
│   │
│   ├── runtime/                 # 运行时层
│   │   ├── __init__.py
│   │   ├── events.py            # EventBus, Event
│   │   ├── checkpoint.py        # Checkpoint Protocol
│   │   └── context.py           # 运行时上下文管理
│   │
│   ├── adapters/                # 适配层（内置适配器）
│   │   ├── __init__.py
│   │   ├── llm/
│   │   │   ├── openai.py        # OpenAI 适配
│   │   │   ├── anthropic.py     # Claude 适配
│   │   │   └── ollama.py        # Ollama 适配
│   │   ├── tool/
│   │   │   ├── function.py      # 函数工具
│   │   │   ├── mcp.py           # MCP 工具适配
│   │   │   └── builtin.py       # 内置工具
│   │   ├── memory/
│   │   │   ├── working.py       # 工作记忆
│   │   │   └── short_term.py    # 短期记忆
│   │   └── planner/
│   │       ├── react.py         # ReAct 规划器
│   │       └── plan_solve.py    # Plan-and-Solve
│   │
│   ├── app/                     # 应用层（语法糖）
│   │   ├── __init__.py
│   │   ├── factory.py           # Agent 工厂
│   │   ├── decorators.py        # @agent, @tool
│   │   └── config.py            # 配置解析
│   │
│   ├── ext/                     # 扩展接口（钩子）
│   │   ├── __init__.py
│   │   ├── registry.py          # 扩展注册表
│   │   └── hooks.py             # 生命周期钩子
│   │
│   └── utils/                   # 工具函数
│       ├── __init__.py
│       ├── schema.py            # JSON Schema 生成
│       ├── async_utils.py       # 异步工具
│       └── logging.py           # 日志配置
│
├── extensions/                  # 独立扩展包
│   ├── zent-chroma/             # 向量记忆
│   ├── zent-rag/                # RAG 检索
│   ├── zent-observability/      # 可观测性
│   └── ...
│
├── tests/
│   ├── unit/                    # 单元测试
│   ├── integration/             # 集成测试
│   └── e2e/                     # 端到端测试
│
├── examples/                    # 示例代码
│   ├── 01-basic-chat.py
│   ├── 02-tools.py
│   ├── 03-memory.py
│   ├── 04-mcp-tools.py
│   └── 05-multi-agent.py
│
├── docs/
│   ├── architecture.md          # 架构文档
│   ├── api-reference.md         # API 参考
│   ├── extensions.md            # 扩展开发指南
│   └── tutorials/               # 教程
│
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 4. 核心 API 设计

### 4.1 Protocol 定义（zent/core/protocols.py）

```python
from typing import Protocol, runtime_checkable, AsyncIterator
from datetime import datetime
from dataclasses import dataclass
from enum import Enum, auto

# ==================== 基础类型 ====================

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str
    metadata: dict = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', datetime.now())

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class ToolResult:
    call_id: str
    output: str
    is_error: bool = False
    duration_ms: float = 0.0

@dataclass
class TokenUsage:
    prompt: int
    completion: int
    total: int

@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    usage: TokenUsage
    model: str
    finish_reason: str | None = None

# ==================== 核心 Protocols ====================

@runtime_checkable
class LLM(Protocol):
    """语言模型协议"""
    
    async def complete(
        self,
        messages: list[Message],
        tools: list["Tool"] | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs
    ) -> LLMResponse: ...
    
    async def stream(
        self,
        messages: list[Message],
        **kwargs
    ) -> AsyncIterator[str]: ...

@runtime_checkable
class Tool(Protocol):
    """工具协议"""
    
    @property
    def name(self) -> str: ...
    
    @property
    def description(self) -> str: ...
    
    @property
    def parameters(self) -> dict: ...
    
    async def run(self, **kwargs) -> str: ...

@runtime_checkable
class Memory(Protocol):
    """记忆协议"""
    
    async def add(self, message: Message) -> None: ...
    
    async def get(
        self, 
        query: str | None = None, 
        limit: int = 10
    ) -> list[Message]: ...
    
    async def clear(self) -> None: ...

@runtime_checkable
class Planner(Protocol):
    """规划器协议"""
    
    async def plan(
        self,
        query: str,
        context: "Session"
    ) -> "Plan": ...
    
    async def next_step(
        self,
        context: "Session"
    ) -> "Step": ...

@runtime_checkable
class Checkpoint(Protocol):
    """检查点协议"""
    
    async def save(self, session: "Session") -> str: ...
    
    async def load(self, checkpoint_id: str) -> "Session": ...
```

### 4.2 Agent 协调器（zent/workflow/agent.py）

```python
from typing import List, Optional, Callable, AsyncIterator
from dataclasses import dataclass, field
import asyncio

@dataclass
class AgentConfig:
    """Agent 配置"""
    llm: LLM
    tools: List[Tool] = field(default_factory=list)
    memory: Optional[Memory] = None
    planner: Optional[Planner] = None
    max_iterations: int = 10
    system_prompt: Optional[str] = None
    
    # 回调
    on_step: Optional[Callable[["Step"], None]] = None
    on_tool: Optional[Callable[[ToolCall, ToolResult], None]] = None

class Agent:
    """
    Agent 协调器：管理工作流执行
    
    Usage:
        agent = Agent(config)
        result = await agent.run("查询天气")
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools_map = {t.name: t for t in config.tools}
        self._event_bus = EventBus()
    
    async def run(self, query: str, **context) -> "AgentResult":
        """执行单次任务"""
        session = Session(
            query=query,
            context=context,
            system_prompt=self.config.system_prompt
        )
        
        try:
            for i in range(self.config.max_iterations):
                step = await self._step(session, i)
                
                if step.type == StepType.ANSWER:
                    return AgentResult(
                        output=step.content,
                        session=session,
                        success=True
                    )
                
                session.add_step(step)
            
            raise MaxIterationsError(f"Max iterations: {self.config.max_iterations}")
            
        except Exception as e:
            return AgentResult(
                output=str(e),
                session=session,
                success=False,
                error=e
            )
    
    async def stream(self, query: str, **context) -> AsyncIterator["StreamChunk"]:
        """流式执行"""
        session = Session(query=query, context=context)
        
        for i in range(self.config.max_iterations):
            step = await self._step(session, i)
            
            if step.type == StepType.THINK:
                yield StreamChunk(type="thought", content=step.content)
            elif step.type == StepType.TOOL_CALL:
                yield StreamChunk(type="tool_start", tool=step.tool_call.name)
                result = await self._execute_tool(step.tool_call)
                yield StreamChunk(type="tool_end", result=result.output)
            elif step.type == StepType.ANSWER:
                yield StreamChunk(type="answer", content=step.content)
                return
    
    async def _step(self, session: "Session", iteration: int) -> "Step":
        """执行单步"""
        # 1. 规划下一步
        planner = self.config.planner or ReActPlanner()
        step = await planner.next_step(session)
        
        # 2. 回调
        if self.config.on_step:
            self.config.on_step(step)
        
        # 3. 如果是工具调用，执行工具
        if step.type == StepType.TOOL_CALL and step.tool_call:
            result = await self._execute_tool(step.tool_call)
            step.result = result
            
            if self.config.on_tool:
                self.config.on_tool(step.tool_call, result)
        
        return step
    
    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """执行工具"""
        tool = self.tools_map.get(call.name)
        if not tool:
            return ToolResult(
                call_id=call.id,
                output=f"Tool '{call.name}' not found",
                is_error=True
            )
        
        start = asyncio.get_event_loop().time()
        try:
            output = await tool.run(**call.arguments)
            duration = (asyncio.get_event_loop().time() - start) * 1000
            return ToolResult(
                call_id=call.id,
                output=output,
                duration_ms=duration
            )
        except Exception as e:
            return ToolResult(
                call_id=call.id,
                output=f"Error: {e}",
                is_error=True
            )
```

### 4.3 应用层语法糖（zent/app/decorators.py）

```python
from functools import wraps
from typing import Callable, TypeVar, ParamSpec

P = ParamSpec("P")
T = TypeVar("T")

def tool(name: str | None = None, description: str | None = None):
    """工具装饰器"""
    def decorator(func: Callable[P, T]) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        
        return FunctionTool(
            func=func,
            name=tool_name,
            description=tool_desc
        )
    
    # 支持 @tool 无括号
    if callable(name):
        func = name
        name = None
        return decorator(func)
    
    return decorator

def agent(
    llm: LLM | str | None = None,
    tools: list[Tool] | None = None,
    memory: Memory | None = None,
    **kwargs
):
    """Agent 类装饰器"""
    def decorator(cls: type) -> type:
        @wraps(cls, updated=())
        class AgentWrapper(cls):
            def __init__(self, *args, **init_kwargs):
                super().__init__(*args, **init_kwargs)
                
                # 解析 LLM
                llm_instance = _resolve_llm(llm)
                
                # 收集工具（从类方法中）
                agent_tools = list(tools or [])
                for attr_name in dir(self):
                    attr = getattr(self, attr_name)
                    if isinstance(attr, Tool):
                        agent_tools.append(attr)
                
                # 创建 Agent
                self._agent = Agent(AgentConfig(
                    llm=llm_instance,
                    tools=agent_tools,
                    memory=memory,
                    **kwargs
                ))
            
            async def run(self, query: str, **context):
                return await self._agent.run(query, **context)
        
        return AgentWrapper
    
    return decorator
```

---

## 5. 扩展机制

### 5.1 扩展注册表（zent/ext/registry.py）

```python
from typing import Type, Dict, TypeVar
import importlib

T = TypeVar("T")

class ExtensionRegistry:
    """扩展注册表：自动发现和注册扩展"""
    
    _registries: Dict[str, Dict[str, Type]] = {
        "llm": {},
        "memory": {},
        "tool": {},
        "planner": {},
    }
    
    @classmethod
    def register(cls, category: str, name: str, impl: Type[T]) -> None:
        """注册实现"""
        if category not in cls._registries:
            cls._registries[category] = {}
        cls._registries[category][name] = impl
    
    @classmethod
    def get(cls, category: str, name: str) -> Type | None:
        """获取实现"""
        return cls._registries.get(category, {}).get(name)
    
    @classmethod
    def auto_discover(cls) -> None:
        """自动发现已安装的扩展"""
        extensions = [
            "zent_chroma",
            "zent_rag",
            "zent_observability",
        ]
        
        for ext_name in extensions:
            try:
                ext = importlib.import_module(ext_name)
                if hasattr(ext, "register_extensions"):
                    ext.register_extensions(cls)
            except ImportError:
                pass

# 便捷注册函数
def register_llm(name: str):
    def decorator(cls: Type[LLM]) -> Type[LLM]:
        ExtensionRegistry.register("llm", name, cls)
        return cls
    return decorator

def register_memory(name: str):
    def decorator(cls: Type[Memory]) -> Type[Memory]:
        ExtensionRegistry.register("memory", name, cls)
        return cls
    return decorator
```

### 5.2 扩展包结构

```python
# zent-chroma/zent_chroma/__init__.py
from zent.ext import register_memory
from .memory import ChromaMemory

def register_extensions(registry):
    """注册扩展"""
    registry.register("memory", "chroma", ChromaMemory)

__all__ = ["ChromaMemory"]

# zent-chroma/zent_chroma/memory.py
from zent import Memory, Message
import chromadb

class ChromaMemory:
    """ChromaDB 向量记忆实现"""
    
    def __init__(
        self,
        collection: str = "default",
        embedding_model: str = "text-embedding-3-small",
        persist_dir: str | None = None
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection)
        self.embedder = OpenAIEmbedding(model=embedding_model)
    
    async def add(self, message: Message) -> None:
        embedding = await self.embedder.embed(message.content)
        self.collection.add(
            embeddings=[embedding],
            documents=[message.content],
            ids=[f"msg_{hash(message.content)}"]
        )
    
    async def get(self, query: str | None = None, limit: int = 10) -> list[Message]:
        if not query:
            return []
        
        embedding = await self.embedder.embed(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit
        )
        
        return [
            Message(role=MessageRole.USER, content=doc)
            for doc in results["documents"][0]
        ]
    
    async def clear(self) -> None:
        self.collection.delete()
```

---

## 6. 使用示例

### 6.1 基础用法

```python
from zent import Agent, AgentConfig, OpenAILLM, tool, InMemoryMemory

@tool
def search(query: str) -> str:
    """搜索知识库"""
    return f"Results for: {query}"

async def main():
    agent = Agent(AgentConfig(
        llm=OpenAILLM(model="gpt-4", api_key="sk-..."),
        tools=[search],
        memory=InMemoryMemory(),
        system_prompt="你是一个 helpful 助手。"
    ))
    
    result = await agent.run("搜索 Python 教程")
    print(result.output)

asyncio.run(main())
```

### 6.2 装饰器语法

```python
from zent import agent, tool, OpenAILLM

@tool
def calculator(expr: str) -> str:
    """计算表达式"""
    return str(eval(expr))

@agent(
    llm=OpenAILLM(model="gpt-4"),
    tools=[calculator],
    system_prompt="你是数学助手。"
)
class MathAssistant:
    pass

assistant = MathAssistant()
result = await assistant.run("计算 123 * 456")
```

### 6.3 配置驱动

```python
from zent import Agent
from zent.app import load_config

# 从 YAML/JSON 加载
config = load_config("agent.yaml")
agent = Agent.from_config(config)

# agent.yaml
# llm:
#   provider: openai
#   model: gpt-4
# memory:
#   provider: chroma
#   collection: my_agent
# tools:
#   - search
#   - calculator
```

---

## 7. 路线图

### Phase 1: 核心（Week 1-2）
- [ ] Protocol 定义（LLM, Tool, Memory, Planner）
- [ ] Agent 协调器（ReAct 循环）
- [ ] Session & Step 状态管理
- [ ] OpenAI / Anthropic LLM 适配
- [ ] FunctionTool + @tool 装饰器
- [ ] InMemoryMemory 实现

### Phase 2: 标准组件（Week 3-4）
- [ ] MCP 工具适配
- [ ] Ollama 本地模型支持
- [ ] ShortTermMemory（摘要）
- [ ] Plan-and-Solve Planner
- [ ] 内置工具（文件、HTTP、数据库）
- [ ] 测试套件 + 文档

### Phase 3: 扩展生态（Week 5+）
- [ ] zent-chroma（向量记忆）
- [ ] zent-rag（RAG 检索）
- [ ] zent-observability（追踪）
- [ ] zent-evaluation（评估）

### Phase 4: 生产就绪（Week 7+）
- [ ] 性能优化
- [ ] 错误处理 & 重试
- [ ] 配置系统完善
- [ ] 完整文档 & 示例

---

## 8. 关键决策记录

| 决策 | 选择 | 理由 |
|:---|:---|:---|
| 架构分层 | 5 层 | 职责清晰，便于测试和扩展 |
| Protocol vs ABC | Protocol | 零继承成本，鸭子类型友好 |
| 异步策略 | async/await 优先 | IO 密集型，现代 Python 标准 |
| MCP 支持 | 内置轻量 + 扩展完整 | 平衡核心体积和功能完整 |
| 配置方式 | 代码 + YAML | 开发期代码，生产期配置 |
| 扩展机制 | 注册表 + 自动发现 | 插件化，零配置使用 |

---

*版本: 2.0*  
*更新: 2024-XX-XX*
