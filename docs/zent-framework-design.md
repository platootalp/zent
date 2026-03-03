# Zent 轻量级 Agent 开发框架设计文档

## 概述

**Zent**（Zephyr + Agent）是一个轻量级的 Python Agent 开发框架，核心理念是 **"Everything is a Tool"**。

### 设计哲学

1. **渐进式架构** - 从简单开始，按需扩展
2. **MCP 优先** - 以 Model Context Protocol 作为工具生态标准
3. **Pythonic 设计** - Protocol 而非继承，类型安全，异步优先
4. **消失的基础设施** - 开发者专注于 Agent 逻辑，而非框架概念

### 核心特点

- **极简核心**：仅 5 个核心抽象（Agent、Tool、LLM、Memory、Planner）
- **分层模块化**：核心层 + 组件层 + 扩展层
- **混合 API**：底层完整控制 + 高层装饰器语法糖
- **协议驱动**：Python Protocol 实现零继承成本扩展

---

## 架构设计

### 项目结构

```
zent/
├── src/zent/
│   ├── __init__.py          # 公开 API
│   ├── core/                # 核心抽象（最小化）
│   │   ├── agent.py         # Agent 门面类
│   │   ├── llm.py           # LLM Protocol
│   │   ├── tool.py          # Tool Protocol + 装饰器
│   │   ├── memory.py        # Memory Protocol
│   │   └── planner.py       # Planner Protocol
│   ├── components/          # 可插拔实现
│   │   ├── llms/            # OpenAI, Claude, Ollama, OpenRouter
│   │   ├── tools/           # 内置工具 + MCP 适配
│   │   ├── memory/          # 工作/短期/长期记忆实现
│   │   └── planners/        # ReAct, Plan-and-Solve, Self-Correction
│   └── utils/               # 工具函数
├── extensions/              # 独立扩展包（可选安装）
│   ├── zent-chroma/         # ChromaDB 向量记忆
│   ├── zent-security/       # 沙箱执行、权限控制
│   ├── zent-evaluation/     # LLM-as-a-Judge 评估
│   └── zent-mcp-advanced/   # 完整 MCP SDK 适配
├── tests/
└── examples/
```

### 分层架构

```
┌─────────────────────────────────────────────────────────┐
│  扩展层 (Extensions)                                      │
│  zent-chroma, zent-security, zent-evaluation...         │
├─────────────────────────────────────────────────────────┤
│  组件层 (Components)                                      │
│  OpenAILLM, MCPTool, ChromaMemory, ReActPlanner...      │
├─────────────────────────────────────────────────────────┤
│  核心层 (Core)                                            │
│  Agent, Tool, LLM, Memory, Planner (Protocols)          │
└─────────────────────────────────────────────────────────┘
```

**核心层** 零外部依赖，仅使用 Python 标准库 + typing  
**组件层** 实现核心协议的适配器和具体实现  
**扩展层** 可选安装的独立包，提供高级功能

---

## 核心 API 设计

### 1. Agent - 门面类

Agent 是框架的主要入口，协调 LLM、Tools、Memory、Planner 完成 ReAct 循环。

```python
from zent import Agent, OpenAILLM, tool, InMemoryMemory

@tool
def search(query: str) -> str:
    """搜索知识库"""
    return f"Results for: {query}"

agent = Agent(
    llm=OpenAILLM(model="gpt-4", api_key="sk-..."),
    tools=[search],
    memory=InMemoryMemory(),
    system_prompt="你是一个 helpful 的助手。"
)

result = await agent.run("搜索 Python 教程")
print(result.output)
```

**核心方法：**
- `run(query: str, **context) -> AgentResult` - 执行完整 ReAct 循环
- `stream(query: str, **context) -> AsyncIterator[str]` - 流式输出

**配置选项：**
- `llm: LLM` - LLM 提供商（必需）
- `tools: list[Tool]` - 可用工具列表
- `memory: Memory | None` - 记忆组件
- `planner: Planner | None` - 规划策略（默认 ReAct）
- `max_iterations: int` - 最大迭代次数（默认 10）
- `system_prompt: str | None` - 系统提示词

### 2. Tool - 工具协议

Tool 是框架的核心抽象，任何实现 Tool 协议的对象都可以被 Agent 使用。

**创建方式 1：装饰器（推荐）**

```python
from zent import tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

@tool(name="web_search", description="搜索网络")
async def search(query: str, top_k: int = 5) -> str:
    """异步搜索"""
    return await web_search_api(query, top_k)
```

**创建方式 2：类继承（复杂工具）**

```python
from zent import BaseTool

class DatabaseTool(BaseTool):
    name = "database"
    description = "执行 SQL 查询"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
    
    def __init__(self, connection_string: str):
        self.db = create_engine(connection_string)
    
    async def run(self, query: str) -> str:
        result = self.db.execute(query)
        return str(result.fetchall())
```

**创建方式 3：直接实现协议**

```python
from zent import Tool

class CustomTool:
    @property
    def name(self) -> str:
        return "custom"
    
    @property
    def description(self) -> str:
        return "Custom tool"
    
    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}
    
    async def run(self, **kwargs) -> str:
        return "result"
```

### 3. LLM - 语言模型协议

LLM 协议抽象不同模型提供商的接口。

```python
from zent import OpenAILLM, ClaudeLLM, OllamaLLM

# OpenAI
openai_llm = OpenAILLM(model="gpt-4", api_key="sk-...")

# Anthropic Claude
claude_llm = ClaudeLLM(model="claude-3-opus", api_key="sk-ant-...")

# 本地 Ollama
ollama_llm = OllamaLLM(model="llama2", host="http://localhost:11434")

# OpenRouter（聚合多个模型）
openrouter = OpenRouterLLM(model="anthropic/claude-3-opus")
```

**核心方法：**
- `complete(messages, tools=None, **kwargs) -> LLMResponse`
- `stream(messages, **kwargs) -> AsyncIterator[str]`
- `with_config(**kwargs) -> LLM` - 创建配置副本

### 4. Memory - 记忆协议

三层记忆模型：工作记忆 → 短期记忆 → 长期记忆

```python
from zent import InMemoryMemory, ShortTermMemory

# 工作记忆（当前会话，内存存储）
working_memory = InMemoryMemory(max_messages=20)

# 短期记忆（带摘要功能）
short_term = ShortTermMemory(
    storage="./memory.json",
    summary_model="gpt-3.5-turbo"
)

# 长期记忆（需要安装 zent-chroma）
from zent_chroma import ChromaMemory
long_term = ChromaMemory(
    collection="agent_memory",
    embedding_model="text-embedding-3-small"
)

agent = Agent(
    llm=llm,
    memory=working_memory,  # 或 short_term, long_term
    tools=tools
)
```

### 5. Planner - 规划协议（可选）

规划策略决定 Agent 如何拆解任务。

```python
from zent.components.planners import (
    ReActPlanner,
    PlanAndSolvePlanner,
    SelfCorrectionPlanner
)

# 默认 ReAct
agent = Agent(llm=llm, planner=ReActPlanner())

# Plan-and-Solve：先规划后执行
agent = Agent(llm=llm, planner=PlanAndSolvePlanner())

# Self-Correction：自动纠错
agent = Agent(
    llm=llm,
    planner=SelfCorrectionPlanner(
        base_planner=ReActPlanner(),
        max_corrections=3
    )
)
```

---

## 使用示例

### 基础示例

```python
import asyncio
from zent import Agent, OpenAILLM, tool, InMemoryMemory

@tool
def greet(name: str) -> str:
    """向某人打招呼"""
    return f"Hello, {name}!"

@tool
def add(a: int, b: int) -> int:
    """相加两个数字"""
    return a + b

async def main():
    agent = Agent(
        llm=OpenAILLM(model="gpt-4", api_key="your-key"),
        tools=[greet, add],
        memory=InMemoryMemory(),
        system_prompt="你是一个 helpful 的数学助手。"
    )
    
    result = await agent.run("计算 123 + 456，然后向结果打招呼")
    print(result.output)
    print(f"执行了 {result.iterations} 步")

if __name__ == "__main__":
    asyncio.run(main())
```

### MCP 工具集成

```python
from zent import Agent, OpenAILLM, load_mcp_tools

async def main():
    # 从配置文件加载
    tools = await load_mcp_tools("mcp-servers.json")
    
    # 或程序化配置
    tools = await load_mcp_tools(
        stdio=["npx -y @modelcontextprotocol/server-filesystem ./docs"],
        sse=["http://localhost:3000/sse"]
    )
    
    agent = Agent(
        llm=OpenAILLM(model="gpt-4"),
        tools=tools,
        system_prompt="你可以访问文件系统和数据库。"
    )
    
    result = await agent.run("读取 README.md 并总结内容")
    print(result.output)
```

### 流式输出

```python
async def stream_example():
    agent = Agent(llm=OpenAILLM(model="gpt-4"), tools=[])
    
    print("AI: ", end="")
    async for chunk in agent.stream("讲一个关于 AI 的故事"):
        print(chunk, end="", flush=True)
    print()
```

### 回调与监控

```python
agent = Agent(
    llm=OpenAILLM(model="gpt-4"),
    tools=tools,
    on_step=lambda step: print(f"[Step {step.iteration}] Action: {step.action}"),
    on_tool_call=lambda call, result: print(f"[Tool] {call.name}: {result.output[:50]}...")
)
```

---

## 扩展能力架构

虽然扩展能力不是 MVP 的核心内容，但框架需要在架构层面**预留扩展点**，确保未来可以无缝集成这些能力。

### 扩展能力总览

| 能力维度 | 扩展包 | 核心协议/接口 | 说明 |
|:---|:---|:---|:---|
| **🧠 记忆增强** | `zent-chroma`<br>`zent-postgres` | `VectorMemory`<br>`EntityMemory` | 向量数据库、实体提取、对话摘要 |
| **🔍 RAG 集成** | `zent-rag` | `Retriever`<br>`DocumentStore` | 混合检索、重排序、检索决策 |
| **🤝 多智能体** | `zent-multiagent` | `AgentTeam`<br>`MessageBus` | 角色分工、协作机制、通信协议 |
| **👥 人机协同** | `zent-human` | `HumanInTheLoop`<br>`ApprovalWorkflow` | 断点审批、反馈修正 |
| **🛡️ 安全治理** | `zent-security` | `Sandbox`<br>`PolicyEngine` | 沙箱执行、权限控制、PII 脱敏 |
| **📊 可观测性** | `zent-observability` | `Tracer`<br>`MetricsCollector` | 链路追踪、Token/延迟监控 |
| **🧪 评估体系** | `zent-evaluation` | `Evaluator`<br>`Benchmark` | LLM-as-a-Judge、回归测试 |
| **⚙️ 工程扩展** | `zent-engineering` | `Checkpoint`<br>`Queue` | 状态持久化、并发调度 |

### 扩展设计原则

1. **按需启用** - 扩展包独立安装，不增加核心包体积
2. **协议兼容** - 扩展通过实现核心协议与框架集成
3. **配置驱动** - 通过 Agent 配置启用扩展能力
4. **零侵入** - 不使用扩展时，核心功能完全不受影响

---

## 扩展协议设计

### 1. 记忆增强扩展

```python
# 扩展协议：zent/core/memory_extensions.py
from typing import Protocol, runtime_checkable
from core.memory import Memory

@runtime_checkable
class VectorMemory(Memory):
    """向量记忆：支持语义检索的长期记忆"""
    
    async def add(self, message: Message, embedding: list[float] | None = None) -> None:
        """添加消息，可选预计算 embedding"""
        ...
    
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        filter: dict | None = None
    ) -> list["MemoryEntry"]:
        """语义检索"""
        ...
    
    async def delete(self, entry_id: str) -> bool:
        """删除特定记忆"""
        ...

@runtime_checkable
class EntityMemory(Memory):
    """实体记忆：提取和跟踪关键实体"""
    
    async def extract_entities(self, text: str) -> list["Entity"]:
        """从文本中提取实体"""
        ...
    
    async def get_entity(self, entity_id: str) -> "Entity" | None:
        """获取实体信息"""
        ...
    
    async def update_entity(self, entity: "Entity") -> None:
        """更新实体状态"""
        ...

# 扩展包实现示例：zent-chroma
class ChromaVectorMemory(VectorMemory):
    """ChromaDB 实现的向量记忆"""
    
    def __init__(
        self,
        collection: str,
        embedding_model: str = "text-embedding-3-small",
        persist_directory: str | None = None
    ):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(collection)
        self.embedder = OpenAIEmbedding(model=embedding_model)
    
    async def add(self, message: Message, embedding: list[float] | None = None):
        if embedding is None:
            embedding = await self.embedder.embed(message.content)
        
        self.collection.add(
            embeddings=[embedding],
            documents=[message.content],
            metadatas=[message.metadata],
            ids=[f"msg_{message.timestamp.timestamp()}"]
        )
    
    async def search(self, query: str, top_k: int = 5, filter: dict | None = None):
        embedding = await self.embedder.embed(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filter
        )
        return [
            MemoryEntry(
                message=Message.user(doc),
                metadata=meta,
                score=score
            )
            for doc, meta, score in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
```

### 2. RAG 集成扩展

```python
# 扩展协议：zent/core/rag.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class Retriever(Protocol):
    """检索器：支持多种检索策略"""
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **filters
    ) -> list["RetrievalResult"]:
        """检索相关文档"""
        ...

@runtime_checkable
class DocumentStore(Protocol):
    """文档存储：支持多种后端"""
    
    async def add_documents(self, documents: list["Document"]) -> None:
        """添加文档"""
        ...
    
    async def get_document(self, doc_id: str) -> "Document" | None:
        """获取文档"""
        ...

# RAG 工具（Agent 可直接使用）
class RAGTool:
    """RAG 检索工具，可被 Agent 调用"""
    
    name = "knowledge_base"
    description = "从知识库中检索相关信息"
    
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
    
    async def run(self, query: str, top_k: int = 3) -> str:
        results = await self.retriever.retrieve(query, top_k=top_k)
        return "\n\n".join([
            f"[{i+1}] {result.content}"
            for i, result in enumerate(results)
        ])

# 使用示例
from zent import Agent, OpenAILLM
from zent_rag import ChromaRetriever, HybridRetriever

# 创建 RAG 工具
retriever = HybridRetriever(
    vector_store=ChromaRetriever(collection="docs"),
    keyword_store=BM25Retriever(),
    reranker=CohereReranker()
)

agent = Agent(
    llm=OpenAILLM(model="gpt-4"),
    tools=[RAGTool(retriever)],
    system_prompt="你可以使用知识库检索工具查找信息。"
)
```

### 3. 多智能体扩展

```python
# 扩展协议：zent/core/multiagent.py
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum, auto

class AgentRole(Enum):
    WORKER = auto()
    MANAGER = auto()
    REVIEWER = auto()

@dataclass
class AgentConfig:
    name: str
    role: AgentRole
    agent: "Agent"
    description: str = ""

@runtime_checkable
class MessageBus(Protocol):
    """消息总线：Agent 间通信"""
    
    async def send(
        self, 
        from_agent: str, 
        to_agent: str, 
        message: str,
        message_type: str = "default"
    ) -> None:
        """发送消息"""
        ...
    
    async def broadcast(
        self, 
        from_agent: str, 
        message: str
    ) -> None:
        """广播消息"""
        ...
    
    async def subscribe(
        self, 
        agent_name: str, 
        callback: callable
    ) -> None:
        """订阅消息"""
        ...

@runtime_checkable
class AgentTeam(Protocol):
    """Agent 团队：多智能体协作"""
    
    def __init__(self, config: list[AgentConfig]):
        ...
    
    async def execute(
        self, 
        task: str,
        mode: str = "hierarchical"  # or "sequential", "joint"
    ) -> "TeamResult":
        """执行任务"""
        ...

# 协作模式实现
class HierarchicalTeam:
    """层级模式：Manager 分配任务给 Workers"""
    
    def __init__(self, config: list[AgentConfig]):
        self.manager = next(
            c for c in config if c.role == AgentRole.MANAGER
        )
        self.workers = [
            c for c in config if c.role == AgentRole.WORKER
        ]
        self.message_bus = InMemoryMessageBus()
    
    async def execute(self, task: str) -> "TeamResult":
        # 1. Manager 分析任务并分解
        plan = await self.manager.agent.run(
            f"分析任务并分解为子任务：{task}"
        )
        
        # 2. 分配给 Workers
        subtasks = self._parse_subtasks(plan.output)
        worker_results = []
        
        for subtask in subtasks:
            worker = self._select_worker(subtask)
            result = await worker.agent.run(subtask)
            worker_results.append(result)
        
        # 3. Manager 整合结果
        final = await self.manager.agent.run(
            f"整合以下结果：{[r.output for r in worker_results]}"
        )
        
        return TeamResult(
            output=final.output,
            steps=worker_results
        )

# 使用示例
from zent_multiagent import HierarchicalTeam, AgentConfig, AgentRole

team = HierarchicalTeam([
    AgentConfig(
        name="manager",
        role=AgentRole.MANAGER,
        agent=Agent(llm=gpt4, system_prompt="你是项目经理..."),
    ),
    AgentConfig(
        name="researcher",
        role=AgentRole.WORKER,
        agent=Agent(llm=gpt4, tools=[search_tool]),
    ),
    AgentConfig(
        name="writer",
        role=AgentRole.WORKER,
        agent=Agent(llm=gpt4, system_prompt="你是技术作家..."),
    ),
])

result = await team.execute("研究 MCP 协议并写一份报告")
```

### 4. 人机协同扩展

```python
# 扩展协议：zent/core/human_in_the_loop.py
from typing import Protocol, runtime_checkable
from enum import Enum, auto
from dataclasses import dataclass

class ApprovalType(Enum):
    TOOL_CALL = auto()
    SENSITIVE_DATA = auto()
    EXTERNAL_ACTION = auto()

@dataclass
class ApprovalRequest:
    type: ApprovalType
    description: str
    context: dict
    timeout: int = 300  # seconds

@runtime_checkable
class HumanInTheLoop(Protocol):
    """人机协同：在关键节点请求人类介入"""
    
    async def request_approval(
        self, 
        request: ApprovalRequest
    ) -> "ApprovalResult":
        """请求审批"""
        ...
    
    async def request_input(
        self, 
        prompt: str,
        options: list[str] | None = None
    ) -> str:
        """请求用户输入"""
        ...
    
    async def notify(
        self, 
        message: str,
        level: str = "info"
    ) -> None:
        """通知用户"""
        ...

# 控制台实现（开发调试）
class ConsoleHumanInterface(HumanInTheLoop):
    """控制台人机交互"""
    
    async def request_approval(self, request: ApprovalRequest):
        print(f"\n🔔 需要审批: {request.description}")
        print(f"类型: {request.type.name}")
        print(f"上下文: {request.context}")
        
        response = input("批准? (y/n): ")
        return ApprovalResult(
            approved=response.lower() == "y",
            feedback=""
        )
    
    async def request_input(self, prompt: str, options: list[str] | None = None):
        print(f"\n❓ {prompt}")
        if options:
            for i, opt in enumerate(options):
                print(f"  {i+1}. {opt}")
        return input("> ")

# 集成到 Agent
class AgentWithHumanLoop:
    """支持人机协同的 Agent 包装器"""
    
    def __init__(
        self, 
        agent: Agent,
        human: HumanInTheLoop,
        approval_rules: list[ApprovalType]
    ):
        self.agent = agent
        self.human = human
        self.approval_rules = approval_rules
        
        # 注册工具调用前钩子
        self.agent.on_tool_call = self._on_tool_call
    
    async def _on_tool_call(self, tool_call: ToolCall, result: ToolResult):
        # 检查是否需要审批
        if tool_call.name in ["transfer_money", "delete_data"]:
            approval = await self.human.request_approval(
                ApprovalRequest(
                    type=ApprovalType.SENSITIVE_DATA,
                    description=f"执行敏感操作: {tool_call.name}",
                    context=tool_call.arguments
                )
            )
            if not approval.approved:
                raise HumanRejectedError("操作被用户拒绝")
```

### 5. 安全治理扩展

```python
# 扩展协议：zent/core/security.py
from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass
class SecurityPolicy:
    """安全策略配置"""
    allowed_tools: list[str] | None = None  # None = all
    blocked_tools: list[str] = None
    max_token_spend: int | None = None
    require_approval_for: list[str] = None
    pii_detection: bool = True
    sandbox_enabled: bool = False

@runtime_checkable
class Sandbox(Protocol):
    """沙箱执行环境"""
    
    async def execute(
        self, 
        code: str,
        timeout: int = 30,
        memory_limit: str = "256m"
    ) -> "SandboxResult":
        """在沙箱中执行代码"""
        ...

@runtime_checkable
class PolicyEngine(Protocol):
    """策略引擎：权限控制与合规检查"""
    
    async def check_tool_permission(
        self, 
        tool_name: str, 
        user: str,
        context: dict
    ) -> bool:
        """检查工具调用权限"""
        ...
    
    async def detect_pii(self, text: str) -> list["PIIInstance"]:
        """检测 PII 信息"""
        ...
    
    async def redact_pii(self, text: str) -> str:
        """脱敏处理"""
        ...

# 实现示例
class DockerSandbox:
    """Docker 沙箱实现"""
    
    async def execute(self, code: str, timeout: int = 30, **kwargs):
        import docker
        client = docker.from_env()
        
        container = client.containers.run(
            "python:3.11-slim",
            command=f"python -c '{code}'",
            detach=True,
            mem_limit=kwargs.get("memory_limit", "256m"),
            network_mode="none",
            read_only=True
        )
        
        try:
            result = container.wait(timeout=timeout)
            logs = container.logs().decode()
            return SandboxResult(
                success=result["StatusCode"] == 0,
                output=logs,
                exit_code=result["StatusCode"]
            )
        finally:
            container.remove(force=True)
```

### 6. 可观测性扩展

```python
# 扩展协议：zent/core/observability.py
from typing import Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime
from contextvars import ContextVar
import uuid

# 追踪上下文
current_trace_id: ContextVar[str] = ContextVar("trace_id")

dataclass
class Span:
    """追踪 Span"""
    id: str
    trace_id: str
    parent_id: str | None
    name: str
    start_time: datetime
    end_time: datetime | None = None
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

@runtime_checkable
class Tracer(Protocol):
    """链路追踪器"""
    
    def start_span(self, name: str, **attributes) -> "Span":
        """开始 Span"""
        ...
    
    def end_span(self, span: Span) -> None:
        """结束 Span"""
        ...
    
    def add_event(self, span: Span, name: str, **attributes) -> None:
        """添加事件"""
        ...

@runtime_checkable
class MetricsCollector(Protocol):
    """指标收集器"""
    
    def record_token_usage(
        self, 
        model: str, 
        prompt_tokens: int, 
        completion_tokens: int
    ) -> None:
        """记录 Token 用量"""
        ...
    
    def record_latency(self, operation: str, duration_ms: float) -> None:
        """记录延迟"""
        ...
    
    def record_tool_call(self, tool_name: str, success: bool) -> None:
        """记录工具调用"""
        ...

# 集成到 Agent
class ObservableAgent:
    """带可观测性的 Agent 包装器"""
    
    def __init__(
        self, 
        agent: Agent,
        tracer: Tracer,
        metrics: MetricsCollector
    ):
        self.agent = agent
        self.tracer = tracer
        self.metrics = metrics
    
    async def run(self, query: str, **context):
        trace_id = str(uuid.uuid4())
        current_trace_id.set(trace_id)
        
        span = self.tracer.start_span(
            "agent.run",
            query=query,
            agent_name=self.agent.name
        )
        
        try:
            start = datetime.now()
            result = await self.agent.run(query, **context)
            duration = (datetime.now() - start).total_seconds() * 1000
            
            # 记录指标
            self.metrics.record_latency("agent.run", duration)
            if result.usage:
                self.metrics.record_token_usage(
                    self.agent.llm.model,
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens
                )
            
            return result
        finally:
            self.tracer.end_span(span)

# 导出到 LangSmith
class LangSmithExporter:
    """LangSmith 导出器"""
    
    def export(self, trace: list[Span]):
        import langsmith
        client = langsmith.Client()
        
        for span in trace:
            client.create_run(
                name=span.name,
                run_type="chain",
                inputs=span.attributes.get("inputs"),
                outputs=span.attributes.get("outputs"),
                start_time=span.start_time,
                end_time=span.end_time
            )
```

### 7. 评估体系扩展

```python
# 扩展协议：zent/core/evaluation.py
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum

class EvaluationMetric(Enum):
    RELEVANCE = "relevance"
    ACCURACY = "accuracy"
    HELPFULNESS = "helpfulness"
    SAFETY = "safety"

@dataclass
class EvaluationResult:
    score: float  # 0-1
    feedback: str
    metric: EvaluationMetric

@runtime_checkable
class Evaluator(Protocol):
    """评估器：LLM-as-a-Judge"""
    
    async def evaluate(
        self,
        input_query: str,
        output: str,
        expected: str | None = None,
        metrics: list[EvaluationMetric] | None = None
    ) -> list[EvaluationResult]:
        """评估 Agent 输出"""
        ...

@runtime_checkable
class Benchmark(Protocol):
    """基准测试套件"""
    
    async def run(self, agent: Agent) -> "BenchmarkResult":
        """运行基准测试"""
        ...
    
    def report(self, result: "BenchmarkResult") -> str:
        """生成报告"""
        ...

# 实现示例
class LLMJudgeEvaluator:
    """LLM 作为评判者"""
    
    def __init__(self, judge_llm: LLM):
        self.judge = judge_llm
    
    async def evaluate(self, input_query, output, expected=None, metrics=None):
        results = []
        
        for metric in (metrics or [EvaluationMetric.RELEVANCE]):
            prompt = self._build_judge_prompt(
                metric, input_query, output, expected
            )
            
            response = await self.judge.complete([
                Message.system("You are an expert evaluator..."),
                Message.user(prompt)
            ])
            
            score, feedback = self._parse_judge_response(response.content)
            results.append(EvaluationResult(
                score=score,
                feedback=feedback,
                metric=metric
            ))
        
        return results

# 回归测试
class RegressionTestSuite:
    """回归测试套件"""
    
    def __init__(self, test_cases: list["TestCase"]):
        self.test_cases = test_cases
    
    async def run(self, agent: Agent) -> "RegressionResult":
        results = []
        
        for test in self.test_cases:
            result = await agent.run(test.input)
            
            # 检查是否与预期一致
            passed = self._check_result(result, test.expected)
            results.append({
                "test": test.name,
                "passed": passed,
                "input": test.input,
                "output": result.output,
                "expected": test.expected
            })
        
        return RegressionResult(
            total=len(test_cases),
            passed=sum(1 for r in results if r["passed"]),
            failed=[r for r in results if not r["passed"]],
            details=results
        )
```

### 8. 工程扩展

```python
# 扩展协议：zent/core/engineering.py
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@runtime_checkable
class Checkpoint(Protocol):
    """检查点：状态持久化"""
    
    async def save(
        self, 
        state: "AgentState",
        checkpoint_id: str | None = None
    ) -> str:
        """保存状态"""
        ...
    
    async def load(self, checkpoint_id: str) -> "AgentState":
        """加载状态"""
        ...
    
    async def list_checkpoints(self) -> list["CheckpointInfo"]:
        """列出所有检查点"""
        ...

@runtime_checkable
class TaskQueue(Protocol):
    """任务队列：支持后台执行"""
    
    async def enqueue(self, task: "AgentTask") -> str:
        """添加任务"""
        ...
    
    async def dequeue(self) -> "AgentTask" | None:
        """获取任务"""
        ...
    
    async def get_status(self, task_id: str) -> "TaskStatus":
        """查询任务状态"""
        ...

# 实现示例
class FileCheckpoint:
    """文件系统检查点"""
    
    def __init__(self, directory: str = "./checkpoints"):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=True)
    
    async def save(self, state: "AgentState", checkpoint_id: str | None = None):
        import json
        
        checkpoint_id = checkpoint_id or str(uuid.uuid4())
        file_path = self.directory / f"{checkpoint_id}.json"
        
        with open(file_path, "w") as f:
            json.dump(state.to_dict(), f)
        
        return checkpoint_id
    
    async def load(self, checkpoint_id: str) -> "AgentState":
        import json
        
        file_path = self.directory / f"{checkpoint_id}.json"
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        return AgentState.from_dict(data)

# 后台 Agent
class BackgroundAgent:
    """支持后台执行的 Agent"""
    
    def __init__(
        self, 
        agent: Agent,
        queue: TaskQueue,
        checkpoint: Checkpoint
    ):
        self.agent = agent
        self.queue = queue
        self.checkpoint = checkpoint
    
    async def submit_task(self, query: str, **context) -> str:
        """提交后台任务"""
        task = AgentTask(
            id=str(uuid.uuid4()),
            query=query,
            context=context,
            status="pending",
            created_at=datetime.now()
        )
        
        await self.queue.enqueue(task)
        return task.id
    
    async def get_result(self, task_id: str) -> "AgentResult" | None:
        """获取任务结果"""
        status = await self.queue.get_status(task_id)
        
        if status.state == "completed":
            return status.result
        elif status.state == "failed":
            raise TaskFailedError(status.error)
        else:
            return None  # 仍在执行
```

---

## 扩展机制

### 自定义 LLM

实现 `LLM` 协议即可接入自定义模型：

```python
from zent import LLM, Message, LLMResponse

class MyLLM:
    async def complete(self, messages, tools=None, **kwargs):
        # 调用你的 LLM API
        response = await call_my_api(messages)
        return LLMResponse(
            content=response.text,
            tool_calls=[],
            usage=response.usage,
            model="my-model"
        )
    
    async def stream(self, messages, **kwargs):
        async for chunk in stream_my_api(messages):
            yield chunk

agent = Agent(llm=MyLLM(), tools=[])
```

### 自定义 Tool

```python
from zent import BaseTool

class APITool(BaseTool):
    name = "weather"
    description = "获取天气信息"
    parameters = {
        "type": "object",
        "properties": {
            "city": {"type": "string"}
        },
        "required": ["city"]
    }
    
    async def run(self, city: str) -> str:
        # 调用天气 API
        return f"{city} 今天晴天，25°C"
```

### 自定义 Memory

```python
from zent import Memory, Message

class RedisMemory:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
    
    async def add(self, message: Message):
        # 存储到 Redis
        pass
    
    async def get_context(self, query=None, limit=10):
        # 从 Redis 检索
        pass
    
    async def clear(self):
        pass
```

---

## 扩展包

扩展包作为独立 Python 包发布：

```bash
# 基础安装
pip install zent

# 带向量记忆
pip install zent zent-chroma

# 完整功能
pip install zent[all]
```

### zent-chroma 示例

```python
from zent import Agent
from zent_chroma import ChromaMemory

agent = Agent(
    llm=OpenAILLM(model="gpt-4"),
    memory=ChromaMemory(
        collection="my_agent",
        embedding_model="text-embedding-3-small"
    ),
    tools=tools
)
```

---

## 扩展注册机制

扩展包通过 `zent.extensions` 命名空间自动注册：

```python
# zent/extensions.py
import importlib
from typing import Type, TypeVar

T = TypeVar("T")

class ExtensionRegistry:
    """扩展注册表"""
    
    _memory_impls: dict[str, Type["Memory"]] = {}
    _llm_impls: dict[str, Type["LLM"]] = {}
    _planner_impls: dict[str, Type["Planner"]] = {}
    
    @classmethod
    def register_memory(cls, name: str, impl: Type["Memory"]):
        cls._memory_impls[name] = impl
    
    @classmethod
    def register_llm(cls, name: str, impl: Type["LLM"]):
        cls._llm_impls[name] = impl
    
    @classmethod
    def get_memory(cls, name: str) -> Type["Memory"] | None:
        return cls._memory_impls.get(name)
    
    @classmethod
    def auto_discover(cls):
        """自动发现已安装的扩展"""
        extensions = [
            "zent_chroma",
            "zent_security",
            "zent_evaluation",
            "zent_multiagent",
        ]
        
        for ext_name in extensions:
            try:
                ext = importlib.import_module(ext_name)
                if hasattr(ext, "register"):
                    ext.register(cls)
            except ImportError:
                pass  # 扩展未安装

# 扩展包注册示例（zent-chroma/zent_chroma/__init__.py）
def register(registry):
    from .memory import ChromaVectorMemory
    registry.register_memory("chroma", ChromaVectorMemory)

# 配置驱动使用
from zent import Agent
from zent.extensions import ExtensionRegistry

ExtensionRegistry.auto_discover()

# 通过配置字符串创建 Agent
agent = Agent.from_config({
    "llm": {
        "provider": "openai",
        "model": "gpt-4"
    },
    "memory": {
        "provider": "chroma",  # 自动使用已注册的 ChromaVectorMemory
        "collection": "my_agent",
        "embedding_model": "text-embedding-3-small"
    },
    "tools": ["search", "calculator"]
})
```

---

## 路线图

### Phase 1: 核心功能（MVP）- 已完成设计
- [x] Agent 门面类（Agent.run() ReAct 循环）
- [x] Tool 协议 + @tool 装饰器
- [x] LLM 协议 + OpenAI/Claude/Ollama 适配
- [x] Memory 协议 + WorkingMemory 实现
- [x] ReAct Planner 基础实现

### Phase 2: 生态集成（3-4 周）
- [ ] MCP 客户端（轻量内置版，stdio/sse）
- [ ] ShortTermMemory（带摘要功能）
- [ ] Plan-and-Solve、Self-Correction Planners
- [ ] 内置工具库（文件、HTTP、数据库）
- [ ] 完整测试套件 + 文档

### Phase 3: 扩展包（按需开发）
| 扩展包 | 优先级 | 说明 |
|:---|:---|:---|
| zent-chroma | P0 | 向量记忆，最常用扩展 |
| zent-rag | P0 | RAG 检索工具 |
| zent-observability | P1 | 链路追踪（LangSmith 导出） |
| zent-evaluation | P1 | LLM-as-a-Judge |
| zent-multiagent | P2 | 多智能体协作 |
| zent-security | P2 | 沙箱、权限控制 |
| zent-human | P2 | 人机协同界面 |
| zent-engineering | P3 | 检查点、任务队列 |

### Phase 4: 生产就绪（2-3 周）
- [ ] 性能基准测试
- [ ] 错误处理与重试机制
- [ ] 完整示例与教程
- [ ] API 稳定性保证（语义化版本）

---

## 与 LangChain 等框架的对比

| 特性 | Zent | LangChain | AutoGen |
|:---|:---|:---|:---|
| **核心设计** | Protocol 驱动，极简 | 抽象基类，丰富 | 多 Agent 对话 |
| **学习曲线** | 平缓 | 陡峭 | 中等 |
| **扩展方式** | 扩展包按需安装 | 内置大量组件 | 团队/对话模式 |
| **MCP 支持** | 一等公民 | 通过适配器 | 有限 |
| **部署体积** | 小（核心<100KB）| 大（依赖多）| 中等 |
| **适用场景** | 轻量级应用、快速原型 | 复杂企业应用 | 多 Agent 协作 |

**选择建议**：
- 需要快速上手、轻量级部署 → **Zent**
- 复杂工作流、丰富生态 → **LangChain**
- 多角色协作、对话模拟 → **AutoGen**

---

## 总结

Zent 是一个**极简但可扩展**的 Python Agent 开发框架：

### 核心特点
- **5 个核心抽象**（Agent、Tool、LLM、Memory、Planner）构成最小能力集
- **Protocol 驱动**实现零成本扩展，无继承负担
- **分层架构**支持渐进增强（核心层 + 组件层 + 扩展层）
- **MCP 优先**设计，复用工具生态
- **混合 API**（底层完整控制 + 高层装饰器语法糖）

### 设计哲学
> **"做减法而非加法"** —— 让框架消失，让 Agent 逻辑浮现。

### 架构优势
1. **渐进式采用** - 从简单 Agent 开始，按需添加扩展
2. **生态兼容** - MCP 工具可直接使用，无缝集成现有生态
3. **生产就绪** - 预留扩展点，支持从原型到生产的平滑过渡
4. **开发者友好** - 类型安全、异步优先、Pythonic API

### 下一步行动
1. **审查设计** - 确认架构和 API 设计是否符合预期
2. **Phase 1 实现** - 开始核心功能开发
3. **创建 Worktree** - 使用 Git Worktree 创建隔离开发环境
4. **编写计划** - 制定详细的实现计划和时间表

---

*设计文档版本: v1.0*  
*最后更新: 2024-XX-XX*
