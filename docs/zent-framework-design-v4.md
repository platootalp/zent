# Zent Agent 框架设计文档 V4

> **核心理念**: ABC抽象基类，模板扩展，渐进增强，生产就绪

## 1. 设计原则

### 1.1 核心原则
- **ABC抽象基类**: 使用Python ABC定义核心接口，提供默认实现
- **模板扩展**: Template Method 模式实现 Agent 可扩展性
- **显式优于隐式**: 配置和依赖显式声明
- **渐进增强**: 从简单开始，按需叠加复杂度
- **类型安全**: 强类型 Memory 步骤，提升可观测性

### 1.2 架构决策
- **3层架构**: Core → Integrations → App
- **4个核心抽象**: Model, Tool, Memory, Agent (均使用ABC)
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
│  Model, Tool, Memory, Agent (Protocols + Base Classes)       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心关系图

```
Agent (抽象基类 - Template Method)
├── model: Model           # 语言模型
├── tools: ToolRegistry    # 工具注册表
├── memory: Memory         # 强类型记忆
├── config: AgentConfig    # 配置
│
├── _run(task)             # 模板方法（框架）
│   ├── _initialize()
│   ├── while not done:
│   │   ├── _plan()        # 可选
│   │   └── _step()        # 抽象 - 子类实现
│   └── _finalize()
│
└── _step()                # 抽象方法（子类实现）

    ToolCallingAgent        CodeAgent
    (继承 Agent)           (继承 Agent)
    ├── _step()             ├── _step()
    │   └── 调用 LLM         │   └── 生成代码
    │       Function         │       AST 执行
    │       Calling
    └── _parse_tool_calls   └── _execute_code
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
│   ├── messages.py          # Message, MessageRole, ToolCall
│   ├── steps.py             # MemoryStep, ActionStep, TaskStep
│   ├── agent.py             # Agent 抽象基类 (Template Method)
│   ├── model.py             # BaseModel ABC
│   ├── tool.py              # BaseTool ABC, ToolRegistry
│   └── agent.py             # Agent ABC, AgentConfig
│
├── agents/                  # Agent 实现
│   ├── __init__.py
│   ├── tool_calling.py      # ToolCallingAgent
│   └── code.py              # CodeAgent
│
├── integrations/            # 集成层
│   ├── __init__.py
│   ├── models/              # 模型适配
│   │   ├── openai.py
│   │   └── anthropic.py
│   ├── tools/               # 工具实现
│   │   ├── function.py      # FunctionTool
│   │   └── mcp.py           # MCP 工具适配
│   └── memory/              # 记忆实现
│       └── in_memory.py
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
├── 05-code-agent.py
└── 06-multi-agent.py
```

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
    """记忆协议 - 存储强类型 MemoryStep"""
    
    async def add(self, step: "MemoryStep") -> None: ...
    
    async def get_messages(self, limit: int = 10) -> list[Message]: ...
    
    async def get_steps(self, limit: int = 10) -> list["MemoryStep"]: ...
    
    async def clear(self) -> None: ...
```

### 4.2 强类型 Memory 步骤（zent/core/memory_steps.py）

```python
"""
强类型 Memory 步骤 - 灵感来自 smolagents
每个步骤都有明确的语义，便于追踪和调试
"""
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime
from enum import Enum

from zent.core.protocols import Message, ToolCall, ToolResult


class StepType(str, Enum):
    """步骤类型"""
    SYSTEM = "system"
    TASK = "task"
    PLANNING = "planning"
    ACTION = "action"
    FINAL_ANSWER = "final_answer"


@dataclass
class MemoryStep:
    """记忆步骤基类"""
    step_type: StepType
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class SystemPromptStep(MemoryStep):
    """系统提示词步骤"""
    system_prompt: str = ""
    
    def __post_init__(self):
        self.step_type = StepType.SYSTEM


@dataclass
class TaskStep(MemoryStep):
    """任务步骤 - 记录用户输入"""
    task: str = ""
    
    def __post_init__(self):
        self.step_type = StepType.TASK


@dataclass
class PlanningStep(MemoryStep):
    """规划步骤 - 记录 Agent 的思考计划"""
    plan: str = ""
    facts: str = ""
    
    def __post_init__(self):
        self.step_type = StepType.PLANNING


@dataclass
class ActionStep(MemoryStep):
    """动作步骤 - 记录工具调用/代码执行"""
    tool_calls: List[ToolCall] = field(default_factory=list)
    observations: str = ""
    error: Optional[str] = None
    duration: float = 0.0  # 执行耗时（秒）
    
    def __post_init__(self):
        self.step_type = StepType.ACTION


@dataclass
class FinalAnswerStep(MemoryStep):
    """最终答案步骤"""
    answer: str = ""
    
    def __post_init__(self):
        self.step_type = StepType.FINAL_ANSWER
```

### 4.3 Agent 抽象基类（zent/core/agent.py）

```python
"""
Agent 抽象基类 - 使用 Template Method 模式

设计灵感:
- smolagents: Template Method 模式实现可扩展性
- LangChain: Runnable 协议实现组合性

子类必须实现:
- _step(): 单步逻辑
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any
import uuid
import time

from zent.core.protocols import (
    Model, Tool, Memory, Message, MessageRole,
    ToolCall, ToolResult, ModelResponse
)
from zent.core.memory_steps import (
    MemoryStep, TaskStep, ActionStep, FinalAnswerStep, PlanningStep
)
from zent.core.result import AgentResult
from zent.core.registry import ToolRegistry


@dataclass
class AgentConfig:
    """Agent 配置"""
    model: Model
    tools: List[Tool] = field(default_factory=list)
    memory: Optional[Memory] = None
    max_iterations: int = 10
    system_prompt: Optional[str] = None
    planning_interval: Optional[int] = None  # 每 N 步重新规划
    
    # 回调函数
    on_step: Optional[Callable[[MemoryStep], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None


class Agent(ABC):
    """
    Agent 抽象基类 - Template Method 模式
    
    子类只需实现 _step() 方法，框架处理:
    - ReAct 循环管理
    - 记忆存储
    - 回调触发
    - 错误处理
    
    Usage (子类实现):
        class MyAgent(Agent):
            async def _step(self) -> ActionStep | FinalAnswerStep:
                # 实现单步逻辑
                pass
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.tools = ToolRegistry(config.tools)
        self.memory = config.memory
        self.step_number = 0
        self.completed = False
        self._task_id: Optional[str] = None
        
        # 初始化记忆
        if self.memory and config.system_prompt:
            import asyncio
            asyncio.create_task(self.memory.add(
                SystemPromptStep(system_prompt=config.system_prompt)
            ))
    
    async def run(self, task: str, **context) -> AgentResult:
        """
        执行任务的模板方法 - 定义 ReAct 循环框架
        
        子类不应覆盖此方法，而是实现 _step()
        """
        self._task_id = str(uuid.uuid4())[:8]
        self.step_number = 0
        self.completed = False
        
        try:
            # 1. 初始化
            await self._initialize(task)
            
            # 2. ReAct 循环
            while not self.completed and self.step_number < self.config.max_iterations:
                self.step_number += 1
                
                # 2.1 规划（如果配置了规划间隔）
                if self._should_plan():
                    await self._do_planning()
                
                # 2.2 执行单步（子类实现）
                step_start = time.time()
                step_result = await self._step()
                
                if isinstance(step_result, ActionStep):
                    step_result.duration = time.time() - step_start
                
                # 2.3 存储到记忆
                if self.memory:
                    await self.memory.add(step_result)
                
                # 2.4 触发回调
                if self.config.on_step:
                    self.config.on_step(step_result)
                
                # 2.5 检查是否完成
                if isinstance(step_result, FinalAnswerStep):
                    self.completed = True
                    return AgentResult(
                        output=step_result.answer,
                        steps=await self._get_steps(),
                        success=True,
                        task_id=self._task_id
                    )
            
            # 3. 达到最大迭代次数
            return AgentResult(
                output="达到最大迭代次数",
                steps=await self._get_steps(),
                success=False,
                task_id=self._task_id
            )
            
        except Exception as e:
            if self.config.on_error:
                self.config.on_error(e)
            
            return AgentResult(
                output=str(e),
                steps=await self._get_steps(),
                success=False,
                error=e,
                task_id=self._task_id
            )
    
    async def _initialize(self, task: str) -> None:
        """初始化任务 - 可被子类覆盖"""
        task_step = TaskStep(task=task)
        if self.memory:
            await self.memory.add(task_step)
    
    def _should_plan(self) -> bool:
        """检查是否需要规划"""
        return (
            self.config.planning_interval is not None
            and self.step_number % self.config.planning_interval == 0
        )
    
    async def _do_planning(self) -> None:
        """执行规划步骤 - 可被子类覆盖"""
        pass  # 默认不实现，子类可覆盖
    
    @abstractmethod
    async def _step(self) -> ActionStep | FinalAnswerStep:
        """
        单步逻辑 - 子类必须实现
        
        Returns:
            ActionStep: 需要继续循环
            FinalAnswerStep: 任务完成
        """
        pass
    
    async def _get_steps(self) -> List[MemoryStep]:
        """获取所有步骤"""
        if self.memory:
            return await self.memory.get_steps(limit=1000)
        return []
    
    async def stream(self, task: str, **context):
        """
        流式执行 - 生成中间步骤
        
        子类可覆盖以提供更细粒度的流式输出
        """
        # 默认实现：yield 每一步
        self.config.on_step = lambda step: self._yield_step(step)
        result = await self.run(task, **context)
        yield result
    
    def _yield_step(self, step: MemoryStep):
        """用于流式输出的辅助方法"""
        pass  # 实际实现需要更复杂的机制
```

### 4.4 ToolCallingAgent 实现（zent/agents/tool_calling.py）

```python
"""
ToolCallingAgent - 使用 LLM 原生 Function Calling

设计灵感:
- smolagents: ToolCallingAgent 实现
- LangChain: Tool binding 机制
"""
from typing import List
import json

from zent.core.agent import Agent, AgentConfig
from zent.core.protocols import Message, MessageRole, ToolCall
from zent.core.memory_steps import ActionStep, FinalAnswerStep


class ToolCallingAgent(Agent):
    """
    使用 LLM 原生 Function Calling 的 Agent
    
    适用于:
    - GPT-4, Claude 等支持 Function Calling 的模型
    - 需要精确工具调用的场景
    - 生产环境（更可控）
    
    Usage:
        agent = ToolCallingAgent(AgentConfig(
            model=OpenAIModel(...),
            tools=[search, calculator]
        ))
        result = await agent.run("计算 2+2")
    """
    
    async def _step(self) -> ActionStep | FinalAnswerStep:
        """
        单步逻辑:
        1. 构建消息历史
        2. 调用 LLM 获取 tool_calls
        3. 执行工具
        4. 返回 ActionStep 或 FinalAnswerStep
        """
        # 1. 获取消息历史
        messages = await self._build_messages()
        
        # 2. 调用模型
        response = await self.config.model.generate(
            messages=messages,
            tools=self.tools.get_tools() if self.tools else None
        )
        
        # 3. 处理 tool_calls
        if response.tool_calls:
            observations = []
            
            for tool_call in response.tool_calls:
                result = await self._execute_tool(tool_call)
                observations.append(f"[{tool_call.name}] {result.output}")
            
            return ActionStep(
                tool_calls=response.tool_calls,
                observations="\n".join(observations)
            )
        
        # 4. 返回最终答案
        elif response.content:
            return FinalAnswerStep(answer=response.content)
        
        # 5. 空响应处理
        else:
            return ActionStep(
                observations="模型返回空响应",
                error="Empty response"
            )
    
    async def _build_messages(self) -> List[Message]:
        """构建消息历史"""
        if self.memory:
            return await self.memory.get_messages(limit=20)
        return []
    
    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """执行工具调用"""
        tool = self.tools.get(call.name)
        
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
```

### 4.5 CodeAgent 实现（zent/agents/code.py）

```python
"""
CodeAgent - 生成并执行 Python 代码

设计灵感:
- smolagents: CodeAgent 实现
- 优势: 任意 LLM 可用，代码表达能力强

安全机制:
- AST 解析执行（非 exec）
- 白名单导入检查
- 危险函数拦截
"""
from typing import List
import ast
import re

from zent.core.agent import Agent, AgentConfig
from zent.core.protocols import Message, MessageRole
from zent.core.memory_steps import ActionStep, FinalAnswerStep


class CodeAgent(Agent):
    """
    生成并执行 Python 代码的 Agent
    
    适用于:
    - 不支持 Function Calling 的模型
    - 复杂数据/计算任务
    - 需要多工具组合的场景
    
    Usage:
        agent = CodeAgent(AgentConfig(
            model=LocalModel(...),
            tools=[search, calculator],
            additional_authorized_imports=["numpy", "pandas"]
        ))
        result = await agent.run("分析数据")
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.authorized_imports = getattr(
            config, 'authorized_imports', 
            ['math', 'random', 'datetime', 'json', 're']
        )
        self.python_executor = LocalPythonExecutor(
            self.authorized_imports,
            self.tools
        )
    
    async def _step(self) -> ActionStep | FinalAnswerStep:
        """
        单步逻辑:
        1. 构建包含工具描述的提示词
        2. 调用 LLM 生成代码
        3. AST 安全执行
        4. 返回结果
        """
        # 1. 构建提示词
        prompt = self._build_code_prompt()
        messages = await self._build_messages()
        messages.append(Message(role=MessageRole.USER, content=prompt))
        
        # 2. 生成代码
        response = await self.config.model.generate(messages=messages)
        code = self._parse_code(response.content)
        
        # 3. 安全执行
        try:
            result = await self.python_executor.execute(code)
            
            if result.error:
                return ActionStep(
                    observations=result.output,
                    error=result.error
                )
            
            # 检查结果是否包含最终答案
            if self._is_final_answer(result.output):
                return FinalAnswerStep(answer=result.output)
            
            return ActionStep(observations=result.output)
            
        except Exception as e:
            return ActionStep(
                observations="",
                error=str(e)
            )
    
    def _build_code_prompt(self) -> str:
        """构建代码生成提示词"""
        tool_descriptions = self.tools.get_code_descriptions()
        return f"""你是一个可以编写 Python 代码解决问题的 Agent。

可用工具:
{tool_descriptions}

请编写 Python 代码解决问题。代码将被安全执行。

要求:
1. 使用 ```python 代码块格式
2. 使用 print() 输出结果
3. 可直接调用上述工具函数
"""
    
    def _parse_code(self, content: str) -> str:
        """从响应中提取代码块"""
        pattern = r'```python\n(.*?)\n```'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1) if match else content
    
    def _is_final_answer(self, output: str) -> bool:
        """检查是否为最终答案"""
        # 简单启发式：包含特定关键词
        final_keywords = ['final_answer', '最终结果', '答案是']
        return any(kw in output.lower() for kw in final_keywords)


class LocalPythonExecutor:
    """本地 Python 安全执行器"""
    
    def __init__(self, authorized_imports: List[str], tools: ToolRegistry):
        self.authorized_imports = authorized_imports
        self.tools = tools
        self.state = {}
    
    async def execute(self, code: str) -> 'ExecutionResult':
        """AST 安全执行代码"""
        try:
            tree = ast.parse(code)
            
            # 安全检查
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.authorized_imports:
                            return ExecutionResult(
                                output="",
                                error=f"未授权导入: {alias.name}"
                            )
            
            # 执行
            exec_globals = {
                **{name: tool.run for name, tool in self.tools.items()},
                '__builtins__': __builtins__
            }
            
            exec(code, exec_globals, self.state)
            
            output = self.state.get('_output', '')
            return ExecutionResult(output=str(output), error=None)
            
        except Exception as e:
            return ExecutionResult(output="", error=str(e))


@dataclass
class ExecutionResult:
    output: str
    error: Optional[str] = None
```

### 4.6 Tool 注册表（zent/core/registry.py）

```python
"""ToolRegistry - 工具注册表"""
from typing import Dict, List, Iterator
from zent.core.protocols import Tool


class ToolRegistry:
    """工具注册表 - 统一管理工具"""
    
    def __init__(self, tools: List[Tool] = None):
        self._tools: Dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)
    
    def register(self, tool: Tool) -> None:
        """注册工具"""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Tool | None:
        """获取工具"""
        return self._tools.get(name)
    
    def get_tools(self) -> List[Tool]:
        """获取所有工具"""
        return list(self._tools.values())
    
    def get_code_descriptions(self) -> str:
        """获取代码格式的工具描述"""
        descriptions = []
        for tool in self._tools.values():
            desc = f"def {tool.name}({self._format_params(tool)}):\n"
            desc += f'    """{tool.description}"""'
            descriptions.append(desc)
        return "\n\n".join(descriptions)
    
    def _format_params(self, tool: Tool) -> str:
        """格式化参数"""
        params = tool.parameters.get('properties', {})
        required = tool.parameters.get('required', [])
        
        parts = []
        for name, schema in params.items():
            param_type = schema.get('type', 'any')
            if name in required:
                parts.append(f"{name}: {param_type}")
            else:
                parts.append(f"{name}: {param_type} = None")
        
        return ", ".join(parts)
    
    def __iter__(self) -> Iterator[tuple[str, Tool]]:
        return iter(self._tools.items())
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools
    
    def __len__(self) -> int:
        return len(self._tools)
```

### 4.7 应用层 - 装饰器（zent/app/decorators.py）

```python
"""
装饰器 - 简化工具创建

设计灵感:
- smolagents: @tool 装饰器
- LangChain: @tool 装饰器

双重 API:
- 装饰器方式: @tool
- 类方式: class MyTool(Tool)
"""
import inspect
from typing import Callable, get_type_hints
from dataclasses import dataclass

from zent.core.protocols import Tool


@dataclass
class FunctionTool:
    """函数工具 - 将函数包装为 Tool"""
    
    func: Callable
    name: str
    description: str
    _parameters: dict = None
    
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
            json_type = self._python_type_to_json(param_type)
            
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
        
        return str(result) if not isinstance(result, str) else result
    
    def _python_type_to_json(self, py_type: type) -> str:
        """Python 类型转 JSON Schema 类型"""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        return type_map.get(py_type, "string")


def tool(name_or_func: str | Callable = None, description: str = None):
    """
    工具装饰器 - 将函数转换为 Tool
    
    Usage:
        # 方式1: 无参数装饰器
        @tool
        def search(query: str) -> str:
            \"\"\"搜索知识库\"\"\"
            return f"Results: {query}"
        
        # 方式2: 带参数装饰器
        @tool(name="web_search", description="搜索网络")
        def search(query: str) -> str:
            return f"Results: {query}"
        
        # 方式3: 直接使用函数
        search_tool = tool(search)
    """
    def decorator(func: Callable) -> FunctionTool:
        tool_name = name if isinstance(name, str) else func.__name__
        tool_desc = description or inspect.getdoc(func) or ""
        
        return FunctionTool(
            func=func,
            name=tool_name,
            description=tool_desc
        )
    
    # 支持 @tool 无括号用法
    if callable(name_or_func):
        func = name_or_func
        name = func.__name__
        return decorator(func)
    
    name = name_or_func
    return decorator
```

### 4.8 应用层 - 工厂函数（zent/app/factory.py）

```python
"""
工厂函数 - 简化 Agent 创建

提供 LangChain 风格的 create_agent API
"""
from typing import List, Optional

from zent.core.agent import AgentConfig
from zent.core.protocols import Model, Tool, Memory
from zent.agents.tool_calling import ToolCallingAgent
from zent.agents.code import CodeAgent


def create_agent(
    model: Model | str,
    tools: List[Tool] | None = None,
    memory: Memory | None = None,
    system_prompt: str | None = None,
    agent_type: str = "tool_calling",
    **kwargs
) -> Agent:
    """
    创建 Agent 的工厂函数
    
    Args:
        model: Model 实例或字符串 (如 "openai:gpt-4")
        tools: 工具列表
        memory: 记忆实现
        system_prompt: 系统提示词
        agent_type: Agent 类型 ("tool_calling" | "code")
        **kwargs: 额外配置
    
    Returns:
        Agent 实例
    
    Usage:
        # 方式1: 字符串快速创建
        agent = create_agent("openai:gpt-4", tools=[search])
        
        # 方式2: 指定 Agent 类型
        agent = create_agent(
            "anthropic:claude-3-opus",
            tools=[search],
            agent_type="code"
        )
        
        # 方式3: 完整配置
        agent = create_agent(
            model=OpenAIModel(api_key="..."),
            tools=[search, calculator],
            memory=InMemoryMemory(),
            system_prompt="你是一个 helpful 助手。",
            max_iterations=15,
            planning_interval=3
        )
    """
    # 解析模型字符串
    if isinstance(model, str):
        model = _resolve_model(model)
    
    # 创建配置
    config = AgentConfig(
        model=model,
        tools=tools or [],
        memory=memory,
        system_prompt=system_prompt,
        **kwargs
    )
    
    # 创建对应类型的 Agent
    if agent_type == "tool_calling":
        return ToolCallingAgent(config)
    elif agent_type == "code":
        return CodeAgent(config)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")


def _resolve_model(model_str: str) -> Model:
    """解析模型字符串"""
    if ":" not in model_str:
        raise ValueError(f"Model string must be in format 'provider:model', got: {model_str}")
    
    provider, model_name = model_str.split(":", 1)
    
    if provider == "openai":
        from zent.integrations.models.openai import OpenAIModel
        return OpenAIModel(model=model_name)
    elif provider == "anthropic":
        from zent.integrations.models.anthropic import AnthropicModel
        return AnthropicModel(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}. Supported: openai, anthropic")
```

---

## 5. 集成层设计

### 5.1 OpenAI 适配（zent/integrations/models/openai.py）

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
        base_url: str | None = None,
        **kwargs
    ):
        if not HAS_OPENAI:
            raise ImportError("pip install openai")
        
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
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

### 5.2 内存实现（zent/integrations/memory/in_memory.py）

```python
"""内存记忆实现"""
from dataclasses import dataclass, field
from typing import List

from zent.core.protocols import Memory, Message, MessageRole
from zent.core.memory_steps import MemoryStep, TaskStep, ActionStep, FinalAnswerStep


@dataclass
class InMemoryMemory:
    """内存记忆实现"""
    
    _steps: List[MemoryStep] = field(default_factory=list)
    max_steps: int = 100
    
    async def add(self, step: MemoryStep) -> None:
        """添加步骤"""
        self._steps.append(step)
        
        # 限制最大步数
        if len(self._steps) > self.max_steps:
            self._steps = self._steps[-self.max_steps:]
    
    async def get_messages(self, limit: int = 10) -> List[Message]:
        """获取消息格式（用于 LLM 上下文）"""
        messages = []
        
        for step in self._steps[-limit:]:
            if isinstance(step, TaskStep):
                messages.append(Message(role=MessageRole.USER, content=step.task))
            elif isinstance(step, ActionStep):
                # 工具调用作为 assistant 消息
                if step.tool_calls:
                    messages.append(Message(
                        role=MessageRole.ASSISTANT,
                        content="",
                        metadata={"tool_calls": step.tool_calls}
                    ))
                # 观察结果作为 tool 消息
                if step.observations:
                    messages.append(Message(
                        role=MessageRole.TOOL,
                        content=step.observations
                    ))
            elif isinstance(step, FinalAnswerStep):
                messages.append(Message(
                    role=MessageRole.ASSISTANT,
                    content=step.answer
                ))
        
        return messages
    
    async def get_steps(self, limit: int = 10) -> List[MemoryStep]:
        """获取步骤列表"""
        return self._steps[-limit:] if self._steps else []
    
    async def clear(self) -> None:
        """清空记忆"""
        self._steps.clear()
```

---

## 6. 使用示例

### 6.1 基础用法

```python
import asyncio
from zent import create_agent, tool
from zent.integrations.models.openai import OpenAIModel
from zent.integrations.memory.in_memory import InMemoryMemory


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

### 6.2 使用 CodeAgent

```python
from zent import create_agent, tool

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

async def code_demo():
    # CodeAgent 适合不支持 Function Calling 的模型
    agent = create_agent(
        "openai:gpt-3.5-turbo",  # 也可用本地模型
        tools=[calculator],
        agent_type="code",
        authorized_imports=["math", "random"]
    )
    
    result = await agent.run("计算 123 * 456")
    print(result.output)

asyncio.run(code_demo())
```

### 6.3 观察执行步骤

```python
async def observe_demo():
    agent = create_agent("openai:gpt-4", tools=[search])
    
    # 通过回调观察每一步
    def on_step(step):
        print(f"[{step.step_type}] {step}")
    
    agent.config.on_step = on_step
    result = await agent.run("搜索天气并总结")
    
    # 或通过 result.steps 查看历史
    for step in result.steps:
        print(f"- {step.step_type}: {getattr(step, 'observations', '')}")

asyncio.run(observe_demo())
```

### 6.4 流式输出

```python
async def stream_demo():
    agent = create_agent("openai:gpt-4")
    
    # 流式执行
    async for chunk in agent.stream("讲个故事"):
        if isinstance(chunk, AgentResult):
            print(f"\n完成: {chunk.output}")
        else:
            print(chunk, end="")

asyncio.run(stream_demo())
```

---

## 7. v3 → v4 变更总结

| 变更项 | v3 (旧) | v4 (新) | 理由 |
|:---|:---|:---|:---|
| **Agent 设计** | 单一 Agent 类 | **Template Method 基类 + 子类** | 支持 ToolCallingAgent + CodeAgent |
| **Memory** | 简单 Message 存储 | **强类型 MemoryStep** | 更好的可观测性和追踪 |
| **Tool 系统** | 仅 @tool 装饰器 | **类 + 装饰器双 API** | 兼顾灵活性和易用性 |
| **代码沙箱** | 无 | **内置 CodeAgent** | 支持任意 LLM |
| **项目结构** | `core/`, `integrations/`, `app/` | **增加 `agents/` 目录** | 分离 Agent 实现 |
| **执行追踪** | 简单回调 | **结构化 MemoryStep** | 生产级可观测性 |
| **流式 API** | AsyncIterator[str] | **保持 + 扩展** | 支持步骤级别流式 |

---

## 8. 关键设计决策

### 8.1 为什么使用 Template Method 而非 Protocol?

**选择**: Template Method (继承) 而非 Protocol (组合)

```python
# Template Method (v4 选择)
class Agent(ABC):
    async def run(self, task):      # 框架流程
        await self._initialize()
        while not done:
            await self._step()      # 子类实现

class ToolCallingAgent(Agent):
    async def _step(self):          # 实现细节
        pass

# 对比: Protocol (另一种选择)
@runtime_checkable
class Agent(Protocol):
    async def run(self, task): ...  # 无默认实现
```

**理由**:
- ✅ **框架复用**: ReAct 循环逻辑在基类实现，子类专注单步逻辑
- ✅ **扩展简单**: 新增 Agent 类型只需实现 `_step()`
- ✅ **统一行为**: 所有 Agent 有相同的记忆管理、错误处理
- ✅ **类型安全**: 强制子类实现必要方法

**权衡**: 牺牲了一些组合灵活性，换取开发效率和一致性。

### 8.2 为什么 Memory 存储 Step 而非 Message?

**选择**: 存储强类型 `MemoryStep` 而非简单 `Message`

```python
# v4: 存储结构化 Step
await memory.add(ActionStep(
    tool_calls=[...],
    observations="...",
    duration=0.5
))

# 对比: 存储简单 Message (v3)
await memory.add(Message(role="tool", content="..."))
```

**理由**:
- ✅ **可观测性**: 每个步骤有类型、时间戳、元数据
- ✅ **调试友好**: 清晰看到 Agent 的思考-行动-观察链
- ✅ **灵活转换**: `get_messages()` 可转换为 LLM 需要的格式
- ✅ **持久化**: 结构化数据更易存储和恢复

### 8.3 为什么同时提供 ToolCallingAgent 和 CodeAgent?

| 维度 | ToolCallingAgent | CodeAgent |
|:---|:---|:---|
| **LLM 要求** | 需要 Function Calling | 任意 LLM 可用 |
| **可控性** | 高（工具白名单） | 中（需要沙箱） |
| **表达能力** | 一次一步 | 多工具组合 |
| **适用场景** | 生产环境 | 复杂计算、本地模型 |

**设计决策**: 同时提供两种，让用户根据场景选择。

---

## 9. 路线图

### Phase 1: 核心 (Week 1)
- [x] 强类型 MemoryStep 设计
- [x] Agent 抽象基类 (Template Method)
- [x] ToolCallingAgent 实现
- [x] ToolRegistry 工具注册表
- [x] @tool 装饰器
- [x] create_agent 工厂函数

### Phase 2: 扩展 (Week 2)
- [ ] CodeAgent 实现
- [ ] Python 沙箱执行器
- [ ] OpenAI 适配
- [ ] Anthropic 适配
- [ ] InMemoryMemory 实现

### Phase 3: 集成 (Week 3)
- [ ] MCP 工具适配
- [ ] 流式输出完善
- [ ] 错误处理 & 重试
- [ ] 测试套件

### Phase 4: 生态 (Week 4+)
- [ ] 更多 Memory 实现 (Chroma, Redis)
- [ ] RAG 支持
- [ ] 可观测性集成
- [ ] 文档 & 示例

---

## 10. 参考架构对比

### LangChain vs smolagents vs Zent v4

| 维度 | LangChain | smolagents | **Zent v4** |
|:---|:---|:---|:---|
| **代码量** | ~100K | ~1,800 | **目标: ~2,500** |
| **架构模式** | Runnable + LCEL | Template Method | **Template Method + Protocols** |
| **Agent 扩展** | LangGraph 复杂 | 类继承简单 | **类继承简单** |
| **Memory** | 多种实现 | 强类型 Step | **强类型 Step** |
| **Tool** | StructuredTool | 类 + @tool 双 API | **类 + @tool 双 API** |
| **CodeAgent** | 无 | 内置 | **内置** |
| **Async** | 完整支持 | Sync only | **Async-first** |
| **MCP** | 社区支持 | 无 | **原生支持** |

---

*版本: 4.0*  
*更新: 2025-03-04*
