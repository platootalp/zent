# 轻量级Agent开发框架设计

## 设计定位

**框架名称**: AgentLite / ZentAgent（待确定）

**核心理念**: 
> "只做必要的事，其他交给扩展"

- **极简核心**: 核心代码 < 1000 行
- **零依赖启动**: 仅依赖标准库 + requests
- **渐进增强**: 核心能力必选，扩展能力可选
- **开发者友好**: 清晰的类型提示、直观的API设计

---

## 需求摘要

| 维度 | 决策 |
|:---|:---|
| **技术栈** | Python 3.9+ |
| **使用场景** | 嵌入式库（可集成到其他应用） |
| **目标用户** | 开发者优先 |
| **架构风格** | 模块化、插件化、类型安全 |

### 能力优先级（按实现顺序）

1. ✅ **Phase 1 - 核心闭环**（MVP）
   - LLM Provider 抽象
   - Tool 定义与调用
   - ReAct 循环

2. ✅ **Phase 2 - 记忆增强**
   - 短期记忆（对话上下文）
   - 长期记忆接口（Vector DB 可插拔）

3. ✅ **Phase 3 - 多Agent协作**
   - Agent 通信协议
   - Manager-Worker 模式

4. ✅ **Phase 4 - 可观测性**
   - 事件系统
   - 追踪与调试

5. ⏳ **Phase 5 - 人机协同**（后续迭代）

---

## 架构设计

### 1. 模块结构

```
agentlite/
├── __init__.py              # 公开API导出
├── core/                    # 核心模块（零外部依赖）
│   ├── __init__.py
│   ├── agent.py            # Agent 基类
│   ├── llm.py              # LLM Provider 抽象
│   ├── tool.py             # Tool 定义与管理
│   ├── memory.py           # 短期记忆实现
│   └── react.py            # ReAct 循环实现
├── providers/               # LLM Provider 实现
│   ├── __init__.py
│   ├── openai.py
│   ├── anthropic.py
│   └── base.py             # Provider 基类
├── tools/                   # 内置工具
│   ├── __init__.py
│   ├── http.py             # HTTP 请求工具
│   └── python.py           # Python 代码执行
├── memory/                  # 长期记忆扩展
│   ├── __init__.py
│   ├── base.py
│   └── vector/             # Vector DB 实现
│       ├── chroma.py
│       └── qdrant.py
├── multi_agent/            # 多Agent协作
│   ├── __init__.py
│   ├── team.py             # Agent 团队管理
│   └── patterns.py         # 协作模式
├── observability/          # 可观测性
│   ├── __init__.py
│   ├── events.py           # 事件系统
│   └── tracing.py          # 追踪实现
└── types.py                # 公共类型定义
```

### 2. 核心抽象

#### 2.1 LLM Provider

```python
# agentlite/core/llm.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncIterator


@dataclass
class Message:
    role: str  # "system", "user", "assistant", "tool"
    content: str
    name: Optional[str] = None  # for tool messages
    tool_calls: Optional[List[Dict]] = None


@dataclass
class LLMResponse:
    content: str
    tool_calls: Optional[List[Dict]] = None
    usage: Optional[Dict[str, int]] = None  # token counts


class BaseLLM(ABC):
    """LLM Provider 抽象基类"""
    
    @abstractmethod
    def complete(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict]] = None
    ) -> LLMResponse:
        """同步完成请求"""
        pass
    
    @abstractmethod
    async def acomplete(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict]] = None
    ) -> LLMResponse:
        """异步完成请求"""
        pass
    
    def stream(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict]] = None
    ) -> Iterator[str]:
        """流式输出（可选实现）"""
        raise NotImplementedError("Streaming not supported")
```

#### 2.2 Tool 定义

```python
# agentlite/core/tool.py
from dataclasses import dataclass
from typing import Callable, Dict, Any, Optional
import inspect
import json


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    fn: Callable
    
    def execute(self, **kwargs) -> Any:
        """执行工具"""
        return self.fn(**kwargs)
    
    def to_openai_format(self) -> Dict:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Callable:
    """工具装饰器"""
    def decorator(fn: Callable) -> Tool:
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__ or ""
        
        # 自动从函数签名生成参数 schema
        sig = inspect.signature(fn)
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param_name, param in sig.parameters.items():
            param_info = {"type": "string"}  # 默认字符串
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_info["type"] = "integer"
                elif param.annotation == float:
                    param_info["type"] = "number"
                elif param.annotation == bool:
                    param_info["type"] = "boolean"
            
            parameters["properties"][param_name] = param_info
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        return Tool(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            fn=fn
        )
    return decorator
```

#### 2.3 记忆系统

```python
# agentlite/core/memory.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryEntry:
    """记忆条目"""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShortTermMemory:
    """短期记忆 - 维护对话上下文"""
    
    def __init__(self, max_messages: int = 20):
        self.messages: List[MemoryEntry] = []
        self.max_messages = max_messages
    
    def add(self, role: str, content: str, **metadata) -> None:
        """添加记忆"""
        entry = MemoryEntry(role=role, content=content, metadata=metadata)
        self.messages.append(entry)
        
        # 滑动窗口：保留最新消息
        if len(self.messages) > self.max_messages:
            # 保留系统消息，移除最早的非系统消息
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"]
            other_msgs = other_msgs[-(self.max_messages - len(system_msgs)):]
            self.messages = system_msgs + other_msgs
    
    def get_context(self) -> List[Dict[str, str]]:
        """获取上下文（用于 LLM 调用）"""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]
    
    def clear(self) -> None:
        """清空记忆"""
        self.messages = []
    
    def summarize(self) -> str:
        """生成对话摘要（用于长期记忆）"""
        # Phase 2 实现：调用 LLM 生成摘要
        pass


class LongTermMemory(ABC):
    """长期记忆抽象 - 需要外部 Vector DB"""
    
    @abstractmethod
    def store(self, content: str, metadata: Dict[str, Any]) -> None:
        """存储记忆"""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """检索相关记忆"""
        pass
```

#### 2.4 ReAct Agent

```python
# agentlite/core/react.py
from typing import List, Dict, Any, Optional
import json
import re

from .llm import BaseLLM, Message
from .tool import Tool
from .memory import ShortTermMemory


class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent
    
    核心循环：
    Thought → Action → Observation → ... → Final Answer
    """
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that can use tools to accomplish tasks.

When you need to use a tool, respond in this format:
Thought: [Your reasoning about what to do]
Action: [Tool name]
Action Input: [JSON object with parameters]

When you have the final answer, respond:
Thought: [Your final reasoning]
Final Answer: [Your answer]

Available tools:
{tool_descriptions}
"""
    
    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        memory: Optional[ShortTermMemory] = None
    ):
        self.llm = llm
        self.tools = {t.name: t for t in (tools or [])}
        self.max_iterations = max_iterations
        self.memory = memory or ShortTermMemory()
        
        # 设置系统提示词
        tool_desc = self._format_tool_descriptions()
        self.system_prompt = (system_prompt or self.DEFAULT_SYSTEM_PROMPT).format(
            tool_descriptions=tool_desc
        )
        
        # 初始化记忆
        if not self.memory.messages:
            self.memory.add("system", self.system_prompt)
    
    def _format_tool_descriptions(self) -> str:
        """格式化工具描述"""
        descriptions = []
        for name, tool in self.tools.items():
            desc = f"- {name}: {tool.description}"
            if tool.parameters.get("properties"):
                params = ", ".join(tool.parameters["properties"].keys())
                desc += f" Parameters: {params}"
            descriptions.append(desc)
        return "\n".join(descriptions) if descriptions else "No tools available."
    
    def run(self, user_input: str) -> str:
        """运行 Agent"""
        # 添加用户输入
        self.memory.add("user", user_input)
        
        for iteration in range(self.max_iterations):
            # 1. 获取 LLM 响应
            context = self.memory.get_context()
            messages = [Message(**m) for m in context]
            response = self.llm.complete(messages)
            
            content = response.content
            
            # 2. 解析响应
            parsed = self._parse_response(content)
            
            if parsed["type"] == "final":
                # 完成任务
                self.memory.add("assistant", parsed["answer"])
                return parsed["answer"]
            
            elif parsed["type"] == "action":
                # 执行工具
                tool_name = parsed["tool"]
                tool_input = parsed["input"]
                
                if tool_name not in self.tools:
                    observation = f"Error: Tool '{tool_name}' not found."
                else:
                    try:
                        result = self.tools[tool_name].execute(**tool_input)
                        observation = str(result)
                    except Exception as e:
                        observation = f"Error: {str(e)}"
                
                # 记录观察和工具调用
                self.memory.add("assistant", content)
                self.memory.add("user", f"Observation: {observation}", is_observation=True)
            
            else:
                # 无法解析，返回错误
                return f"Error: Unable to parse agent response. Raw: {content}"
        
        return "Error: Maximum iterations reached without final answer."
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """解析 Agent 响应"""
        # 检查 Final Answer
        if "Final Answer:" in content:
            match = re.search(r"Final Answer:\s*(.+)", content, re.DOTALL)
            if match:
                return {
                    "type": "final",
                    "answer": match.group(1).strip()
                }
        
        # 检查 Action
        action_match = re.search(
            r"Thought:\s*(.+?)\s*Action:\s*(\w+)\s*Action Input:\s*(\{.*?\})",
            content,
            re.DOTALL
        )
        if action_match:
            try:
                action_input = json.loads(action_match.group(3))
                return {
                    "type": "action",
                    "thought": action_match.group(1).strip(),
                    "tool": action_match.group(2).strip(),
                    "input": action_input
                }
            except json.JSONDecodeError:
                pass
        
        return {"type": "unknown", "content": content}
```

---

## 使用示例

### 基础示例

```python
from agentlite import ReActAgent, tool
from agentlite.providers import OpenAIProvider

# 1. 定义工具
@tool(description="Search for weather information")
def get_weather(location: str, date: str) -> str:
    """获取天气信息"""
    # 实际实现会调用天气 API
    return f"Weather in {location} on {date}: Sunny, 25°C"

@tool(description="Perform mathematical calculations")
def calculate(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

# 2. 创建 Agent
llm = OpenAIProvider(api_key="your-key", model="gpt-4")
agent = ReActAgent(
    llm=llm,
    tools=[get_weather, calculate],
    system_prompt="You are a helpful assistant."
)

# 3. 运行
result = agent.run("What's the weather in Beijing today and what's 25 * 4?")
print(result)
```

### 使用 Function Calling 的增强版本

```python
from agentlite import Agent, tool
from agentlite.providers import OpenAIProvider

# 现代 LLM 支持原生 Function Calling，更高效
class FunctionCallingAgent:
    """使用 LLM 原生 Function Calling 的 Agent"""
    
    def __init__(self, llm: BaseLLM, tools: List[Tool]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.memory = ShortTermMemory()
    
    def run(self, user_input: str) -> str:
        self.memory.add("user", user_input)
        
        # 转换为 OpenAI 格式的工具定义
        openai_tools = [t.to_openai_format() for t in self.tools.values()]
        
        for _ in range(self.max_iterations):
            messages = [Message(**m) for m in self.memory.get_context()]
            response = self.llm.complete(messages, tools=openai_tools)
            
            if response.tool_calls:
                # 执行工具调用
                self.memory.add("assistant", response.content or "", 
                              tool_calls=response.tool_calls)
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = json.loads(tool_call["function"]["arguments"])
                    
                    if tool_name in self.tools:
                        result = self.tools[tool_name].execute(**tool_args)
                        self.memory.add("tool", str(result), name=tool_name)
            else:
                # 最终答案
                self.memory.add("assistant", response.content)
                return response.content
        
        return "Max iterations reached"
```

### 多 Agent 协作示例

```python
from agentlite.multi_agent import Team, SequentialPattern

# 创建专业 Agent
researcher = ReActAgent(llm=llm, tools=[search_tool], 
                       system_prompt="You are a research specialist.")
writer = ReActAgent(llm=llm, tools=[],
                   system_prompt="You are a technical writer.")
reviewer = ReActAgent(llm=llm, tools=[],
                     system_prompt="You are a content reviewer.")

# 组建团队
team = Team(
    agents={"researcher": researcher, "writer": writer, "reviewer": reviewer},
    pattern=SequentialPattern()  # 顺序执行: researcher → writer → reviewer
)

# 执行任务
result = team.run("Research the latest AI trends and write a blog post")
```

---

## 扩展接口设计

### 1. Provider 扩展

```python
# 自定义 LLM Provider
from agentlite.core import BaseLLM, Message, LLMResponse

class ClaudeProvider(BaseLLM):
    def __init__(self, api_key: str, model: str = "claude-3-opus"):
        self.api_key = api_key
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def complete(self, messages: List[Message], tools=None) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            max_tokens=4096
        )
        return LLMResponse(
            content=response.content[0].text,
            usage={"input": response.usage.input_tokens, 
                   "output": response.usage.output_tokens}
        )
```

### 2. 长期记忆扩展

```python
# 使用 ChromaDB
from agentlite.memory.vector import ChromaMemory

memory = ChromaMemory(
    collection_name="my_agent",
    persist_directory="./memory"
)

# 集成到 Agent
agent = ReActAgent(
    llm=llm,
    tools=tools,
    long_term_memory=memory  # 自动在需要时检索相关记忆
)
```

### 3. 事件/追踪扩展

```python
from agentlite.observability import EventLogger

# 监听所有事件
logger = EventLogger()
logger.on("llm.request", lambda e: print(f"LLM Call: {e['model']}"))
logger.on("tool.execute", lambda e: print(f"Tool: {e['tool']} -> {e['result']}"))
logger.on("agent.complete", lambda e: print(f"Done: {e['iterations']} iterations"))

# 附加到 Agent
agent = ReActAgent(llm=llm, tools=tools, event_logger=logger)
```

---

## 技术决策说明

### 为什么选择 ReAct 作为默认模式？

| 方案 | 优点 | 缺点 | 选择理由 |
|:---|:---|:---|:---|
| **ReAct** (推理-行动循环) | 可解释性强、通用性好、无需特殊模型支持 | 需要多轮交互、Token 消耗较多 | 平衡了通用性和实现复杂度 |
| **Function Calling** | 一轮完成、效率高 | 依赖模型支持、可解释性稍弱 | 作为优化版本提供 |
| **Plan-and-Execute** | 提前规划、步骤清晰 | 计划可能偏离、需要重规划机制 | 复杂任务场景使用 |

**决策**: 以 ReAct 为核心，同时支持 Function Calling 优化。

### 同步 vs 异步架构

```python
# 同时支持两种模式
class BaseAgent:
    def run(self, query: str) -> str:
        """同步入口"""
        return asyncio.get_event_loop().run_until_complete(self.arun(query))
    
    async def arun(self, query: str) -> str:
        """异步入口（实际实现）"""
        # 异步实现
        pass
```

### 依赖策略

| 层级 | 依赖 | 说明 |
|:---|:---|:---|
| **Core** | 仅标准库 | typing, dataclasses, abc, json, re |
| **Providers** | 按需安装 | openai, anthropic, etc. |
| **Memory** | 按需安装 | chromadb, qdrant-client, etc. |
| **Observability** | 可选 | rich (for pretty printing), loguru |

**安装策略**:
```bash
# 最小安装
pip install agentlite

# 带 OpenAI 支持
pip install agentlite[openai]

# 完整功能
pip install agentlite[all]
```

---

## 实现路线图

### Phase 1: MVP（2-3周）

- [ ] Core 模块（Agent, LLM抽象, Tool, ReAct循环）
- [ ] OpenAI Provider 实现
- [ ] 基础测试覆盖
- [ ] 使用文档

### Phase 2: 记忆系统（1-2周）

- [ ] 短期记忆完善
- [ ] 长期记忆接口
- [ ] ChromaDB 实现

### Phase 3: 多Agent（2周）

- [ ] Agent 通信协议
- [ ] Sequential/Hierarchical 模式
- [ ] Team 管理

### Phase 4: 可观测性（1-2周）

- [ ] 事件系统
- [ ] 基础追踪
- [ ] 调试工具

### Phase 5: 生态（持续）

- [ ] 更多 Provider（Anthropic, Gemini, Local models）
- [ ] 内置工具集
- [ ] 示例应用

---

## API 设计原则总结

1. **渐进式暴露**: 基础功能开箱即用，高级功能显式导入
2. **类型安全**: 全程类型提示，IDE 友好
3. **装饰器优先**: `@tool`, `@agent` 等装饰器降低样板代码
4. **可组合性**: Agent, Tool, Memory 可自由组合
5. **零魔法**: 避免隐式全局状态，显式依赖注入
