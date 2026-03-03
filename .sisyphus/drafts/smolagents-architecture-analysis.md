# smolagents 架构设计方案解析

> Hugging Face 出品的轻量级 Agent 框架 (~1,800 行核心代码)

---

## 一、整体架构

### 1.1 代码组织结构

```
smolagents/src/smolagents/
├── __init__.py              # 公开 API 导出 (~50 行)
├── agents.py               # 核心 Agent 类 (~1,814 行)
├── tools.py                # Tool 抽象与实现 (~1,200 行)
├── models.py               # LLM Provider 抽象 (~2,102 行)
├── memory.py               # 记忆管理 (~316 行)
├── local_python_executor.py # Python 代码沙箱执行 (~1,403 行)
├── default_tools.py        # 内置工具集 (~661 行)
├── monitoring.py           # 可观测性 (~273 行)
├── utils.py                # 工具函数 (~606 行)
└── prompts/                # 提示词模板
    ├── toolcalling_agent.yaml
    ├── code_agent.yaml
    └── structured_code_agent.yaml
```

**设计哲学**: 模块化但不过度拆分，每个文件职责明确，便于理解和维护。

### 1.2 模块依赖关系

```
                    ┌─────────────┐
                    │   Agents    │
                    │  (核心协调)  │
                    └──────┬──────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │  Tools   │    │  Models  │    │  Memory  │
    │ (工具执行)│    │ (LLM调用)│    │ (状态管理)│
    └──────────┘    └──────────┘    └──────────┘
          │                │                │
          └────────────────┼────────────────┘
                           │
                    ┌──────┴──────┐
                    │   Utils     │
                    │ (通用工具)   │
                    └─────────────┘
```

---

## 二、核心抽象详解

### 2.1 Agent 继承体系

```python
# agents.py

class MultiStepAgent(ABC):
    """所有 Agent 的抽象基类 - 实现 ReAct 框架"""
    # lines 268-891
    
    def __init__(self, tools, model, ...):
        # 初始化：工具、模型、记忆、日志
        
    def _run_stream(self, task: str, ...):
        # 核心 ReAct 循环实现
        # lines 540-611
        
    @abstractmethod
    def _step_stream(self, ...):
        # 子类必须实现的单步逻辑
        
class ToolCallingAgent(MultiStepAgent):
    """使用 LLM 原生 Function Calling"""
    # lines 1215-1503
    
    def _step_stream(self, ...):
        # 生成 tool_calls JSON，调用工具
        # lines 1276-1359
        
class CodeAgent(MultiStepAgent):
    """生成并执行 Python 代码"""
    # lines 1505-1805
    
    def _step_stream(self, ...):
        # 生成代码字符串，AST 解析执行
        # lines 1639-1765
```

**设计模式**: **模板方法模式 (Template Method)**
- 基类定义 `_run_stream` 框架流程
- 子类实现 `_step_stream` 具体行为
- 统一记忆管理、日志、监控

### 2.2 Tool 系统设计

```python
# tools.py

class BaseTool(ABC):
    """最小工具接口"""
    # lines 98-103
    name: str
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class Tool(BaseTool):
    """完整工具实现 - 支持验证、序列化、Hub集成"""
    # lines 106-598
    
    def __init__(self, name, description, inputs, output_type, fn):
        # 支持自动从函数签名推断 schema
        
    def __call__(self, *args, **kwargs):
        # 自动验证输入、执行函数、验证输出
        # lines 231-249
        
    def to_dict(self):
        # 序列化为 Hub 格式
        
    @classmethod
    def from_hub(cls, repo_id):
        # 从 HuggingFace Hub 加载工具
        # lines 420-498

# @tool 装饰器 - 函数转 Tool 类
def tool(tool_function):
    """将函数动态转换为 Tool 子类"""
    # lines 1061-1168
    # 使用 type() 动态创建类，继承 Tool
```

**设计亮点**:
- 双重 API: 类继承(`class MyTool(Tool)`) + 装饰器(`@tool`)
- 自动 schema 推断: 从函数签名 + 类型注解生成 JSON Schema
- 内置验证: 输入/输出自动校验，错误友好提示

### 2.3 Model 抽象层

```python
# models.py

class Model(ABC):
    """统一 LLM Provider 接口"""
    # lines 452-631
    
    def __call__(self, messages: List[Dict], ...):
        # 同步调用入口
        
    def generate_stream(self, messages, ...):
        # 流式生成接口
        # 返回 Generator[ChatMessageStreamDelta]
        
    def _prepare_completion_kwargs(self, messages, tools):
        # 统一处理工具格式转换
        # lines 502-551

# 8+ 种 Provider 实现
class OpenAIModel(Model):
    # lines 1699-1850
    
class AzureOpenAIModel(Model):
    # lines 1852-1946
    
class AnthropicModel(Model):
    # lines 1948-2050
    
class TransformersModel(Model):
    # 本地 transformers 模型
    # lines 971-1100
    
# ... 更多实现
```

**设计模式**: **策略模式 (Strategy)**
- 统一接口，不同实现
- 通过 `MODEL_REGISTRY` 支持安全反序列化

### 2.4 Memory 记忆系统

```python
# memory.py

@dataclass
class MemoryStep:
    """记忆步骤基类"""
    # lines 51-61

@dataclass  
class ActionStep(MemoryStep):
    """工具调用/代码执行步骤"""
    # lines 107-183
    tool_calls: List[ToolCall]
    observations: str
    timing: Timing

@dataclass
class PlanningStep(MemoryStep):
    """规划步骤 - 阶段性重新规划"""
    # lines 63-88
    plan: str

@dataclass
class FinalAnswerStep(MemoryStep):
    """最终答案步骤"""
    # lines 90-105
    final_answer: str

class AgentMemory:
    """Agent 记忆管理器"""
    # lines 214-277
    
    def __init__(self, system_prompt):
        self.system_prompt = SystemPromptStep(system_prompt)
        self.steps: List[MemoryStep] = []
        
    def get_succinct_steps(self):
        # 压缩历史为消息列表，用于 LLM 上下文
```

**设计亮点**:
- 强类型步骤: 每个步骤类型有明确的字段和语义
- 可序列化: 支持保存/恢复会话状态
- 回调机制: `CallbackRegistry` 支持事件监听

---

## 三、ReAct 循环实现

### 3.1 主循环流程

```python
# agents.py MultiStepAgent._run_stream
# lines 540-611

def _run_stream(self, task: str, max_steps: int = 10):
    """
    ReAct 循环核心实现
    
    流程:
    1. 初始化任务
    2. while 循环直到完成或达到最大步数
    3. 每步：调用 _step_stream 获取下一步动作
    4. 根据返回结果判断：继续 / 完成 / 错误
    """
    
    # 1. 创建初始任务步骤
    self.memory.steps.append(TaskStep(task=task))
    
    # 2. ReAct 循环
    while not self.completed and self.step_number <= max_steps:
        
        # 2.1 触发规划（如果配置了规划间隔）
        if self.planning_interval and step % self.planning_interval == 0:
            yield from self._planning_step()
        
        # 2.2 执行单步（由子类实现）
        step_output = yield from self._step_stream()
        
        # 2.3 处理步结果
        if step_output.output:
            # 代码 Agent 返回代码块
            self.memory.steps.append(ActionStep(
                tool_calls=[ToolCall(code=step_output.output)],
                observations=execute_code(step_output.output)
            ))
        elif step_output.tool_calls:
            # ToolCalling Agent 返回 tool_calls
            self.memory.steps.append(ActionStep(
                tool_calls=step_output.tool_calls,
                observations=execute_tools(step_output.tool_calls)
            ))
```

### 3.2 ToolCallingAgent 单步实现

```python
# agents.py ToolCallingAgent._step_stream
# lines 1276-1359

def _step_stream(self, ...):
    """
    使用 LLM 原生 Function Calling
    
    流程:
    1. 构建消息历史
    2. 调用 LLM 获取 tool_calls
    3. 解析并执行工具
    4. 返回观察结果
    """
    
    # 1. 构建消息
    messages = self.memory.get_succinct_steps()
    
    # 2. 调用 LLM
    model_output = self.model(messages, tools=self.tools)
    
    if model_output.tool_calls:
        # 3. 执行工具
        observations = []
        for tool_call in model_output.tool_calls:
            tool = self.tools[tool_call.name]
            result = tool(**tool_call.arguments)
            observations.append(result)
        
        # 4. 返回 ActionStep 数据
        yield StepOutput(
            tool_calls=model_output.tool_calls,
            observations="\n".join(observations)
        )
    else:
        # 直接得到最终答案
        yield StepOutput(final_answer=model_output.content)
```

### 3.3 CodeAgent 单步实现

```python
# agents.py CodeAgent._step_stream
# lines 1639-1765

def _step_stream(self, ...):
    """
    生成并执行 Python 代码
    
    流程:
    1. 构建包含工具描述的提示词
    2. 调用 LLM 生成代码
    3. 解析代码块
    4. AST 安全执行
    5. 返回结果
    """
    
    # 1. 构建消息（包含代码示例）
    messages = self.write_message_to_user(prompt=self.prompt_template)
    
    # 2. 生成代码
    model_output = self.model(messages)
    
    # 3. 解析代码块
    code = parse_code_blob(model_output.content)
    
    # 4. 安全执行
    result = self.python_executor(code)
    
    # 5. 返回结果
    if result.output:
        yield StepOutput(output=result.output)
    elif result.error:
        yield StepOutput(error=result.error)
```

### 3.4 Python 沙箱执行器

```python
# local_python_executor.py

def evaluate_ast(expression, state, tools):
    """
    AST 节点求值 - 安全沙箱核心
    
    特点:
    - 白名单机制：只允许安全操作
    - 危险函数拦截：os.system, eval, exec 等
    - 自定义导入检查：只允许授权模块
    """
    
    if isinstance(expression, ast.Import):
        # 检查导入是否授权
        if not is_authorized_import(expression):
            raise InterpreterError(f"Unauthorized import")
    
    elif isinstance(expression, ast.Call):
        # 检查函数调用是否安全
        if is_dangerous_function(expression.func):
            raise InterpreterError(f"Dangerous function call")
        
        # 执行函数调用
        return function(*args, **kwargs)

class LocalPythonExecutor:
    """本地 Python 执行器"""
    # lines 1598-1620
    
    def __call__(self, code: str):
        # 1. 解析 AST
        tree = ast.parse(code)
        
        # 2. 逐节点求值
        result = evaluate_ast(tree, self.state, self.tools)
        
        return result
```

---

## 四、设计模式分析

### 4.1 使用的模式

| 模式 | 应用位置 | 目的 |
|:---|:---|:---|
| **模板方法** | `MultiStepAgent._run_stream` + 子类 `_step_stream` | 统一 ReAct 框架，允许不同执行策略 |
| **策略** | `Model` 接口 + 多个 Provider 实现 | 统一不同 LLM 的调用方式 |
| **工厂** | `Tool.from_hub`, `Tool.from_code` | 多种方式创建工具 |
| **注册表** | `AGENT_REGISTRY`, `MODEL_REGISTRY` | 安全反序列化 |
| **观察者** | `CallbackRegistry` | 事件监听和扩展 |
| **装饰器** | `@tool` | 函数转类，降低使用门槛 |

### 4.2 扩展点设计

```python
# 1. 自定义 Agent
class MyAgent(MultiStepAgent):
    def _step_stream(self, ...):
        # 自定义单步逻辑
        pass

# 2. 自定义 Model
class MyModel(Model):
    def generate_stream(self, messages, ...):
        # 接入新 LLM 提供商
        pass

# 3. 自定义 Tool
@tool
def my_tool(input: str) -> str:
    """自定义工具"""
    return result

# 4. 自定义 Memory
class MyMemory(AgentMemory):
    def get_succinct_steps(self):
        # 自定义历史压缩策略
        pass

# 5. 事件监听
from smolagents.memory import CallbackRegistry

@CallbackRegistry.register(ActionStep)
def on_action(step):
    print(f"Tool executed: {step}")
```

---

## 五、设计决策与权衡

### 5.1 Class-based vs Function-based

**选择**: **Class-based** (面向对象)

```python
# 类方式（smolagents 选择）
class MyTool(Tool):
    def forward(self, input):
        return process(input)

# vs 函数方式（另一种选择）
@tool
def my_tool(input: str) -> str:
    return process(input)
```

**理由**:
- ✅ 状态管理: 工具可以有内部状态 (Tool 类可保存属性)
- ✅ 序列化: Hub 分享需要类的元数据
- ✅ 验证: 统一的输入/输出验证逻辑
- ✅ 继承: 可以创建工具基类（如 APIBaseTool）

**妥协**: 同时提供 `@tool` 装饰器，让简单工具可以用函数方式定义

### 5.2 Sync vs Async

**选择**: **Sync为主，Generator实现流式**

```python
# smolagents 方式
for chunk in agent.run_stream(task):
    print(chunk)  # 实时输出

# vs 纯 Async 方式（某些框架选择）
async for chunk in agent.arun(task):
    print(chunk)
```

**理由**:
- ✅ 简单性: Python 用户更熟悉同步代码
- ✅ 流式: 用 Generator 实现，不需要 async 复杂性
- ✅ 调试: 同步代码更容易调试
- ⚠️ 性能: 大量并发时需要自己管理线程

**权衡**: 不提供原生 async，但可以通过 `asyncio.run_in_executor` 包装

### 5.3 CodeAgent vs ToolCallingAgent

| 维度 | ToolCallingAgent | CodeAgent |
|:---|:---|:---|
| **LLM 要求** | 需要支持 Function Calling | 任意 LLM 可用 |
| **表达能力** | 一次一步，串行工具 | 多工具组合，代码逻辑 |
| **安全性** | 受控（只有注册工具可调用） | 需要沙箱（AST 执行器） |
| **调试性** | 好（每步清晰） | 差（需要看代码） |
| **适用场景** | 通用任务 | 复杂数据/计算任务 |

**设计决策**: 同时提供两种，让用户根据 LLM 能力和任务复杂度选择

### 5.4 轻量化的取舍

**包含** (必要功能):
- ✅ ReAct 循环框架
- ✅ Tool 抽象 + 装饰器
- ✅ Multi-LLM Provider 支持
- ✅ Memory 管理
- ✅ Python 代码沙箱
- ✅ 基础日志/监控

**排除** (保持轻量):
- ❌ RAG/Vector DB (建议外部集成)
- ❌ 多 Agent 编排 (建议上层封装)
- ❌ 持久化存储 (提供接口，不内置)
- ❌ Web UI (纯代码库)
- ❌ 复杂工作流引擎

---

## 六、关键代码参考

### 6.1 核心类定义

```python
# MultiStepAgent 初始化
# agents.py lines 294-353
class MultiStepAgent:
    def __init__(
        self,
        tools: List[Tool],
        model: Callable,
        system_prompt: Optional[str] = None,
        planning_interval: Optional[int] = None,
        max_steps: int = 6,
        tool_parser: Optional[Callable] = None,
        add_base_tools: bool = False,
        verbosity: int = 0,
        grammar: Optional[Dict] = None,
        managed_agents: Optional[List] = None,
        step_callbacks: Optional[List[Callable]] = None,
    ):

# Tool 核心方法
# tools.py lines 231-249
def __call__(self, *args, **kwargs):
    """执行工具，自动验证输入输出"""
    # 1. 验证输入
    # 2. 调用 forward
    # 3. 验证输出
    # 4. 返回结果

# Model 统一接口
# models.py lines 502-551
def _prepare_completion_kwargs(self, messages, tools):
    """统一处理工具格式，适配不同 Provider"""
    # OpenAI 格式 → Provider 特定格式
```

### 6.2 ReAct 循环核心

```python
# 主循环
# agents.py lines 540-611
def _run_stream(self, task, max_steps=10):
    # 初始化
    self.memory.steps.append(TaskStep(task=task))
    
    # ReAct 循环
    while not self.completed and self.step_number <= max_steps:
        # 规划（可选）
        if self.planning_interval and step % self.planning_interval == 0:
            yield from self._planning_step()
        
        # 执行单步
        step_output = yield from self._step_stream()
        
        # 处理结果
        self.process_step_output(step_output)
```

### 6.3 安全注册表

```python
# agents.py lines 1811-1814
AGENT_REGISTRY = {
    "ToolCallingAgent": ToolCallingAgent,
    "CodeAgent": CodeAgent,
}

# models.py lines 2070-2080
MODEL_REGISTRY = {
    "VLLMModel": VLLMModel,
    "OpenAIModel": OpenAIModel,
    # ... 8+ providers
}
```

---

## 七、学习要点

### 7.1 值得借鉴的设计

1. **双重 API 设计**: 类 + 装饰器，兼顾灵活性和易用性
2. **强类型 Memory**: 不同步骤类型有明确语义，便于追踪和调试
3. **沙箱执行**: AST 解析 + 白名单，安全执行用户代码
4. **Provider 统一**: 策略模式封装不同 LLM 的差异
5. **模板方法**: 基类定框架，子类填实现，保持扩展性

### 7.2 可改进之处

1. **缺乏原生 async**: 高并发场景下性能受限
2. **Memory 压缩策略简单**: 长对话时 token 消耗大
3. **缺少内置 RAG**: 需要外部集成
4. **多 Agent 支持弱**: 建议上层框架补充

### 7.3 适用于自己的框架

- ✅ 模板方法模式定义 Agent 基类
- ✅ 策略模式统一 LLM Provider
- ✅ @tool 装饰器简化工具定义
- ✅ 强类型 Memory 步骤
- ✅ AST 沙箱执行代码（如需要）
- ⚠️ 可考虑增加原生 async 支持
- ⚠️ 可考虑增加 RAG 接口（optional）

---

## 参考链接

- 源码: `/tmp/smolagents/`
- 文档: https://huggingface.co/docs/smolagents
- GitHub: https://github.com/huggingface/smolagents
