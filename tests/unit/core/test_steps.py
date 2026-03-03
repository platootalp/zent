"""Tests for memory steps."""

from datetime import datetime

from zent.core.messages import ToolCall
from zent.core.steps import (
    ActionStep,
    AgentResult,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    StepType,
    SystemPromptStep,
    TaskStep,
)


class TestMemoryStep:
    """Test base MemoryStep class."""

    def test_base_step_has_timestamp(self) -> None:
        """Test base step has timestamp."""
        step = MemoryStep()
        assert isinstance(step.timestamp, datetime)
        assert step.metadata == {}

    def test_step_metadata(self) -> None:
        """Test step metadata."""
        step = MemoryStep()
        step.metadata["custom_key"] = "custom_value"
        assert step.metadata["custom_key"] == "custom_value"


class TestSystemPromptStep:
    """Test SystemPromptStep."""

    def test_system_prompt_creation(self) -> None:
        """Test creating system prompt step."""
        step = SystemPromptStep(system_prompt="You are a helpful assistant")
        assert step.step_type == StepType.SYSTEM
        assert step.system_prompt == "You are a helpful assistant"


class TestTaskStep:
    """Test TaskStep."""

    def test_task_step_creation(self) -> None:
        """Test creating task step."""
        step = TaskStep(task="Calculate 2 + 2")
        assert step.step_type == StepType.TASK
        assert step.task == "Calculate 2 + 2"


class TestPlanningStep:
    """Test PlanningStep."""

    def test_planning_step_creation(self) -> None:
        """Test creating planning step."""
        step = PlanningStep(
            plan="1. Search for data\n2. Analyze results",
            facts="User wants to know about Python",
        )
        assert step.step_type == StepType.PLANNING
        assert "Search" in step.plan
        assert "Python" in step.facts


class TestActionStep:
    """Test ActionStep."""

    def test_action_step_creation(self) -> None:
        """Test creating action step."""
        tool_call = ToolCall(id="call_1", name="search", arguments={"query": "python"})
        step = ActionStep(
            tool_calls=[tool_call],
            observations="Found 10 results",
            duration=1.5,
        )
        assert step.step_type == StepType.ACTION
        assert len(step.tool_calls) == 1
        assert step.observations == "Found 10 results"
        assert step.duration == 1.5
        assert step.error is None

    def test_action_step_with_error(self) -> None:
        """Test action step with error."""
        step = ActionStep(
            tool_calls=[],
            observations="",
            error="Tool not found",
            duration=0.1,
        )
        assert step.error == "Tool not found"


class TestFinalAnswerStep:
    """Test FinalAnswerStep."""

    def test_final_answer_creation(self) -> None:
        """Test creating final answer step."""
        step = FinalAnswerStep(answer="The answer is 4")
        assert step.step_type == StepType.FINAL_ANSWER
        assert step.answer == "The answer is 4"


class TestAgentResult:
    """Test AgentResult."""

    def test_result_creation(self) -> None:
        """Test creating agent result."""
        result = AgentResult(
            output="Final answer",
            steps=[],
            success=True,
            task_id="abc123",
        )
        assert result.output == "Final answer"
        assert result.success is True
        assert result.task_id == "abc123"
        assert result.error is None

    def test_result_with_error(self) -> None:
        """Test result with error."""
        error = ValueError("Something went wrong")
        result = AgentResult(
            output="Error occurred",
            success=False,
            error=error,
        )
        assert result.success is False
        assert result.error == error

    def test_step_count(self) -> None:
        """Test step count property."""
        steps = [
            SystemPromptStep(),
            TaskStep(task="Do something"),
            PlanningStep(plan="Step 1"),
            ActionStep(tool_calls=[], observations="Done"),
            FinalAnswerStep(answer="Result"),
        ]
        result = AgentResult(steps=steps)
        # Should count PlanningStep, ActionStep, FinalAnswerStep = 3
        assert result.step_count == 3

    def test_step_count_empty(self) -> None:
        """Test step count with empty steps."""
        result = AgentResult()
        assert result.step_count == 0

    def test_step_count_only_system_task(self) -> None:
        """Test step count with only system and task steps."""
        steps = [
            SystemPromptStep(),
            TaskStep(task="Do something"),
        ]
        result = AgentResult(steps=steps)
        assert result.step_count == 0
