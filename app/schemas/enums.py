from enum import StrEnum

class AgentRole(StrEnum):
    DIRECTOR = "director"
    SPECIALIST = "specialist"
    RUNNER = "runner"


class AgentStatus(StrEnum):
    IDLE = "idle"
    THINKING = "thinking"
    RUNNING_TOOL = "running_tool"
    WAITING_FOR_REVIEW = "waiting_for_review"
    DONE = "done"
    ERROR = "error"


class TaskComplexity(StrEnum):
    SIMPLE = "simple"
    NORMAL = "normal"
    HARD = "hard"


class TaskDomain(StrEnum):
    LITERATURE = "literature"
    INTERNAL_DOCS = "internal_docs"
    BIOINFORMATICS = "bioinformatics"
    CHEMISTRY = "chemistry"
    DATA_ANALYSIS = "data_analysis"
    REGULATORY = "regulatory"
    CLINICAL = "clinical"
    LAB_DATA = "lab_data"
    GENERAL = "general"


class TaskStatus(StrEnum):
    PENDING = "pending"
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    WAITING_FOR_REVIEW = "waiting_for_review"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED_ON_HUMAN = "blocked_on_human"
    CANCELLED = "cancelled"


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class EventType(StrEnum):
    SESSION_STARTED = "session.started"
    TASK_PLANNED = "task.planned"
    TASK_ASSIGNED = "task.assigned"
    AGENT_STARTED = "agent.started"
    AGENT_STATUS = "agent.status"
    AGENT_TOKEN = "agent.token"
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    REVIEW_APPROVED = "review.approved"
    REVIEW_REJECTED = "review.rejected"
    APPROVAL_REQUIRED = "approval.required"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    USAGE_UPDATED = "usage.updated"
