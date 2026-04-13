from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# -----------------------------
# Allowed values / literals
# -----------------------------

ActionType = Literal[
    "classify_ticket",
    "search_kb",
    "check_account",
    "reply_customer",
    "escalate_ticket",
    "resolve_ticket",
]

TicketCategory = Literal[
    "billing",
    "account_access",
    "feature_access",
    "technical_issue",
]

UrgencyLevel = Literal["low", "medium", "high"]

TicketStatus = Literal["open", "in_progress", "resolved", "escalated"]

DifficultyLevel = Literal["easy", "medium", "hard"]


# -----------------------------
# Interaction history item
# -----------------------------

class InteractionEntry(BaseModel):
    step_number: int
    action_type: str
    action_summary: str
    result: str


# -----------------------------
# Observation model
# -----------------------------

class Observation(BaseModel):
    ticket_id: str
    task_id: str
    customer_message: str
    customer_profile: Dict[str, str | int | bool | float]
    interaction_history: List[InteractionEntry] = Field(default_factory=list)
    visible_tools: List[ActionType]
    current_status: TicketStatus
    remaining_steps: int
    last_tool_result: Optional[str] = None


# -----------------------------
# Action model
# -----------------------------

class Action(BaseModel):
    action_type: ActionType

    # Used in classify_ticket
    category: Optional[TicketCategory] = None
    urgency: Optional[UrgencyLevel] = None

    # Used in search_kb
    query: Optional[str] = None

    # Used in check_account
    account_field: Optional[str] = None

    # Used in reply_customer
    message: Optional[str] = None

    # Used in escalate_ticket / resolve_ticket
    reason: Optional[str] = None
    resolution_note: Optional[str] = None

    @field_validator("query", "account_field", "message", "reason", "resolution_note")
    @classmethod
    def strip_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        cleaned = value.strip()
        return cleaned if cleaned else None


# -----------------------------
# Reward model
# -----------------------------

class Reward(BaseModel):
    score: float = Field(ge=-1.0, le=1.0)
    reason: str


# -----------------------------
# Task definition model
# -----------------------------

class TaskDefinition(BaseModel):
    task_id: str
    difficulty: DifficultyLevel
    title: str
    description: str

    ticket_id: str
    customer_message: str
    customer_profile: Dict[str, str | int | bool | float]

    true_category: TicketCategory
    true_urgency: UrgencyLevel

    relevant_kb_article: Optional[str] = None
    required_account_fields: List[str] = Field(default_factory=list)
    required_reply_keywords: List[str] = Field(default_factory=list)

    escalation_allowed: bool = False
    expected_final_action: Literal["resolve_ticket", "escalate_ticket"]

    max_steps: int = 8


# -----------------------------
# Full internal environment state
# -----------------------------

class SupportEnvState(BaseModel):
    task_id: str
    difficulty: DifficultyLevel

    ticket_id: str
    customer_message: str
    customer_profile: Dict[str, str | int | bool | float]

    true_category: TicketCategory
    true_urgency: UrgencyLevel
    relevant_kb_article: Optional[str] = None
    required_account_fields: List[str] = Field(default_factory=list)
    required_reply_keywords: List[str] = Field(default_factory=list)
    escalation_allowed: bool = False
    expected_final_action: Literal["resolve_ticket", "escalate_ticket"] = "resolve_ticket"

    current_status: TicketStatus = "open"
    step_count: int = 0
    max_steps: int = 8
    done: bool = False

    classification_correct: bool = False
    urgency_correct: bool = False
    resolved_correctly: bool = False
    escalated_correctly: bool = False

    checked_account_fields: List[str] = Field(default_factory=list)
    retrieved_kb_articles: List[str] = Field(default_factory=list)
    interaction_history: List[InteractionEntry] = Field(default_factory=list)

    last_tool_result: Optional[str] = None
    last_reply: Optional[str] = None

    invalid_action_count: int = 0
    repeated_action_count: int = 0
    wasted_step_count: int = 0

    action_history: List[Action] = Field(default_factory=list)