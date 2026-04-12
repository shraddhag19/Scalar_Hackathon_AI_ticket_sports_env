from __future__ import annotations

from typing import Dict, List

from envs.models import TaskDefinition


TASKS: List[TaskDefinition] = [
    TaskDefinition(
        task_id="task_easy_billing_triage",
        difficulty="easy",
        title="Billing Triage for Duplicate Charge",
        description=(
            "The agent must classify a duplicate charge complaint, assign the correct urgency, "
            "and avoid an unnecessary escalation."
        ),
        ticket_id="TCK-1001",
        customer_message=(
            "Hi support, I think I was charged twice for my subscription this month. "
            "Can you please check what happened?"
        ),
        customer_profile={
            "plan": "Pro",
            "region": "IN",
            "account_age_days": 210,
            "last_payment_status": "successful",
            "invoice_available": True,
            "feature_flag_enabled": True,
            "workspace_role": "member",
            "seat_permission_granted": True,
        },
        true_category="billing",
        true_urgency="medium",
        relevant_kb_article="double_charge_review_policy",
        required_account_fields=["last_payment_status"],
        required_reply_keywords=["billing", "review"],
        escalation_allowed=False,
        expected_final_action="resolve_ticket",
        max_steps=5,
    ),
    TaskDefinition(
        task_id="task_medium_missing_invoice",
        difficulty="medium",
        title="Missing Invoice After Successful Payment",
        description=(
            "The agent must inspect billing status, retrieve the correct KB article, respond "
            "with the invoice timing guidance, and resolve the ticket."
        ),
        ticket_id="TCK-1002",
        customer_message=(
            "Hello, my payment went through yesterday but I still can't see my invoice in the "
            "billing dashboard. Could you help?"
        ),
        customer_profile={
            "plan": "Business",
            "region": "US",
            "account_age_days": 420,
            "last_payment_status": "successful",
            "invoice_available": False,
            "feature_flag_enabled": True,
            "workspace_role": "owner",
            "seat_permission_granted": True,
        },
        true_category="billing",
        true_urgency="medium",
        relevant_kb_article="invoice_visibility_after_payment",
        required_account_fields=["last_payment_status", "invoice_available"],
        required_reply_keywords=["invoice", "24 hours", "billing dashboard"],
        escalation_allowed=False,
        expected_final_action="resolve_ticket",
        max_steps=7,
    ),
    TaskDefinition(
        task_id="task_hard_feature_locked_after_upgrade",
        difficulty="hard",
        title="Premium Export Locked After Upgrade",
        description=(
            "The agent must combine account plan information, seat permission status, and "
            "knowledge base guidance to explain why premium export is still locked after upgrade."
        ),
        ticket_id="TCK-1003",
        customer_message=(
            "I upgraded to Pro today, but premium export is still locked for me. "
            "Why is it not working yet?"
        ),
        customer_profile={
            "plan": "Pro",
            "region": "EU",
            "account_age_days": 95,
            "last_payment_status": "successful",
            "invoice_available": True,
            "feature_flag_enabled": True,
            "workspace_role": "member",
            "seat_permission_granted": False,
        },
        true_category="feature_access",
        true_urgency="high",
        relevant_kb_article="workspace_permission_for_premium_features",
        required_account_fields=["plan", "seat_permission_granted", "workspace_role"],
        required_reply_keywords=[
            "permission",
            "workspace owner",
            "premium export",
            "pro",
        ],
        escalation_allowed=False,
        expected_final_action="resolve_ticket",
        max_steps=9,
    ),
]


TASKS_BY_ID: Dict[str, TaskDefinition] = {task.task_id: task for task in TASKS}


def get_task_by_id(task_id: str) -> TaskDefinition:
    """
    Return a task by its task_id.
    Raises KeyError if task_id is unknown.
    """
    return TASKS_BY_ID[task_id]


def list_tasks() -> List[TaskDefinition]:
    """
    Return all tasks in stable order: easy -> medium -> hard.
    """
    return TASKS.copy()