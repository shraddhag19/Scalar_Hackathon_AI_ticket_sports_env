from __future__ import annotations

from typing import Any, Dict, Optional

from envs.graders import grade_episode
from envs.knowledge_base import get_article_by_id, search_knowledge_base
from envs.models import (
    Action,
    InteractionEntry,
    Observation,
    Reward,
    SupportEnvState,
    TaskDefinition,
)
from envs.tasks import get_task_by_id, list_tasks


class SupportDeskEnv:
    """
    OpenEnv-style customer support environment.
    """

    def __init__(self) -> None:
        self.available_tasks = list_tasks()
        self.current_task: Optional[TaskDefinition] = None
        self._state: Optional[SupportEnvState] = None

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """
        Reset the environment to a fresh task state.
        If task_id is not provided, default to the first task.
        """
        if task_id is None:
            task = self.available_tasks[0]
        else:
            task = get_task_by_id(task_id)

        self.current_task = task
        self._state = SupportEnvState(
            task_id=task.task_id,
            difficulty=task.difficulty,
            ticket_id=task.ticket_id,
            customer_message=task.customer_message,
            customer_profile=task.customer_profile,
            true_category=task.true_category,
            true_urgency=task.true_urgency,
            relevant_kb_article=task.relevant_kb_article,
            required_account_fields=task.required_account_fields,
            required_reply_keywords=task.required_reply_keywords,
            escalation_allowed=task.escalation_allowed,
            expected_final_action=task.expected_final_action,
            max_steps=task.max_steps,
        )
        return self._build_observation()

    def state(self) -> SupportEnvState:
        """
        Return the full internal environment state.
        """
        if self._state is None:
            raise RuntimeError("Environment has not been reset yet.")
        return self._state

    def step(self, action: Action | Dict[str, Any]) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply one action to the environment.
        Returns: observation, reward, done, info
        """
        if self._state is None:
            raise RuntimeError("Environment has not been reset yet.")

        if self._state.done:
            observation = self._build_observation()
            reward = Reward(score=0.0, reason="Episode is already finished.")
            info = {"warning": "No further actions allowed after done=True."}
            return observation, reward, True, info

        if isinstance(action, dict):
            action = Action(**action)

        self._state.step_count += 1
        self._state.action_history.append(action)
        self._state.current_status = "in_progress"

        reward_score = 0.0
        reward_reason = "Action processed."
        info: Dict[str, Any] = {"task_id": self._state.task_id}

        if action.action_type == "classify_ticket":
            reward_score, reward_reason = self._handle_classify_ticket(action)

        elif action.action_type == "search_kb":
            reward_score, reward_reason, info = self._handle_search_kb(action)

        elif action.action_type == "check_account":
            reward_score, reward_reason, info = self._handle_check_account(action)

        elif action.action_type == "reply_customer":
            reward_score, reward_reason = self._handle_reply_customer(action)

        elif action.action_type == "escalate_ticket":
            reward_score, reward_reason = self._handle_escalate_ticket(action)

        elif action.action_type == "resolve_ticket":
            reward_score, reward_reason = self._handle_resolve_ticket(action)

        else:
            reward_score = -0.05
            reward_reason = f"Unsupported action type: {action.action_type}"
            self._state.invalid_action_count += 1

        # Penalize exceeding useful trajectory length
        if not self._state.done and self._state.step_count > self._state.max_steps:
            self._state.done = True
            self._state.wasted_step_count += 1
            reward_score -= 0.10
            reward_reason += " Maximum step limit exceeded."

        # End episode automatically if max steps reached
        if not self._state.done and self._state.step_count >= self._state.max_steps:
            self._state.done = True
            if self._state.current_status not in ("resolved", "escalated"):
                self._state.current_status = "open"
            info["grader"] = grade_episode(self._state)

        if self._state.done and "grader" not in info:
            info["grader"] = grade_episode(self._state)

        observation = self._build_observation()
        reward = Reward(score=max(-1.0, min(1.0, reward_score)), reason=reward_reason)
        return observation, reward, self._state.done, info

    def _build_observation(self) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment has not been reset yet.")

        remaining_steps = max(0, self._state.max_steps - self._state.step_count)

        return Observation(
            ticket_id=self._state.ticket_id,
            task_id=self._state.task_id,
            customer_message=self._state.customer_message,
            customer_profile=self._state.customer_profile,
            interaction_history=self._state.interaction_history,
            visible_tools=[
                "classify_ticket",
                "search_kb",
                "check_account",
                "reply_customer",
                "escalate_ticket",
                "resolve_ticket",
            ],
            current_status=self._state.current_status,
            remaining_steps=remaining_steps,
            last_tool_result=self._state.last_tool_result,
        )

    def _record_interaction(self, action_type: str, action_summary: str, result: str) -> None:
        if self._state is None:
            return

        entry = InteractionEntry(
            step_number=self._state.step_count,
            action_type=action_type,
            action_summary=action_summary,
            result=result,
        )
        self._state.interaction_history.append(entry)

    def _handle_classify_ticket(self, action: Action) -> tuple[float, str]:
        assert self._state is not None

        if action.category is None or action.urgency is None:
            self._state.invalid_action_count += 1
            self._state.wasted_step_count += 1
            result = "Classification failed because category or urgency was missing."
            self._record_interaction("classify_ticket", "missing category/urgency", result)
            return -0.05, result

        reward = 0.0
        messages = []

        if action.category == self._state.true_category:
            self._state.classification_correct = True
            reward += 0.15
            messages.append("Correct category selected.")
        else:
            messages.append("Incorrect category selected.")
            self._state.wasted_step_count += 1

        if action.urgency == self._state.true_urgency:
            self._state.urgency_correct = True
            reward += 0.05
            messages.append("Correct urgency selected.")
        else:
            messages.append("Incorrect urgency selected.")
            self._state.wasted_step_count += 1

        result = " ".join(messages)
        self._record_interaction(
            "classify_ticket",
            f"category={action.category}, urgency={action.urgency}",
            result,
        )
        return reward, result

    def _handle_search_kb(self, action: Action) -> tuple[float, str, Dict[str, Any]]:
        assert self._state is not None

        if not action.query:
            self._state.invalid_action_count += 1
            self._state.wasted_step_count += 1
            result = "KB search failed because query was empty."
            self._state.last_tool_result = result
            self._record_interaction("search_kb", "empty query", result)
            return -0.05, result, {"kb_results": []}

        results = search_knowledge_base(action.query, top_k=3)
        kb_ids = [str(article["id"]) for article in results]

        reward = 0.0
        if self._state.relevant_kb_article and self._state.relevant_kb_article in kb_ids:
            if self._state.relevant_kb_article not in self._state.retrieved_kb_articles:
                self._state.retrieved_kb_articles.append(self._state.relevant_kb_article)
                reward += 0.15
                result = f"Relevant KB article found: {self._state.relevant_kb_article}"
            else:
                self._state.repeated_action_count += 1
                reward -= 0.04
                result = f"Relevant KB article already retrieved earlier: {self._state.relevant_kb_article}"
        else:
            self._state.wasted_step_count += 1
            reward -= 0.02
            result = "No highly relevant KB article found."

        self._state.last_tool_result = result
        self._record_interaction("search_kb", action.query, result)

        return reward, result, {"kb_results": kb_ids}

    def _handle_check_account(self, action: Action) -> tuple[float, str, Dict[str, Any]]:
        assert self._state is not None

        if not action.account_field:
            self._state.invalid_action_count += 1
            self._state.wasted_step_count += 1
            result = "Account check failed because account_field was missing."
            self._state.last_tool_result = result
            self._record_interaction("check_account", "missing account_field", result)
            return -0.05, result, {}

        field_name = action.account_field

        if field_name not in self._state.customer_profile:
            self._state.invalid_action_count += 1
            self._state.wasted_step_count += 1
            result = f"Unknown account field: {field_name}"
            self._state.last_tool_result = result
            self._record_interaction("check_account", field_name, result)
            return -0.05, result, {}

        field_value = self._state.customer_profile[field_name]

        reward = 0.0
        if field_name in self._state.required_account_fields:
            if field_name not in self._state.checked_account_fields:
                self._state.checked_account_fields.append(field_name)
                reward += 0.10
                result = f"Relevant account field checked: {field_name}={field_value}"
            else:
                self._state.repeated_action_count += 1
                reward -= 0.04
                result = f"Account field already checked earlier: {field_name}={field_value}"
        else:
            self._state.wasted_step_count += 1
            reward -= 0.02
            result = f"Account field checked but not especially relevant: {field_name}={field_value}"

        self._state.last_tool_result = result
        self._record_interaction("check_account", field_name, result)

        return reward, result, {"account_field": field_name, "value": field_value}

    def _handle_reply_customer(self, action: Action) -> tuple[float, str]:
        assert self._state is not None

        if not action.message:
            self._state.invalid_action_count += 1
            self._state.wasted_step_count += 1
            result = "Reply failed because message was empty."
            self._record_interaction("reply_customer", "empty message", result)
            return -0.05, result

        self._state.last_reply = action.message
        reply_lower = action.message.lower()

        matched_keywords = [
            keyword for keyword in self._state.required_reply_keywords if keyword.lower() in reply_lower
        ]

        if self._state.required_reply_keywords:
            fraction = len(matched_keywords) / len(self._state.required_reply_keywords)
        else:
            fraction = 0.0

        reward = 0.20 * fraction

        if fraction == 0:
            self._state.wasted_step_count += 1
            result = "Reply sent, but it missed the required guidance."
        elif fraction < 1:
            result = f"Reply sent with partial guidance. Matched keywords: {matched_keywords}"
        else:
            result = "Reply sent with all required guidance."

        self._record_interaction("reply_customer", action.message, result)
        return reward, result

    def _handle_escalate_ticket(self, action: Action) -> tuple[float, str]:
        assert self._state is not None

        self._state.done = True
        self._state.current_status = "escalated"

        if self._state.escalation_allowed and self._state.expected_final_action == "escalate_ticket":
            self._state.escalated_correctly = True
            result = "Ticket escalated correctly."
            reward = 0.25
        else:
            self._state.escalated_correctly = False
            self._state.wasted_step_count += 1
            result = "Ticket escalated unnecessarily."
            reward = -0.15

        summary = action.reason or "no reason provided"
        self._record_interaction("escalate_ticket", summary, result)
        return reward, result

    def _handle_resolve_ticket(self, action: Action) -> tuple[float, str]:
        assert self._state is not None

        self._state.done = True
        self._state.current_status = "resolved"

        enough_progress = (
            self._state.classification_correct
            and (
                not self._state.required_account_fields
                or len(set(self._state.required_account_fields).intersection(set(self._state.checked_account_fields))) > 0
            )
        )

        if self._state.expected_final_action == "resolve_ticket" and enough_progress:
            self._state.resolved_correctly = True
            result = "Ticket resolved correctly."
            reward = 0.25
        else:
            self._state.resolved_correctly = False
            self._state.wasted_step_count += 1
            result = "Ticket resolved prematurely or incorrectly."
            reward = -0.20

        summary = action.resolution_note or action.reason or "no resolution note provided"
        self._record_interaction("resolve_ticket", summary, result)
        return reward, result