from envs.support_env import SupportDeskEnv
from envs.tasks import list_tasks


def test_all_tasks_are_available():
    tasks = list_tasks()
    assert len(tasks) >= 3
    assert tasks[0].difficulty == "easy"
    assert tasks[1].difficulty == "medium"
    assert tasks[2].difficulty == "hard"


def test_reset_returns_valid_observation():
    env = SupportDeskEnv()
    obs = env.reset("task_easy_billing_triage")

    assert obs.ticket_id == "TCK-1001"
    assert obs.task_id == "task_easy_billing_triage"
    assert obs.current_status == "open"
    assert obs.remaining_steps == 5
    assert isinstance(obs.visible_tools, list)
    assert "classify_ticket" in obs.visible_tools


def test_step_classification_returns_reward():
    env = SupportDeskEnv()
    env.reset("task_easy_billing_triage")

    obs, reward, done, info = env.step(
        {
            "action_type": "classify_ticket",
            "category": "billing",
            "urgency": "medium",
        }
    )

    assert reward.score >= 0.0
    assert done is False
    assert obs.current_status == "in_progress"
    assert "task_id" in info


def test_check_account_relevant_field_gives_positive_reward():
    env = SupportDeskEnv()
    env.reset("task_medium_missing_invoice")

    env.step(
        {
            "action_type": "classify_ticket",
            "category": "billing",
            "urgency": "medium",
        }
    )

    obs, reward, done, info = env.step(
        {
            "action_type": "check_account",
            "account_field": "last_payment_status",
        }
    )

    assert reward.score > 0
    assert done is False
    assert "account_field" in info


def test_search_kb_can_find_relevant_article():
    env = SupportDeskEnv()
    env.reset("task_medium_missing_invoice")

    obs, reward, done, info = env.step(
        {
            "action_type": "search_kb",
            "query": "missing invoice payment successful billing dashboard",
        }
    )

    assert done is False
    assert "kb_results" in info
    assert isinstance(info["kb_results"], list)


def test_resolve_finishes_episode():
    env = SupportDeskEnv()
    env.reset("task_easy_billing_triage")

    env.step(
        {
            "action_type": "classify_ticket",
            "category": "billing",
            "urgency": "medium",
        }
    )

    env.step(
        {
            "action_type": "check_account",
            "account_field": "last_payment_status",
        }
    )

    env.step(
        {
            "action_type": "reply_customer",
            "message": "We are reviewing this billing issue now.",
        }
    )

    obs, reward, done, info = env.step(
        {
            "action_type": "resolve_ticket",
            "resolution_note": "Resolved after billing review.",
        }
    )

    assert done is True
    assert obs.current_status == "resolved"
    assert "grader" in info
    assert 0.0 <= info["grader"]["score"] <= 1.0


def test_escalate_finishes_episode():
    env = SupportDeskEnv()
    env.reset("task_easy_billing_triage")

    obs, reward, done, info = env.step(
        {
            "action_type": "escalate_ticket",
            "reason": "Escalating to billing team.",
        }
    )

    assert done is True
    assert obs.current_status == "escalated"
    assert "grader" in info
    assert 0.0 <= info["grader"]["score"] <= 1.0


def test_invalid_account_field_gets_penalty():
    env = SupportDeskEnv()
    env.reset("task_easy_billing_triage")

    obs, reward, done, info = env.step(
        {
            "action_type": "check_account",
            "account_field": "unknown_field",
        }
    )

    assert reward.score < 0
    assert done is False


def test_state_returns_internal_state():
    env = SupportDeskEnv()
    env.reset("task_hard_feature_locked_after_upgrade")
    state = env.state()

    assert state.task_id == "task_hard_feature_locked_after_upgrade"
    assert state.true_category == "feature_access"
    assert state.done is False


def test_done_episode_blocks_further_progress():
    env = SupportDeskEnv()
    env.reset("task_easy_billing_triage")

    env.step(
        {
            "action_type": "classify_ticket",
            "category": "billing",
            "urgency": "medium",
        }
    )

    env.step(
        {
            "action_type": "resolve_ticket",
            "resolution_note": "Resolved.",
        }
    )

    obs, reward, done, info = env.step(
        {
            "action_type": "reply_customer",
            "message": "Extra message after done.",
        }
    )

    assert done is True
    assert reward.score == 0.0
    assert "warning" in info