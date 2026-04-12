from __future__ import annotations

from typing import Dict, List

from envs.models import SupportEnvState


def _contains_required_keywords(reply: str | None, required_keywords: List[str]) -> float:
    """
    Returns a fractional keyword match score between 0.0 and 1.0.
    """
    if not reply or not required_keywords:
        return 0.0

    reply_lower = reply.lower()
    matched = sum(1 for keyword in required_keywords if keyword.lower() in reply_lower)
    return matched / len(required_keywords)


def grade_episode(state: SupportEnvState) -> Dict[str, object]:
    """
    Deterministically grade the completed episode using weighted partial credit.
    Final score is clamped to [0.0, 1.0].
    """
    breakdown: Dict[str, float] = {
        "classification": 0.0,
        "urgency": 0.0,
        "account_checks": 0.0,
        "kb_retrieval": 0.0,
        "reply_quality": 0.0,
        "final_action": 0.0,
        "penalties": 0.0,
    }

    # 1. Classification (weight 0.20)
    if state.classification_correct:
        breakdown["classification"] = 0.20

    # 2. Urgency (weight 0.10)
    if state.urgency_correct:
        breakdown["urgency"] = 0.10

    # 3. Account field checks (weight 0.15)
    required_fields = set(state.required_account_fields)
    checked_fields = set(state.checked_account_fields)

    if required_fields:
        matched_fields = len(required_fields.intersection(checked_fields))
        breakdown["account_checks"] = 0.15 * (matched_fields / len(required_fields))

    # 4. KB retrieval (weight 0.15)
    if state.relevant_kb_article:
        if state.relevant_kb_article in state.retrieved_kb_articles:
            breakdown["kb_retrieval"] = 0.15

    # 5. Reply quality (weight 0.20)
    keyword_fraction = _contains_required_keywords(state.last_reply, state.required_reply_keywords)
    breakdown["reply_quality"] = 0.20 * keyword_fraction

    # 6. Final action correctness (weight 0.20)
    if state.expected_final_action == "resolve_ticket" and state.resolved_correctly:
        breakdown["final_action"] = 0.20
    elif state.expected_final_action == "escalate_ticket" and state.escalated_correctly:
        breakdown["final_action"] = 0.20

    # Penalties
    penalties = 0.0
    penalties += min(state.invalid_action_count * 0.05, 0.20)
    penalties += min(state.repeated_action_count * 0.04, 0.16)
    penalties += min(state.wasted_step_count * 0.02, 0.10)

    # Wrong terminal behavior penalty
    if state.current_status == "escalated" and not state.escalated_correctly:
        penalties += 0.15

    if state.current_status == "resolved" and not state.resolved_correctly:
        penalties += 0.20

    breakdown["penalties"] = penalties

    raw_score = (
        breakdown["classification"]
        + breakdown["urgency"]
        + breakdown["account_checks"]
        + breakdown["kb_retrieval"]
        + breakdown["reply_quality"]
        + breakdown["final_action"]
        - breakdown["penalties"]
    )

    final_score = max(0.0, min(1.0, raw_score))

    return {
        "task_id": state.task_id,
        "difficulty": state.difficulty,
        "score": round(final_score, 4),
        "breakdown": breakdown,
        "summary": _build_summary(state, breakdown, final_score),
    }


def _build_summary(
    state: SupportEnvState, breakdown: Dict[str, float], final_score: float
) -> str:
    """
    Build a short deterministic summary for logs/debugging.
    """
    parts = []

    if breakdown["classification"] > 0:
        parts.append("correct classification")
    else:
        parts.append("missed classification")

    if breakdown["urgency"] > 0:
        parts.append("correct urgency")
    else:
        parts.append("missed urgency")

    if breakdown["account_checks"] > 0:
        parts.append("relevant account fields checked")
    else:
        parts.append("no relevant account checks")

    if breakdown["kb_retrieval"] > 0:
        parts.append("correct KB retrieved")
    else:
        parts.append("KB not retrieved")

    if breakdown["reply_quality"] > 0:
        parts.append("useful customer reply")
    else:
        parts.append("reply missing required guidance")

    if breakdown["final_action"] > 0:
        parts.append("correct terminal action")
    else:
        parts.append("incorrect terminal action")

    if breakdown["penalties"] > 0:
        parts.append(f"penalties applied: {breakdown['penalties']:.2f}")

    parts.append(f"final score: {final_score:.4f}")
    return "; ".join(parts)