from __future__ import annotations

from typing import Dict, List, Optional


KNOWLEDGE_BASE: List[Dict[str, object]] = [
    {
        "id": "double_charge_review_policy",
        "title": "Double Charge Review Policy",
        "content": (
            "If a customer reports a duplicate charge, support should first verify the latest "
            "payment records. If two successful charges appear for the same billing cycle, the "
            "case should be escalated to billing operations. If one charge is pending and one is "
            "successful, advise the customer to wait for settlement before escalation."
        ),
        "keywords": [
            "double charge",
            "charged twice",
            "duplicate payment",
            "billing",
            "refund",
        ],
    },
    {
        "id": "invoice_visibility_after_payment",
        "title": "Invoice Visibility After Payment",
        "content": (
            "Invoices can take a short time to appear after a successful payment. If payment "
            "status is successful but invoice_available is false, advise the customer that the "
            "invoice may take up to 24 hours to appear in the billing dashboard."
        ),
        "keywords": [
            "invoice",
            "missing invoice",
            "payment successful",
            "billing dashboard",
            "receipt",
        ],
    },
    {
        "id": "premium_export_after_upgrade",
        "title": "Premium Export Access After Upgrade",
        "content": (
            "After a plan upgrade, premium export access may require a fresh login session. "
            "Support should confirm that the account plan is upgraded successfully and advise the "
            "customer to sign out and sign back in."
        ),
        "keywords": [
            "premium export",
            "upgrade",
            "feature locked",
            "pro plan",
            "re-login",
            "login again",
        ],
    },
    {
        "id": "workspace_permission_for_premium_features",
        "title": "Workspace Permission for Premium Features",
        "content": (
            "Some premium features require both an eligible plan and the correct workspace seat "
            "permission. If seat_permission_granted is false, the workspace owner must assign the "
            "required permission before the feature becomes available."
        ),
        "keywords": [
            "workspace permission",
            "seat permission",
            "owner",
            "feature access",
            "premium feature",
        ],
    },
    {
        "id": "password_reset_link_expiry",
        "title": "Password Reset Link Expiry",
        "content": (
            "Password reset links expire after a short time for security reasons. If a customer "
            "reports an expired reset link, ask them to generate a new reset request and use the "
            "most recent email only."
        ),
        "keywords": [
            "password reset",
            "expired link",
            "account access",
            "reset email",
        ],
    },
    {
        "id": "api_rate_limit_basics",
        "title": "API Rate Limit Basics",
        "content": (
            "API rate limits depend on the customer's plan and current usage. Support should "
            "confirm the plan tier and explain the relevant request-per-minute limit before "
            "suggesting request batching or retry logic."
        ),
        "keywords": [
            "api",
            "rate limit",
            "requests per minute",
            "technical issue",
            "usage",
        ],
    },
]


def get_article_by_id(article_id: str) -> Optional[Dict[str, object]]:
    """
    Return a KB article by its exact id.
    """
    for article in KNOWLEDGE_BASE:
        if article["id"] == article_id:
            return article
    return None


def search_knowledge_base(query: str, top_k: int = 3) -> List[Dict[str, object]]:
    """
    Deterministic keyword-overlap KB search.
    Returns top_k results ordered by score, then by article id.
    """
    if not query or not query.strip():
        return []

    query_lower = query.lower().strip()
    query_tokens = set(query_lower.replace("-", " ").split())

    scored_results = []

    for article in KNOWLEDGE_BASE:
        title = str(article["title"]).lower()
        content = str(article["content"]).lower()
        keywords = [str(k).lower() for k in article.get("keywords", [])]

        score = 0

        # Exact keyword phrase matches
        for kw in keywords:
            if kw in query_lower:
                score += 3

        # Title overlap
        for token in query_tokens:
            if token in title:
                score += 2

        # Content overlap
        for token in query_tokens:
            if token in content:
                score += 1

        if score > 0:
            scored_results.append((score, article))

    scored_results.sort(key=lambda item: (-item[0], str(item[1]["id"])))
    return [article for _, article in scored_results[:top_k]]