"""
Risk engine – computes risk score, risk level, and decision
when Claude's own score is missing or needs recalculation.
"""

from __future__ import annotations

from typing import Any


def calculate_risk_score(review: dict[str, Any]) -> int:
    """
    Compute a risk score (0–100, higher = safer) based on issue counts and flags.

    Rules
    -----
    - Start at 100
    - −20 per High severity issue
    - −10 per Medium severity issue
    -  −5 per Low severity issue
    - −10 extra if core business-logic change impacts related modules
    - −10 extra if determinism / backward-compatibility is flagged
    - Clamped to [0, 100]
    """
    score = 100

    issues: list[dict] = review.get("issues", [])
    for iss in issues:
        sev = (iss.get("severity") or "").lower()
        if sev == "high":
            score -= 20
        elif sev == "medium":
            score -= 10
        elif sev == "low":
            score -= 5

    # Extra penalties based on cross-file impact & regressions
    cross_impact = review.get("cross_file_impact", [])
    if cross_impact:
        for ci in cross_impact:
            impact_text = (ci.get("impact") or "").lower()
            if any(kw in impact_text for kw in ("business logic", "core", "critical")):
                score -= 10
                break

    regressions = review.get("bad_regressions", [])
    for reg in regressions:
        reg_lower = reg.lower()
        if any(kw in reg_lower for kw in ("determinism", "backward", "compatibility", "breaking")):
            score -= 10
            break

    return max(0, min(100, score))


def risk_level(score: int) -> str:
    """Map a numeric risk score to a label."""
    if score >= 80:
        return "Low"
    if score >= 50:
        return "Medium"
    if score >= 25:
        return "High"
    return "Critical"


def generate_decision(score: int) -> str:
    """Map a numeric risk score to a review decision."""
    if score >= 80:
        return "Approve"
    if score >= 50:
        return "Needs Changes"
    return "Reject"


def ensure_risk_fields(review: dict[str, Any]) -> dict[str, Any]:
    """
    Fill in risk_score / risk_level / decision if Claude left them out
    or returned invalid values.
    """
    # Validate / recompute score
    raw_score = review.get("risk_score")
    if not isinstance(raw_score, (int, float)) or not (0 <= raw_score <= 100):
        raw_score = calculate_risk_score(review)
    score = int(raw_score)

    review["risk_score"] = score
    review["risk_level"] = risk_level(score)

    # Validate / recompute decision
    valid_decisions = {"Approve", "Needs Changes", "Reject"}
    if review.get("decision") not in valid_decisions:
        review["decision"] = generate_decision(score)

    # Validate overall_assessment
    valid_assessments = {"Good Improvement", "Mixed Change", "Risky Change", "Bad Change"}
    if review.get("overall_assessment") not in valid_assessments:
        if score >= 80:
            review["overall_assessment"] = "Good Improvement"
        elif score >= 50:
            review["overall_assessment"] = "Mixed Change"
        elif score >= 25:
            review["overall_assessment"] = "Risky Change"
        else:
            review["overall_assessment"] = "Bad Change"

    return review
