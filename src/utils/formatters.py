"""
Markdown / text formatters for GitHub PR comments and Streamlit display.
"""

from __future__ import annotations

from typing import Any


# ── Inline comment body ──────────────────────────────────────────────────────

def format_inline_comment(issue: dict[str, Any]) -> str:
    """Build the markdown body for a single inline review comment."""
    severity = issue.get("severity", "Medium")
    emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(severity, "⚪")
    parts = [
        f"**{emoji} {severity} Severity**",
        "",
        f"**Issue:** {issue.get('issue', '')}",
        "",
        f"**Risk:** {issue.get('risk', '')}",
    ]

    affected = issue.get("affected_related_code", [])
    if affected:
        parts.append("")
        parts.append("**Affected related code:** " + ", ".join(f"`{a}`" for a in affected))

    suggestion = issue.get("suggestion", "")
    if suggestion:
        parts.append("")
        parts.append(f"**Suggestion:** {suggestion}")

    code = issue.get("suggested_code", "")
    if code:
        parts.append("")
        parts.append("```suggestion")
        parts.append(code)
        parts.append("```")

    return "\n".join(parts)


# ── PR summary comment ──────────────────────────────────────────────────────

def format_summary_comment(review: dict[str, Any], repo: str, pr_number: int) -> str:
    """Build the polished markdown summary posted as a PR comment."""
    decision_badge = {
        "Approve": "✅ Approve",
        "Needs Changes": "⚠️ Needs Changes",
        "Reject": "❌ Reject",
    }.get(review.get("decision", ""), review.get("decision", ""))

    risk_emoji = {
        "Low": "🟢",
        "Medium": "🟡",
        "High": "🟠",
        "Critical": "🔴",
    }.get(review.get("risk_level", ""), "⚪")

    lines: list[str] = []

    # Header
    lines.append("# 🤖 Code Review Autopilot")
    lines.append("")

    # 1. Summary
    lines.append("## 1. Pull Request Review Summary")
    lines.append("")
    lines.append(review.get("summary", ""))
    lines.append("")

    # 2. Risk Assessment
    lines.append("## 2. Risk Assessment")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| **Risk Score** | **{review.get('risk_score', 'N/A')}** / 100 |")
    lines.append(f"| **Risk Level** | {risk_emoji} {review.get('risk_level', 'N/A')} |")
    lines.append(f"| **Decision** | {decision_badge} |")
    lines.append(f"| **Overall** | {review.get('overall_assessment', 'N/A')} |")
    lines.append("")
    reasoning = review.get("reasoning", "")
    if reasoning:
        lines.append(f"> {reasoning}")
        lines.append("")

    # 3. File-wise Impact
    files = review.get("files", [])
    if files:
        lines.append("## 3. File-wise Impact")
        lines.append("")
        lines.append("| File | Summary |")
        lines.append("|------|---------|")
        for f in files:
            lines.append(f"| `{f.get('file', '')}` | {f.get('summary', '')} |")
        lines.append("")

    # 4. Cross-file Impact
    cross = review.get("cross_file_impact", [])
    if cross:
        lines.append("## 4. Cross-file Impact")
        lines.append("")
        for c in cross:
            lines.append(f"- **{c.get('component', '')}** – {c.get('impact', '')}")
        lines.append("")

    # 5. Key Issues Found
    issues = review.get("issues", [])
    if issues:
        lines.append("## 5. Key Issues Found")
        lines.append("")
        for i, iss in enumerate(issues, 1):
            sev = iss.get("severity", "Medium")
            emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(sev, "⚪")
            lines.append(f"### {i}. {emoji} [{sev}] `{iss.get('file', '')}` (line {iss.get('line', '?')})")
            lines.append("")
            lines.append(f"**Issue:** {iss.get('issue', '')}")
            lines.append("")
            lines.append(f"**Risk:** {iss.get('risk', '')}")
            affected = iss.get("affected_related_code", [])
            if affected:
                lines.append("")
                lines.append("**Affected:** " + ", ".join(f"`{a}`" for a in affected))
            lines.append("")
            lines.append(f"**Suggestion:** {iss.get('suggestion', '')}")
            code = iss.get("suggested_code", "")
            if code:
                lines.append("")
                lines.append("```suggestion")
                lines.append(code)
                lines.append("```")
            lines.append("")

    # 6. Good Improvements
    goods = review.get("good_improvements", [])
    if goods:
        lines.append("## 6. Good Improvements")
        lines.append("")
        for g in goods:
            lines.append(f"- ✅ {g}")
        lines.append("")

    # 7. Bad Regressions
    bads = review.get("bad_regressions", [])
    if bads:
        lines.append("## 7. Bad Regressions")
        lines.append("")
        for b in bads:
            lines.append(f"- ❌ {b}")
        lines.append("")

    # 8. Recommended Actions
    actions = review.get("recommended_actions", [])
    if actions:
        lines.append("## 8. Recommended Actions Before Merge")
        lines.append("")
        for a in actions:
            lines.append(f"- {a}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Generated by Code Review Autopilot for `{repo}` PR #{pr_number}*")

    return "\n".join(lines)


# ── Extract inline comments from review JSON ─────────────────────────────────

def extract_inline_comments(
    review: dict[str, Any],
    valid_positions: dict[str, list[int]] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert the issues array into a list of {file, line, body} dicts
    suitable for posting as inline PR comments.

    If *valid_positions* is supplied (mapping file → list of valid new-side
    line numbers), issues on invalid lines are skipped or snapped to the
    nearest valid line.
    """
    comments: list[dict[str, Any]] = []
    for iss in review.get("issues", []):
        file_path = iss.get("file", "")
        line = iss.get("line")
        if not file_path or not isinstance(line, int) or line < 1:
            continue

        # Validate line position if we have the diff data
        if valid_positions and file_path in valid_positions:
            valid = valid_positions[file_path]
            if line not in valid:
                # Snap to nearest valid line
                if valid:
                    line = min(valid, key=lambda v: abs(v - line))
                else:
                    continue  # no valid lines for this file

        body = format_inline_comment(iss)
        comments.append({"file": file_path, "line": line, "body": body})

    return comments
