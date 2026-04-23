"""
Review service – orchestrates the full review pipeline:
  fetch → context → analyse → risk → format → post → store
"""

from __future__ import annotations

import logging
from typing import Any

from src.services.claude_client import ClaudeClient
from src.services.context_builder import get_related_code_context
from src.services.github_client import GitHubClient, PRFile
from src.services.storage_service import StorageService
from src.utils.formatters import extract_inline_comments, format_summary_comment
from src.utils.risk_engine import ensure_risk_fields

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_files_and_diffs_text(files: list[PRFile]) -> str:
    """Render changed files and their patches for the Claude prompt."""
    parts: list[str] = []
    for f in files:
        header = f"### {f.filename}  (status: {f.status}, +{f.additions}/−{f.deletions})"
        patch = f.patch or "(binary or no patch available)"
        parts.append(f"{header}\n```diff\n{patch}\n```")
    return "\n\n".join(parts)


def _valid_line_map(files: list[PRFile]) -> dict[str, list[int]]:
    """Build a mapping of file → valid new-side line numbers from patches."""
    mapping: dict[str, list[int]] = {}
    for f in files:
        if f.patch:
            mapping[f.filename] = GitHubClient.parse_patch_line_numbers(f.patch)
    return mapping


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_review(
    pr_number: int,
    gh: GitHubClient,
    claude: ClaudeClient,
    storage: StorageService,
) -> dict[str, Any]:
    """
    Execute the full review pipeline for a given PR and return the review dict.

    Steps:
        1. Fetch PR metadata
        2. Fetch changed files + diffs
        3. Fetch related code context
        4. Build AI payload
        5. Call Claude
        6. Validate / recalculate risk fields
        7. Post inline comments
        8. Post summary comment
        9. Save review result
    """

    # 1 ─ PR metadata
    logger.info("Fetching PR #%s metadata …", pr_number)
    meta = gh.get_pr_metadata(pr_number)
    logger.info(
        "PR #%s: '%s' by %s (%s → %s) @ %s",
        meta.number, meta.title, meta.author, meta.branch, meta.base_branch, meta.commit_sha[:8],
    )

    # 2 ─ Changed files & diffs
    logger.info("Fetching changed files …")
    files = gh.get_pr_files_and_diff(pr_number)
    meta.files = files
    logger.info("Changed files: %d", len(files))

    # 3 ─ Related code context
    logger.info("Building related code context …")
    related = get_related_code_context(gh, files, meta.commit_sha)
    logger.info("Related context files gathered: %d", len(related.files))

    # 4 ─ Build AI payload
    files_text = _build_files_and_diffs_text(files)
    context_text = related.as_text()
    user_msg = claude.build_user_message(
        pr_number=meta.number,
        pr_title=meta.title,
        pr_author=meta.author,
        pr_body=meta.body,
        branch=meta.branch,
        base_branch=meta.base_branch,
        files_and_diffs=files_text,
        related_context=context_text,
    )

    # 5 ─ Call Claude
    logger.info("Sending review payload to Claude (%s) …", claude.model)
    review = claude.analyze_with_claude(user_msg)
    logger.info("Claude review received – raw risk_score=%s", review.get("risk_score"))

    # 6 ─ Validate / recalculate risk
    review = ensure_risk_fields(review)
    logger.info(
        "Final risk: score=%s level=%s decision=%s",
        review["risk_score"], review["risk_level"], review["decision"],
    )

    # 7 ─ Post inline comments
    valid_positions = _valid_line_map(files)
    inline_comments = extract_inline_comments(review, valid_positions)
    if inline_comments:
        logger.info("Posting %d inline comments …", len(inline_comments))
        try:
            gh.post_inline_comments(meta.number, meta.commit_sha, inline_comments)
        except Exception:
            logger.exception("Error posting inline comments (non-fatal)")
    else:
        logger.info("No inline comments to post.")

    # 8 ─ Post summary comment
    summary_md = format_summary_comment(review, gh.repo, meta.number)
    logger.info("Posting summary comment …")
    try:
        gh.post_summary_comment(meta.number, summary_md)
    except Exception:
        logger.exception("Error posting summary comment (non-fatal)")

    # 9 ─ Save review result
    review_record = {
        "repo": gh.repo,
        "pr_number": meta.number,
        "pr_title": meta.title,
        "pr_author": meta.author,
        "branch": meta.branch,
        "commit_sha": meta.commit_sha,
        **review,
    }
    logger.info("Saving review result …")
    try:
        storage.save_review_result(review_record)
    except Exception:
        logger.exception("Error saving review result (non-fatal)")

    logger.info("✅ Review pipeline complete for PR #%s", pr_number)
    return review_record
