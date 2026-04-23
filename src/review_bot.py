"""
review_bot.py – CLI entry point invoked by GitHub Actions.

Usage (in Actions):
    python -m src.review_bot

Required environment variables:
    GITHUB_TOKEN, GITHUB_REPOSITORY, PR_NUMBER
    + one of: GROQ_API_KEY (default) or ANTHROPIC_API_KEY (set LLM_PROVIDER=claude)
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()  # allow local .env during development

from src.services.claude_client import get_llm_client
from src.services.github_client import GitHubClient
from src.services.review_service import run_review
from src.services.storage_service import get_storage

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("review_bot")


def main() -> None:
    # ── Validate env ─────────────────────────────────────────────────────────
    pr_number_raw = os.getenv("PR_NUMBER")
    if not pr_number_raw:
        logger.error("PR_NUMBER environment variable is not set.")
        sys.exit(1)
    try:
        pr_number = int(pr_number_raw)
    except ValueError:
        logger.error("PR_NUMBER must be an integer, got: %s", pr_number_raw)
        sys.exit(1)

    github_token = os.getenv("GITHUB_TOKEN", "")
    if not github_token:
        logger.error("GITHUB_TOKEN is not set.")
        sys.exit(1)

    repo = os.getenv("GITHUB_REPOSITORY", "")
    if not repo:
        logger.error("GITHUB_REPOSITORY is not set.")
        sys.exit(1)

    # Choose LLM provider (groq default, claude fallback)
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("LLM_PROVIDER=claude but ANTHROPIC_API_KEY is not set.")
        sys.exit(1)
    if provider != "claude" and not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY is not set. Get a free key at https://console.groq.com")
        sys.exit(1)

    # ── Instantiate services ────────────────────────────────────────────────
    gh = GitHubClient(token=github_token, repo=repo)
    claude = get_llm_client()
    storage = get_storage()

    # ── Run ──────────────────────────────────────────────────────────────────
    logger.info("Starting Code Review Autopilot for %s PR #%s", repo, pr_number)
    try:
        result = run_review(pr_number, gh, claude, storage)
        logger.info(
            "Review complete — decision: %s | risk: %s (%s)",
            result.get("decision"),
            result.get("risk_score"),
            result.get("risk_level"),
        )
    except Exception:
        logger.exception("Review pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
