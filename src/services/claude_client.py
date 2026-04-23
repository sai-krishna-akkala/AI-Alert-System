"""
LLM client – supports Groq (default, free tier) and Anthropic Claude.

Set LLM_PROVIDER=groq  (default) or LLM_PROVIDER=claude in your .env.

Groq models  : llama-3.3-70b-versatile, llama3-70b-8192, mixtral-8x7b-32768
Claude models: claude-sonnet-4-20250514, claude-3-5-haiku-20241022
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── System prompt (reusable constant) ────────────────────────────────────────

REVIEW_SYSTEM_PROMPT = """\
You are **Code Review Autopilot**, an expert senior software engineer performing a \
thorough, production-grade code review of a GitHub Pull Request.

## Your responsibilities
1. Analyse **every changed file** and its **diff/patch**.
2. Analyse the **related code context** provided (imports, callers, callees, sibling \
   modules, tests, schemas, configs).
3. Identify **good improvements** the PR introduces.
4. Identify **bad regressions** or degradations.
5. Evaluate **cross-file and downstream impact** — business logic side effects, \
   interface breakage, backward-compatibility, determinism, security, performance, \
   reliability, maintainability.
6. Score the risk and recommend a decision (Approve / Needs Changes / Reject).
7. For every issue found, provide a **practical suggestion** and, when helpful, a \
   short **suggested code snippet**.

## Output contract
Return ONLY a single valid JSON object (no markdown fences, no commentary).

{
  "summary": "<concise executive summary>",
  "overall_assessment": "<Good Improvement|Mixed Change|Risky Change|Bad Change>",
  "risk_score": 0,
  "risk_level": "<Low|Medium|High|Critical>",
  "decision": "<Approve|Needs Changes|Reject>",
  "reasoning": "<one-paragraph justification>",
  "cross_file_impact": [{"component": "<name>", "impact": "<description>"}],
  "files": [{"file": "<path>", "summary": "<what changed & why it matters>"}],
  "issues": [
    {
      "file": "<path>",
      "line": 1,
      "severity": "<High|Medium|Low>",
      "issue": "<what is wrong>",
      "risk": "<what could go wrong>",
      "affected_related_code": ["<related file>"],
      "suggestion": "<what to do instead>",
      "suggested_code": "<short code fix or empty string>"
    }
  ],
  "good_improvements": ["<string>"],
  "bad_regressions": ["<string>"],
  "recommended_actions": ["<string>"]
}

Rules: output ONLY the JSON. Use real line numbers. Be honest and concrete.
"""

REVIEW_USER_PROMPT_TEMPLATE = """\
## Pull Request #{pr_number}: {pr_title}

**Author:** {pr_author}
**Branch:** {branch} -> {base_branch}
**Description:**
{pr_body}

---

## Changed Files & Diffs

{files_and_diffs}

---

## Related Code Context

{related_context}

---

Perform a thorough review. Return ONLY the JSON described in the system prompt.
"""


def _parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{", cleaned)
    if match:
        depth, start = 0, match.start()
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start: i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError("LLM did not return valid JSON.")


def _build_user_message(pr_number, pr_title, pr_author, pr_body,
                         branch, base_branch, files_and_diffs, related_context):
    return REVIEW_USER_PROMPT_TEMPLATE.format(
        pr_number=pr_number, pr_title=pr_title, pr_author=pr_author,
        branch=branch, base_branch=base_branch,
        pr_body=pr_body or "(no description)",
        files_and_diffs=files_and_diffs,
        related_context=related_context or "(none available)",
    )


class GroqClient:
    """Groq Chat Completions API – OpenAI-compatible, has free tier."""

    API_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    @staticmethod
    def build_user_message(pr_number, pr_title, pr_author, pr_body,
                            branch, base_branch, files_and_diffs, related_context):
        return _build_user_message(pr_number, pr_title, pr_author, pr_body,
                                    branch, base_branch, files_and_diffs, related_context)

    def analyze_with_claude(self, user_message, max_tokens=8192):
        """Same method name as ClaudeClient so review_service needs no changes."""
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.2,
        }
        resp = self.session.post(self.API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        return _parse_json(text)


class ClaudeClient:
    """Anthropic Messages API client."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        })

    @staticmethod
    def build_user_message(pr_number, pr_title, pr_author, pr_body,
                            branch, base_branch, files_and_diffs, related_context):
        return _build_user_message(pr_number, pr_title, pr_author, pr_body,
                                    branch, base_branch, files_and_diffs, related_context)

    def analyze_with_claude(self, user_message, max_tokens=8192):
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": REVIEW_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }
        resp = self.session.post(self.API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        text = ""
        for block in resp.json().get("content", []):
            if block.get("type") == "text":
                text += block["text"]
        return _parse_json(text)


def get_llm_client():
    """
    Factory – returns Groq or Claude client based on LLM_PROVIDER env var.
    Default: groq
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider == "claude":
        logger.info("LLM provider: Claude (%s)", os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"))
        return ClaudeClient()
    logger.info("LLM provider: Groq (%s)", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    return GroqClient()
