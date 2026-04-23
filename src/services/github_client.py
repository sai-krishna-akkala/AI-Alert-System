"""
GitHub REST API client – fetches PR metadata, diffs, files, and posts review comments.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

API_BASE = "https://api.github.com"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PRFile:
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str | None
    raw_url: str | None


@dataclass
class PRMetadata:
    number: int
    title: str
    body: str
    author: str
    branch: str
    base_branch: str
    commit_sha: str
    files: list[PRFile] = field(default_factory=list)


# ── Client ────────────────────────────────────────────────────────────────────

class GitHubClient:
    """Thin wrapper around the GitHub REST API."""

    def __init__(self, token: str | None = None, repo: str | None = None):
        self.token = token or os.getenv("GITHUB_TOKEN", "")
        self.repo = repo or os.getenv("GITHUB_REPOSITORY", "")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _url(self, path: str) -> str:
        return f"{API_BASE}/repos/{self.repo}/{path}"

    def _get(self, path: str, **kwargs: Any) -> Any:
        url = self._url(path)
        resp = self.session.get(url, **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: dict, **kwargs: Any) -> Any:
        url = self._url(path)
        resp = self.session.post(url, json=json, **kwargs)
        resp.raise_for_status()
        return resp.json()

    # ── PR metadata ──────────────────────────────────────────────────────────

    def get_pr_metadata(self, pr_number: int) -> PRMetadata:
        """Return basic PR metadata including commit SHA."""
        data = self._get(f"pulls/{pr_number}")
        return PRMetadata(
            number=data["number"],
            title=data["title"],
            body=data.get("body") or "",
            author=data["user"]["login"],
            branch=data["head"]["ref"],
            base_branch=data["base"]["ref"],
            commit_sha=data["head"]["sha"],
        )

    # ── PR files & diff ──────────────────────────────────────────────────────

    def get_pr_files_and_diff(self, pr_number: int) -> list[PRFile]:
        """Return the list of changed files with their patches."""
        page, all_files = 1, []
        while True:
            items = self._get(f"pulls/{pr_number}/files", params={"per_page": 100, "page": page})
            if not items:
                break
            for f in items:
                all_files.append(
                    PRFile(
                        filename=f["filename"],
                        status=f["status"],
                        additions=f.get("additions", 0),
                        deletions=f.get("deletions", 0),
                        patch=f.get("patch"),
                        raw_url=f.get("raw_url"),
                    )
                )
            if len(items) < 100:
                break
            page += 1
        return all_files

    # ── Fetch file content from repo ────────────────────────────────────────

    def get_file_content(self, path: str, ref: str | None = None) -> str | None:
        """Return the decoded text content of a file at a given ref."""
        try:
            params: dict[str, str] = {}
            if ref:
                params["ref"] = ref
            data = self._get(f"contents/{path}", params=params)
            if data.get("encoding") == "base64":
                import base64
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            return data.get("content", "")
        except requests.HTTPError:
            logger.debug("Could not fetch file %s (ref=%s)", path, ref)
            return None

    # ── Search code in repo ─────────────────────────────────────────────────

    def search_code(self, query: str, max_results: int = 10) -> list[dict]:
        """Search code in the repository. Returns list of {path, text_matches}."""
        try:
            url = f"{API_BASE}/search/code"
            resp = self.session.get(
                url,
                params={"q": f"{query} repo:{self.repo}", "per_page": max_results},
                headers={"Accept": "application/vnd.github.text-match+json"},
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            return [{"path": it["path"], "text_matches": it.get("text_matches", [])} for it in items]
        except requests.HTTPError as exc:
            logger.warning("Code search failed: %s", exc)
            return []

    # ── List directory tree (shallow) ───────────────────────────────────────

    def list_directory(self, path: str, ref: str | None = None) -> list[str]:
        """Return file names in a directory at a given ref."""
        try:
            params: dict[str, str] = {}
            if ref:
                params["ref"] = ref
            data = self._get(f"contents/{path}", params=params)
            if isinstance(data, list):
                return [item["path"] for item in data]
            return []
        except requests.HTTPError:
            return []

    # ── Repo tree (recursive) ───────────────────────────────────────────────

    def get_repo_tree(self, ref: str = "HEAD") -> list[str]:
        """Return all file paths in the repo at a given ref."""
        try:
            data = self._get(f"git/trees/{ref}", params={"recursive": "1"})
            return [item["path"] for item in data.get("tree", []) if item["type"] == "blob"]
        except requests.HTTPError:
            logger.warning("Could not fetch repo tree")
            return []

    # ── Post inline review comment ──────────────────────────────────────────

    def post_inline_comments(
        self,
        pr_number: int,
        commit_sha: str,
        comments: list[dict],
    ) -> list[dict]:
        """
        Post individual review comments on a PR.
        Each item in *comments* must have keys: file, line, body.
        Returns list of API responses (or error dicts).
        """
        results = []
        for c in comments:
            try:
                body = {
                    "body": c["body"],
                    "commit_id": commit_sha,
                    "path": c["file"],
                    "line": c["line"],
                    "side": "RIGHT",
                }
                resp = self._post(f"pulls/{pr_number}/comments", json=body)
                results.append(resp)
            except requests.HTTPError as exc:
                logger.warning(
                    "Failed to post inline comment on %s:%s – %s",
                    c["file"],
                    c["line"],
                    exc,
                )
                # Retry with subject_type=file if line mapping fails
                try:
                    body_fallback = {
                        "body": c["body"],
                        "commit_id": commit_sha,
                        "path": c["file"],
                        "subject_type": "file",
                    }
                    resp = self._post(f"pulls/{pr_number}/comments", json=body_fallback)
                    results.append(resp)
                except requests.HTTPError as exc2:
                    logger.error("Fallback comment also failed: %s", exc2)
                    results.append({"error": str(exc2), "file": c["file"], "line": c["line"]})
        return results

    # ── Post summary comment ────────────────────────────────────────────────

    def post_summary_comment(self, pr_number: int, body: str) -> dict:
        """Post a top-level issue comment as the review summary."""
        return self._post(f"issues/{pr_number}/comments", json={"body": body})

    # ── Helpers for diff parsing ────────────────────────────────────────────

    @staticmethod
    def parse_patch_line_numbers(patch: str) -> list[int]:
        """
        Extract the set of *new-side* line numbers that appear in a unified-diff patch.
        Useful for validating inline comment positions.
        """
        if not patch:
            return []
        lines_in_patch: list[int] = []
        current_line = 0
        for raw in patch.splitlines():
            hunk = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", raw)
            if hunk:
                current_line = int(hunk.group(1))
                continue
            if raw.startswith("-"):
                continue  # deleted line – not in new file
            lines_in_patch.append(current_line)
            current_line += 1
        return lines_in_patch
