"""
Context builder – gathers related code files so Claude can review
not just the diff but also the surrounding code that might be affected.

Heuristics used:
  1. Imports / usings found in changed files
  2. Sibling files in the same directory / module
  3. Test files matching naming conventions
  4. Schema / config / model files matching symbol names
  5. Direct callers / callees discovered via code search
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.services.github_client import GitHubClient, PRFile

logger = logging.getLogger(__name__)

MAX_CONTEXT_FILES = int(os.getenv("MAX_CONTEXT_FILES", "15"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "80000"))
# rough chars-per-token estimate for code
CHARS_PER_TOKEN = 4


@dataclass
class ContextFile:
    path: str
    snippet: str
    reason: str


@dataclass
class RelatedContext:
    files: list[ContextFile] = field(default_factory=list)

    def as_text(self) -> str:
        if not self.files:
            return "(no related context gathered)"
        parts: list[str] = []
        for cf in self.files:
            parts.append(f"### {cf.path}  (reason: {cf.reason})\n```\n{cf.snippet}\n```")
        return "\n\n".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def get_related_code_context(
    gh: GitHubClient,
    changed_files: list[PRFile],
    commit_sha: str,
) -> RelatedContext:
    """
    Main entry – collects dependency context, sibling context, test context,
    and caller/callee context up to token budget.
    """
    ctx = RelatedContext()
    budget = MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN
    seen_paths: set[str] = {f.filename for f in changed_files}

    # 1. Dependency context (imports)
    _add_dependency_context(gh, changed_files, commit_sha, ctx, seen_paths, budget)
    # 2. Sibling / module context
    _add_sibling_context(gh, changed_files, commit_sha, ctx, seen_paths, budget)
    # 3. Test context
    _add_test_context(gh, changed_files, commit_sha, ctx, seen_paths, budget)
    # 4. Caller / callee context via code search
    _add_caller_callee_context(gh, changed_files, ctx, seen_paths, budget)

    # Trim to budget
    _trim_to_budget(ctx, budget)
    return ctx


def get_dependency_context(
    gh: GitHubClient,
    changed_files: list[PRFile],
    commit_sha: str,
) -> list[ContextFile]:
    """Return files imported/used by changed files."""
    ctx = RelatedContext()
    seen: set[str] = {f.filename for f in changed_files}
    _add_dependency_context(gh, changed_files, commit_sha, ctx, seen, MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN)
    return ctx.files


def get_relevant_tests(
    gh: GitHubClient,
    changed_files: list[PRFile],
    commit_sha: str,
) -> list[ContextFile]:
    """Return test files related to the changed files."""
    ctx = RelatedContext()
    seen: set[str] = {f.filename for f in changed_files}
    _add_test_context(gh, changed_files, commit_sha, ctx, seen, MAX_CONTEXT_TOKENS * CHARS_PER_TOKEN)
    return ctx.files


# ── Internal collectors ──────────────────────────────────────────────────────

def _used_chars(ctx: RelatedContext) -> int:
    return sum(len(cf.snippet) for cf in ctx.files)


def _room(ctx: RelatedContext, budget: int) -> bool:
    return _used_chars(ctx) < budget and len(ctx.files) < MAX_CONTEXT_FILES


def _fetch_and_add(
    gh: GitHubClient,
    path: str,
    ref: str,
    reason: str,
    ctx: RelatedContext,
    seen: set[str],
    budget: int,
    max_chars: int = 12000,
) -> None:
    if path in seen or not _room(ctx, budget):
        return
    content = gh.get_file_content(path, ref=ref)
    if content is None:
        return
    snippet = content[:max_chars]
    ctx.files.append(ContextFile(path=path, snippet=snippet, reason=reason))
    seen.add(path)


# ---- 1. Imports / dependency context ----------------------------------------

_IMPORT_PATTERNS: list[re.Pattern[str]] = [
    # Python: from x.y import z  |  import x.y
    re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE),
    # JS/TS: import … from "path"  |  require("path")
    re.compile(r"""(?:import\s+.*?\s+from\s+['"]([^'"]+)['"]|require\s*\(\s*['"]([^'"]+)['"]\s*\))""", re.MULTILINE),
    # Go: import "path"
    re.compile(r'import\s+"([^"]+)"', re.MULTILINE),
    # Java/Kotlin: import x.y.z;
    re.compile(r"^\s*import\s+([\w.]+);", re.MULTILINE),
]


def _extract_imports(source: str) -> list[str]:
    """Extract import paths from source code (best-effort, multi-language)."""
    imports: list[str] = []
    for pat in _IMPORT_PATTERNS:
        for m in pat.finditer(source):
            groups = [g for g in m.groups() if g]
            imports.extend(groups)
    return imports


def _resolve_import_to_path(imp: str, changed_dirs: set[str], tree: list[str]) -> str | None:
    """Try to map an import string to a file in the repo tree."""
    # Python style: replace dots → /
    candidates = [
        imp.replace(".", "/") + ".py",
        imp.replace(".", "/") + "/__init__.py",
        imp.replace(".", "/") + ".ts",
        imp.replace(".", "/") + ".js",
        imp + ".py",
        imp + ".ts",
        imp + ".js",
        imp,
    ]
    # Also try relative to changed directories
    for d in changed_dirs:
        candidates.append(f"{d}/{imp.split('.')[-1]}.py")
        candidates.append(f"{d}/{imp.replace('.', '/')}.py")

    tree_set = set(tree)
    for c in candidates:
        c_norm = c.lstrip("./")
        if c_norm in tree_set:
            return c_norm
    return None


def _add_dependency_context(
    gh: GitHubClient,
    changed_files: list[PRFile],
    ref: str,
    ctx: RelatedContext,
    seen: set[str],
    budget: int,
) -> None:
    tree: list[str] | None = None
    changed_dirs = {str(PurePosixPath(f.filename).parent) for f in changed_files}

    for f in changed_files:
        if not f.patch:
            continue
        imports = _extract_imports(f.patch)
        if not imports:
            continue
        if tree is None:
            tree = gh.get_repo_tree(ref)
        for imp in imports:
            resolved = _resolve_import_to_path(imp, changed_dirs, tree)
            if resolved:
                _fetch_and_add(gh, resolved, ref, f"imported by {f.filename}", ctx, seen, budget)


# ---- 2. Sibling / module context -------------------------------------------

_SIBLING_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".java", ".kt", ".rs"}


def _add_sibling_context(
    gh: GitHubClient,
    changed_files: list[PRFile],
    ref: str,
    ctx: RelatedContext,
    seen: set[str],
    budget: int,
) -> None:
    dirs_fetched: set[str] = set()
    for f in changed_files:
        parent = str(PurePosixPath(f.filename).parent)
        if parent in dirs_fetched:
            continue
        dirs_fetched.add(parent)
        siblings = gh.list_directory(parent, ref=ref)
        for sib in siblings:
            ext = PurePosixPath(sib).suffix
            if ext in _SIBLING_EXTENSIONS:
                _fetch_and_add(gh, sib, ref, f"sibling in {parent}/", ctx, seen, budget, max_chars=6000)


# ---- 3. Test context -------------------------------------------------------

_TEST_PATTERNS = [
    # test_<name>.py, <name>_test.py, <name>.test.ts, __tests__/<name>.js …
    re.compile(r"(?:test_|_test\.|\.test\.|\.spec\.|__tests__/)"),
]


def _test_candidates(filename: str) -> list[str]:
    """Generate candidate test file names for a source file."""
    p = PurePosixPath(filename)
    stem, ext = p.stem, p.suffix
    parent = str(p.parent)
    candidates = [
        f"{parent}/test_{stem}{ext}",
        f"{parent}/{stem}_test{ext}",
        f"{parent}/{stem}.test{ext}",
        f"{parent}/{stem}.spec{ext}",
        f"{parent}/__tests__/{stem}{ext}",
        f"tests/{parent}/{stem}{ext}",
        f"tests/test_{stem}{ext}",
        f"test/{stem}{ext}",
    ]
    return candidates


def _add_test_context(
    gh: GitHubClient,
    changed_files: list[PRFile],
    ref: str,
    ctx: RelatedContext,
    seen: set[str],
    budget: int,
) -> None:
    tree: list[str] | None = None
    for f in changed_files:
        # Skip if the changed file itself is already a test
        if any(pat.search(f.filename) for pat in _TEST_PATTERNS):
            continue
        candidates = _test_candidates(f.filename)
        if tree is None:
            tree = gh.get_repo_tree(ref)
        tree_set = set(tree)
        for cand in candidates:
            if cand in tree_set:
                _fetch_and_add(gh, cand, ref, f"test for {f.filename}", ctx, seen, budget, max_chars=8000)


# ---- 4. Caller / callee context via code search ----------------------------

_SYMBOL_RE = re.compile(r"\b(?:def|class|function|func|interface|type)\s+(\w+)")


def _extract_symbols(patch: str) -> list[str]:
    """Extract function/class names defined or modified in a patch."""
    return list({m.group(1) for m in _SYMBOL_RE.finditer(patch)})


def _add_caller_callee_context(
    gh: GitHubClient,
    changed_files: list[PRFile],
    ctx: RelatedContext,
    seen: set[str],
    budget: int,
) -> None:
    for f in changed_files:
        if not f.patch:
            continue
        symbols = _extract_symbols(f.patch)
        for sym in symbols[:5]:  # cap to avoid excessive API calls
            results = gh.search_code(sym, max_results=5)
            for r in results:
                path = r["path"]
                if path not in seen and _room(ctx, budget):
                    # Just note the path + reason; we won't fetch full content to save budget
                    snippet = ""
                    for tm in r.get("text_matches", []):
                        snippet += tm.get("fragment", "") + "\n"
                    if snippet:
                        snippet = snippet[:4000]
                        ctx.files.append(
                            ContextFile(
                                path=path,
                                snippet=snippet,
                                reason=f"references symbol `{sym}` from {f.filename}",
                            )
                        )
                        seen.add(path)


# ── Budget trimming ──────────────────────────────────────────────────────────

def _trim_to_budget(ctx: RelatedContext, budget: int) -> None:
    total = 0
    keep: list[ContextFile] = []
    for cf in ctx.files:
        if total + len(cf.snippet) > budget:
            remaining = budget - total
            if remaining > 500:
                cf.snippet = cf.snippet[:remaining] + "\n… (truncated)"
                keep.append(cf)
            break
        keep.append(cf)
        total += len(cf.snippet)
    ctx.files = keep
