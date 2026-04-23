"""
Code Review Autopilot — Streamlit Dashboard

A read-only dashboard that displays PR reviews stored by the review bot.
No manual PR analysis triggers — data comes from the shared SQLite / Postgres store.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.services.storage_service import get_storage

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Code Review Autopilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* tighter spacing */
    .block-container { padding-top: 1.5rem; }

    /* decision badges */
    .badge-approve  { background:#22c55e; color:#fff; padding:4px 12px; border-radius:12px; font-weight:700; }
    .badge-needs    { background:#f59e0b; color:#fff; padding:4px 12px; border-radius:12px; font-weight:700; }
    .badge-reject   { background:#ef4444; color:#fff; padding:4px 12px; border-radius:12px; font-weight:700; }

    /* risk colours */
    .risk-low      { color:#22c55e; font-weight:700; }
    .risk-medium   { color:#f59e0b; font-weight:700; }
    .risk-high     { color:#f97316; font-weight:700; }
    .risk-critical { color:#ef4444; font-weight:700; }

    /* card look */
    .review-card {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        background: rgba(255,255,255,0.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _decision_badge(decision: str) -> str:
    cls = {"Approve": "badge-approve", "Needs Changes": "badge-needs", "Reject": "badge-reject"}.get(
        decision, "badge-needs"
    )
    return f'<span class="{cls}">{decision}</span>'


def _risk_span(level: str) -> str:
    cls = {"Low": "risk-low", "Medium": "risk-medium", "High": "risk-high", "Critical": "risk-critical"}.get(
        level, ""
    )
    return f'<span class="{cls}">{level}</span>'


def _severity_icon(sev: str) -> str:
    return {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(sev, "⚪")


# ── Sidebar filters ─────────────────────────────────────────────────────────

def _sidebar_filters() -> dict:
    st.sidebar.title("🤖 Code Review Autopilot")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")

    # We load all reviews first so we can derive filter options
    storage = get_storage()
    all_reviews = storage.load_review_results()

    repos = sorted({r.get("repo", "") for r in all_reviews if r.get("repo")})
    selected_repo = st.sidebar.selectbox("Repository", ["All"] + repos)

    risk_levels = ["All", "Low", "Medium", "High", "Critical"]
    selected_risk = st.sidebar.selectbox("Risk Level", risk_levels)

    decisions = ["All", "Approve", "Needs Changes", "Reject"]
    selected_decision = st.sidebar.selectbox("Decision", decisions)

    date_range = st.sidebar.date_input(
        "Date range",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
    )

    filters: dict = {}
    if selected_repo != "All":
        filters["repo"] = selected_repo
    if selected_risk != "All":
        filters["risk_level"] = selected_risk
    if selected_decision != "All":
        filters["decision"] = selected_decision
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        filters["date_from"] = str(date_range[0])
        filters["date_to"] = str(date_range[1]) + "T23:59:59"

    st.sidebar.markdown("---")
    st.sidebar.caption("Data refreshes on page load.")

    return filters


# ── Review list ──────────────────────────────────────────────────────────────

def _render_review_list(reviews: list[dict]) -> int | None:
    """Render a compact list of reviews and return the selected index."""
    if not reviews:
        st.info("No reviews found. Reviews appear here automatically after the GitHub Actions bot runs.")
        return None

    st.markdown(f"### Showing **{len(reviews)}** review(s)")

    selected_idx: int | None = None
    for idx, r in enumerate(reviews):
        with st.container():
            cols = st.columns([0.5, 3, 1.5, 1, 1, 1.5])
            cols[0].markdown(f"**#{r.get('pr_number', '?')}**")
            cols[1].markdown(f"**{r.get('pr_title', 'Untitled')}**  \n`{r.get('repo', '')}`")
            cols[2].markdown(f"🧑‍💻 {r.get('pr_author', 'unknown')}  \n🌿 `{r.get('branch', '')}`")
            cols[3].markdown(
                f"Score: **{r.get('risk_score', '?')}**  \n{_risk_span(r.get('risk_level', ''))}",
                unsafe_allow_html=True,
            )
            cols[4].markdown(_decision_badge(r.get("decision", "")), unsafe_allow_html=True)
            if cols[5].button("View", key=f"view_{idx}"):
                selected_idx = idx
            st.divider()

    return selected_idx


# ── Detailed review view ────────────────────────────────────────────────────

def _render_detail(r: dict) -> None:
    """Render the full review report for a single PR."""
    st.markdown("---")
    st.markdown(f"## 🤖 Review: PR #{r.get('pr_number')} — {r.get('pr_title', '')}")

    # Meta
    meta_cols = st.columns(4)
    meta_cols[0].metric("Repository", r.get("repo", ""))
    meta_cols[1].metric("Author", r.get("pr_author", ""))
    meta_cols[2].metric("Branch", r.get("branch", ""))
    meta_cols[3].metric("Commit", (r.get("commit_sha") or "")[:8])

    st.markdown("")

    # ─ 1. Summary
    st.subheader("1. Pull Request Review Summary")
    st.markdown(r.get("summary", ""))

    # ─ 2. Risk Assessment
    st.subheader("2. Risk Assessment")
    risk_cols = st.columns(4)
    risk_cols[0].metric("Risk Score", f"{r.get('risk_score', '?')} / 100")
    risk_cols[1].markdown(f"**Risk Level:** {_risk_span(r.get('risk_level', ''))}", unsafe_allow_html=True)
    risk_cols[2].markdown(f"**Decision:** {_decision_badge(r.get('decision', ''))}", unsafe_allow_html=True)
    risk_cols[3].metric("Assessment", r.get("overall_assessment", ""))
    reasoning = r.get("reasoning", "")
    if reasoning:
        st.info(reasoning)

    # ─ 3. File-wise Impact
    files = r.get("files", [])
    if files:
        st.subheader("3. File-wise Impact")
        file_data = [{"File": f.get("file", ""), "Summary": f.get("summary", "")} for f in files]
        st.table(file_data)

    # ─ 4. Cross-file Impact
    cross = r.get("cross_file_impact", [])
    if cross:
        st.subheader("4. Cross-file Impact")
        for c in cross:
            st.markdown(f"- **{c.get('component', '')}** — {c.get('impact', '')}")

    # ─ 5. Key Issues
    issues = r.get("issues", [])
    if issues:
        st.subheader("5. Key Issues Found")
        for i, iss in enumerate(issues, 1):
            sev = iss.get("severity", "Medium")
            icon = _severity_icon(sev)
            with st.expander(f"{icon} Issue {i}: [{sev}] {iss.get('file', '')} (line {iss.get('line', '?')})"):
                st.markdown(f"**Issue:** {iss.get('issue', '')}")
                st.markdown(f"**Risk:** {iss.get('risk', '')}")
                affected = iss.get("affected_related_code", [])
                if affected:
                    st.markdown("**Affected related code:** " + ", ".join(f"`{a}`" for a in affected))
                st.markdown(f"**Suggestion:** {iss.get('suggestion', '')}")
                code = iss.get("suggested_code", "")
                if code:
                    st.code(code, language="python")

    # ─ 6. Good Improvements
    goods = r.get("good_improvements", [])
    if goods:
        st.subheader("6. Good Improvements")
        for g in goods:
            st.markdown(f"- ✅ {g}")

    # ─ 7. Bad Regressions
    bads = r.get("bad_regressions", [])
    if bads:
        st.subheader("7. Bad Regressions")
        for b in bads:
            st.markdown(f"- ❌ {b}")

    # ─ 8. Recommended Actions
    actions = r.get("recommended_actions", [])
    if actions:
        st.subheader("8. Recommended Actions Before Merge")
        for a in actions:
            st.markdown(f"- {a}")

    # ─ Timestamp
    st.markdown("---")
    st.caption(f"Review generated at {r.get('created_at', 'N/A')}")


# ── Main ─────────────────────────────────────────────────────────────────────

def render_streamlit_dashboard() -> None:
    filters = _sidebar_filters()

    storage = get_storage()
    reviews = storage.load_review_results(filters if filters else None)

    # Session-state for selected review
    if "selected_review_idx" not in st.session_state:
        st.session_state.selected_review_idx = None

    selected = _render_review_list(reviews)
    if selected is not None:
        st.session_state.selected_review_idx = selected

    idx = st.session_state.selected_review_idx
    if idx is not None and 0 <= idx < len(reviews):
        _render_detail(reviews[idx])


# ── Entrypoint ───────────────────────────────────────────────────────────────

render_streamlit_dashboard()
