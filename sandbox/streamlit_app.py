"""Streamlit demo for the competition ranking pipeline.

Run with:
    streamlit run sandbox/streamlit_app.py

This app loads a sample of candidates from the JSONL, runs the
deterministic ranking pipeline, and displays results interactively.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ranker.io import load_candidates_jsonl
from src.ranker.score import score_all, select_top_n
from src.ranker.honeypot import detect_honeypot

st.set_page_config(
    page_title="RankAI — Competition Demo",
    page_icon="🏆",
    layout="wide",
)

st.title("🏆 RankAI — Candidate Ranking Demo")
st.markdown(
    "Deterministic, zero-LLM candidate ranking for the **Senior AI Engineer** role at Redrob AI. "
    "Processes 100K candidates in ~37 seconds on CPU."
)

# --- Sidebar ---
st.sidebar.header("Configuration")
max_candidates = st.sidebar.slider(
    "Candidates to load", min_value=100, max_value=10000, value=1000, step=100
)
top_n = st.sidebar.slider(
    "Top N to display", min_value=10, max_value=200, value=100, step=10
)

# --- Data loading ---
DATA_DIR = (
    project_root
    / "indiaruns"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
)
JSONL_PATH = DATA_DIR / "candidates.jsonl"

if not JSONL_PATH.exists():
    st.error(f"Candidates file not found: {JSONL_PATH}")
    st.info("Place candidates.jsonl in the expected location.")
    st.stop()

# --- Run pipeline ---
with st.spinner("Loading candidates..."):
    candidates = list(load_candidates_jsonl(JSONL_PATH))[:max_candidates]
    st.success(f"Loaded {len(candidates):,} candidates")

with st.spinner("Scoring candidates..."):
    scored = score_all(candidates)
    top = select_top_n(scored, n=top_n)
    st.success(f"Scored {len(scored):,} candidates, selected top {len(top)}")

# --- Tabs ---
tab_rank, tab_dist, tab_honeypot = st.tabs(
    ["📋 Rankings", "📊 Score Distribution", "🍯 Honeypot Analysis"]
)

with tab_rank:
    st.subheader(f"Top {len(top)} Candidates")

    rows = []
    for i, s in enumerate(top, start=1):
        rows.append({
            "Rank": i,
            "Candidate ID": s.candidate_id,
            "Score": round(s.score, 4),
            "Name": s.name or "—",
            "Current Title": s.current_title or "—",
            "Honeypot": "⚠️" if s.is_honeypot else "✅",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="submission_preview.csv",
        mime="text/csv",
    )

with tab_dist:
    st.subheader("Score Distribution")

    all_scores = [s.score for s in scored]
    top_scores = [s.score for s in top]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("All candidates", len(all_scores))
        score_series = pd.Series(all_scores)
        bins = pd.cut(score_series, bins=30).value_counts().sort_index()
        st.bar_chart(bins)
    with col2:
        st.metric("Top candidates", len(top_scores))
        st.metric("Min score (top)", f"{min(top_scores):.4f}")
        st.metric("Max score (top)", f"{max(top_scores):.4f}")
        st.metric("Mean score (all)", f"{score_series.mean():.4f}")
        st.metric("Std dev (all)", f"{score_series.std():.4f}")

with tab_honeypot:
    st.subheader("Honeypot Detection Summary")

    honeypot_count = sum(1 for s in scored if s.is_honeypot)
    safe_count = len(scored) - honeypot_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total candidates", f"{len(scored):,}")
    col2.metric("Safe candidates", f"{safe_count:,}", delta=f"{100*safe_count/len(scored):.1f}%")
    col3.metric("Honeypots detected", f"{honeypot_count:,}", delta=f"{100*honeypot_count/len(scored):.1f}%", delta_color="inverse")

    st.markdown("**Honeypot Detectors:**")
    st.markdown("""
    - 🕐 **Timeline Impossibility**: Overlapping full-time roles
    - 📊 **YOE Span Mismatch**: Career history doesn't cover declared experience
    - 🏅 **Expert Without Endorsements**: AI-heavy skills but no endorsements
    - 📈 **Title Seniority Inflation**: Senior title with <3 years experience
    - 🧩 **Skill Stuffing**: Non-technical role with 6+ AI keywords
    """)

# --- Footer ---
st.divider()
st.caption(
    "RankAI — Deterministic candidate ranking. Zero LLM calls. ~37s for 100K candidates."
)
