"""Streamlit demo for the competition ranking pipeline.

Run with:
    streamlit run sandbox/streamlit_app.py

This app accepts a candidates.jsonl file upload (or uses a local path
for development), runs the deterministic ranking pipeline, and displays
results interactively.
"""

import tempfile
import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ranker.io import load_candidates_jsonl
from src.ranker.score import score_all, select_top_n

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

# --- File Upload ---
st.sidebar.header("📁 Data Source")

# Try local path first, fall back to upload
LOCAL_JSONL = (
    project_root
    / "indiaruns"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "candidates.jsonl"
)

jsonl_path = None

if LOCAL_JSONL.exists():
    st.sidebar.success(f"Found local file ({LOCAL_JSONL.stat().st_size / 1e6:.0f} MB)")
    jsonl_path = LOCAL_JSONL
else:
    st.sidebar.info("No local file found — upload candidates.jsonl")
    uploaded = st.sidebar.file_uploader(
        "Upload candidates.jsonl",
        type=["jsonl", "json"],
        help="Upload the 465MB candidates.jsonl from the challenge dataset",
    )
    if uploaded is not None:
        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
        tmp.write(uploaded.read())
        tmp.close()
        jsonl_path = Path(tmp.name)
        st.sidebar.success(f"Uploaded: {uploaded.name} ({uploaded.size / 1e6:.0f} MB)")

# --- Sidebar Config ---
st.sidebar.header("⚙️ Configuration")
max_candidates = st.sidebar.slider(
    "Candidates to load", min_value=100, max_value=100000, value=10000, step=1000
)
top_n = st.sidebar.slider(
    "Top N to display", min_value=10, max_value=200, value=100, step=10
)

# --- Run pipeline ---
if jsonl_path is None:
    st.warning("👆 Upload candidates.jsonl to get started.")
    st.stop()

with st.spinner("Loading candidates..."):
    candidates = list(load_candidates_jsonl(jsonl_path))[:max_candidates]
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
        label="📥 Download Top Candidates CSV",
        data=csv,
        file_name="rankai_top_candidates.csv",
        mime="text/csv",
    )

with tab_dist:
    st.subheader("Score Distribution")

    all_scores = [s.score for s in scored]
    top_scores = [s.score for s in top]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("All candidates", f"{len(all_scores):,}")
        score_series = pd.Series(all_scores)
        hist_data = score_series.round(3).value_counts().sort_index()
        chart_df = pd.DataFrame({"score": hist_data.index.astype(str), "count": hist_data.values})
        st.bar_chart(chart_df.set_index("score"))
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
