"""
Streamlit UI for comparing Normal RAG vs Agentic RAG.

Run with: uv run streamlit run app.py
"""

import streamlit as st
import pandas as pd
import re
from pathlib import Path
from rag import ask_with_metadata as normal_rag
from agentic_rag import ask_with_metadata as agentic_rag
from responses import ask_with_metadata as responses_rag


CATEGORY_INDEX_MAP = {
    "aragque": "aragdoc",
}


def get_index_name_for_category(category: str) -> str:
    """Return the Azure Search index to use for the selected question category."""
    return CATEGORY_INDEX_MAP.get(category, "annual-reports-index")

def load_test_questions(file_path: str) -> dict:
    """Dynamically load questions from test_questions.md."""
    questions = {}
    current_category = "General"
    
    if not Path(file_path).exists():
        return {"Default": ["No questions found in test_questions.md"]}

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by categories (## headers)
    sections = re.split(r'^##\s+', content, flags=re.MULTILINE)
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split("\n")
        category_name = lines[0].strip()
        
        # Look for numbered questions (e.g., 1. Question or 1) Question)
        # Using a regex that handles both single line and multiline questions
        category_content = "\n".join(lines[1:])
        raw_questions = re.split(r'^\d+[\.\)]\s*', category_content, flags=re.MULTILINE)
        
        parsed_questions = [q.strip() for q in raw_questions if q.strip()]
        if parsed_questions:
            questions[category_name] = parsed_questions
            
    return questions if questions else {"Default": ["No questions parsed"]}

# --- Test Questions ---
TEST_QUESTIONS = load_test_questions("test_questions.md")

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="RAG vs Agentic RAG", layout="wide")
st.title("RAG vs Agentic RAG vs Responses API")


def run_strategy(strategy_name: str, runner, question: str, index_name: str) -> dict:
    """Run a strategy and convert runtime failures into UI-friendly results."""
    try:
        result = runner(question, index_name=index_name)
    except Exception as exc:
        return {
            "answer": (
                f"{strategy_name} could not complete.\n\n"
                "This is usually a configuration problem with the Azure OpenAI deployment or endpoint.\n\n"
                f"Error: `{exc}`"
            ),
            "chunks_retrieved": 0,
            "search_calls": 0,
            "sources": [],
            "search_time": 0.0,
            "generation_time": 0.0,
            "total_time": 0.0,
            "tool_calls": [],
            "available": False,
            "error": str(exc),
        }

    result.setdefault("tool_calls", [])
    result.setdefault("sources", [])
    result.setdefault("available", True)
    return result

# Per-question results cache: { question_text: { "rag_result": ..., "agent_result": ..., "run_mode": ... } }
if "results_cache" not in st.session_state:
    st.session_state["results_cache"] = {}

# Sidebar: question selection
with st.sidebar:
    st.header("Question")
    category = st.selectbox("Category", list(TEST_QUESTIONS.keys()))
    question = st.selectbox("Pick a question", TEST_QUESTIONS[category])
    st.divider()
    custom = st.text_area("Or type your own:")
    st.divider()
    run_mode = st.radio(
        "Run Mode",
        ["Compare All", "Normal RAG only", "Agentic RAG only", "Responses API only"],
    )
    run_btn = st.button("Run", type="primary", use_container_width=True)

    # Show how many questions have cached results
    cached_count = len(st.session_state["results_cache"])
    if cached_count:
        st.divider()
        st.caption(f"{cached_count} question(s) cached this session")
        if st.button("Clear all cached results"):
            st.session_state["results_cache"] = {}
            st.rerun()

active_question = custom.strip() if custom.strip() else question
active_index_name = get_index_name_for_category(category)

st.markdown(f"**Question:** {active_question}")
st.caption(f"Using search index: {active_index_name}")

if run_btn:
    rag_result = None
    agent_result = None
    responses_result = None

    # Run the selected mode(s)
    if run_mode in ["Compare All", "Normal RAG only"]:
        with st.spinner("Running Normal RAG..."):
            rag_result = run_strategy("Normal RAG", normal_rag, active_question, active_index_name)

    if run_mode in ["Compare All", "Agentic RAG only"]:
        with st.spinner("Running Agentic RAG (agent is thinking & searching)..."):
            agent_result = run_strategy("Agentic RAG", agentic_rag, active_question, active_index_name)

    if run_mode in ["Compare All", "Responses API only"]:
        with st.spinner("Running Responses API RAG..."):
            responses_result = run_strategy("Responses API RAG", responses_rag, active_question, active_index_name)

    # Save to per-question cache (overrides any previous run for this question)
    st.session_state["results_cache"][active_question] = {
        "rag_result": rag_result,
        "agent_result": agent_result,
        "responses_result": responses_result,
        "run_mode": run_mode,
    }

# Load results for the current question from cache
cached = st.session_state["results_cache"].get(active_question, {})
rag_result = cached.get("rag_result")
agent_result = cached.get("agent_result")
responses_result = cached.get("responses_result")
run_mode = cached.get("run_mode")

if cached:
    st.caption("Showing cached results. Click **Run** to re-run.")

if rag_result or agent_result or responses_result:
    # --- Tabs ---
    tab_names = []
    if rag_result:
        tab_names.append("Normal RAG Answer")
    if agent_result:
        tab_names.append("Agentic RAG Answer")
    if responses_result:
        tab_names.append("Responses API Answer")
    tab_names.append("Analytics & Comparison")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # --- Normal RAG Answer Tab ---
    if rag_result:
        with tabs[tab_idx]:
            if rag_result.get("available") is False and rag_result.get("error"):
                st.warning(rag_result["error"])
            st.markdown(rag_result["answer"])
            with st.expander("Sources Used"):
                for src in rag_result["sources"]:
                    st.write(f"- {src}")
        tab_idx += 1

    # --- Agentic RAG Answer Tab ---
    if agent_result:
        with tabs[tab_idx]:
            if agent_result.get("available") is False and agent_result.get("error"):
                st.warning(agent_result["error"])
            st.markdown(agent_result["answer"])
            with st.expander(f"Agent Tool Calls ({agent_result['search_calls']} searches)"):
                for i, tc in enumerate(agent_result["tool_calls"], 1):
                    st.markdown(f"**Search {i}:** `{tc.get('query', 'N/A')}`")
                    if tc.get("source_filter"):
                        st.caption(f"Filter: `{tc['source_filter']}`")
                    st.caption(f"Results returned: {tc.get('results_count', 0)}")
                    if tc.get("error"):
                        st.error(f"Error: {tc['error']}")
                    st.divider()
        tab_idx += 1

    # --- Responses API Answer Tab ---
    if responses_result:
        with tabs[tab_idx]:
            if responses_result.get("available") is False and responses_result.get("error"):
                st.warning(responses_result["error"])
            st.markdown(responses_result["answer"])
            with st.expander("Sources Used"):
                for src in responses_result["sources"]:
                    st.write(f"- {src}")
        tab_idx += 1

    # --- Analytics Tab ---
    with tabs[tab_idx]:
        if rag_result and agent_result and responses_result:
            st.subheader("Performance Comparison")

            comparison_df = pd.DataFrame({
                "Metric": [
                    "Search Calls",
                    "Chunks Retrieved",
                    "Search Time",
                    "Generation Time",
                    "Total Time",
                ],
                "Normal RAG": [
                    rag_result["search_calls"],
                    rag_result["chunks_retrieved"],
                    f"{rag_result['search_time']}s",
                    f"{rag_result['generation_time']}s",
                    f"{rag_result['total_time']}s",
                ],
                "Agentic RAG": [
                    agent_result["search_calls"],
                    agent_result["chunks_retrieved"],
                    f"{agent_result['search_time']}s",
                    f"{agent_result['generation_time']}s",
                    f"{agent_result['total_time']}s",
                ],
                "Responses API": [
                    responses_result["search_calls"],
                    responses_result["chunks_retrieved"],
                    f"{responses_result['search_time']}s",
                    f"{responses_result['generation_time']}s",
                    f"{responses_result['total_time']}s",
                ],
            })

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            st.subheader("Agent Search Queries")
            if agent_result["tool_calls"]:
                tool_rows = []
                for i, tc in enumerate(agent_result["tool_calls"], 1):
                    tool_rows.append({
                        "#": i,
                        "Query": tc.get("query", "N/A"),
                        "Source Filter": tc.get("source_filter") or "—",
                        "Results": tc.get("results_count", 0),
                    })
                st.dataframe(
                    pd.DataFrame(tool_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("No tool calls recorded.")

            st.divider()

            st.subheader("Normal RAG Sources")
            source_rows = [{"#": i, "Source": src} for i, src in enumerate(rag_result["sources"], 1)]
            st.dataframe(
                pd.DataFrame(source_rows),
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            st.subheader("Responses API Sources")
            response_source_rows = [{"#": i, "Source": src} for i, src in enumerate(responses_result["sources"], 1)]
            st.dataframe(
                pd.DataFrame(response_source_rows),
                use_container_width=True,
                hide_index=True,
            )

        elif rag_result and agent_result:
            # --- Comparison Table ---
            st.subheader("Performance Comparison")

            def _fmt_delta(val, unit=""):
                sign = "+" if val >= 0 else ""
                return f"{sign}{val}{unit}"

            diff_chunks = agent_result["chunks_retrieved"] - rag_result["chunks_retrieved"]
            diff_calls = agent_result["search_calls"] - rag_result["search_calls"]
            diff_time = round(agent_result["total_time"] - rag_result["total_time"], 2)

            comparison_df = pd.DataFrame({
                "Metric": [
                    "Search Calls",
                    "Chunks Retrieved",
                    "Search Time",
                    "Generation Time",
                    "Total Time",
                ],
                "Normal RAG": [
                    rag_result["search_calls"],
                    rag_result["chunks_retrieved"],
                    f"{rag_result['search_time']}s",
                    f"{rag_result['generation_time']}s",
                    f"{rag_result['total_time']}s",
                ],
                "Agentic RAG": [
                    agent_result["search_calls"],
                    agent_result["chunks_retrieved"],
                    f"{agent_result['search_time']}s",
                    f"{agent_result['generation_time']}s",
                    f"{agent_result['total_time']}s",
                ],
                "Difference": [
                    _fmt_delta(diff_calls),
                    _fmt_delta(diff_chunks),
                    _fmt_delta(round(agent_result["search_time"] - rag_result["search_time"], 2), "s"),
                    _fmt_delta(round(agent_result["generation_time"] - rag_result["generation_time"], 2), "s"),
                    _fmt_delta(diff_time, "s"),
                ],
            })

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            # --- Agent Search Queries ---
            st.subheader("Agent Search Queries")
            if agent_result["tool_calls"]:
                tool_rows = []
                for i, tc in enumerate(agent_result["tool_calls"], 1):
                    tool_rows.append({
                        "#": i,
                        "Query": tc.get("query", "N/A"),
                        "Source Filter": tc.get("source_filter") or "—",
                        "Results": tc.get("results_count", 0),
                    })
                st.dataframe(
                    pd.DataFrame(tool_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.write("No tool calls recorded.")

            st.divider()

            # --- Normal RAG Sources ---
            st.subheader("Normal RAG Sources")
            source_rows = [{"#": i, "Source": src} for i, src in enumerate(rag_result["sources"], 1)]
            st.dataframe(
                pd.DataFrame(source_rows),
                use_container_width=True,
                hide_index=True,
            )

        elif responses_result:
            st.subheader("Responses API Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Chunks Retrieved", "Search Calls", "Total Time"],
                "Value": [responses_result["chunks_retrieved"], responses_result["search_calls"], f"{responses_result['total_time']}s"],
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            st.divider()
            st.subheader("Sources")
            response_source_rows = [{"#": i, "Source": src} for i, src in enumerate(responses_result["sources"], 1)]
            st.dataframe(pd.DataFrame(response_source_rows), use_container_width=True, hide_index=True)

        elif rag_result:
            st.subheader("Normal RAG Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Chunks Retrieved", "Search Calls", "Total Time"],
                "Value": [rag_result["chunks_retrieved"], rag_result["search_calls"], f"{rag_result['total_time']}s"],
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            st.divider()
            st.subheader("Sources")
            source_rows = [{"#": i, "Source": src} for i, src in enumerate(rag_result["sources"], 1)]
            st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)

        elif agent_result:
            st.subheader("Agentic RAG Metrics")
            metrics_df = pd.DataFrame({
                "Metric": ["Chunks Retrieved", "Search Calls", "Total Time"],
                "Value": [agent_result["chunks_retrieved"], agent_result["search_calls"], f"{agent_result['total_time']}s"],
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            st.divider()
            st.subheader("Agent Search Queries")
            if agent_result["tool_calls"]:
                tool_rows = []
                for i, tc in enumerate(agent_result["tool_calls"], 1):
                    tool_rows.append({
                        "#": i,
                        "Query": tc.get("query", "N/A"),
                        "Source Filter": tc.get("source_filter") or "—",
                        "Results": tc.get("results_count", 0),
                    })
                st.dataframe(pd.DataFrame(tool_rows), use_container_width=True, hide_index=True)
