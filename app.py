"""
Streamlit UI for comparing multiple RAG strategies.

Run with: uv run streamlit run app.py
"""

import streamlit as st
import pandas as pd
import re
from pathlib import Path
from rag import ask_with_metadata as normal_rag
from agentic_rag import ask_with_metadata as agentic_rag
from orchestrated_rag import ask_with_metadata as orchestrated_rag
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
st.set_page_config(page_title="RAG Strategy Comparison", layout="wide")
st.title("RAG Strategy Comparison")


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
    result.setdefault("plan_rounds", [])
    result.setdefault("gaps", [])
    result.setdefault("llm_calls", 0)
    result.setdefault("strategy", strategy_name)
    result.setdefault("available", True)
    return result


def render_tool_calls(result: dict, label: str) -> None:
    """Render per-strategy tool call details."""
    with st.expander(label):
        if not result["tool_calls"]:
            st.write("No tool calls recorded.")
            return

        for index, tool_call in enumerate(result["tool_calls"], 1):
            heading = tool_call.get("objective") or tool_call.get("query", "N/A")
            st.markdown(f"**Search {index}:** `{heading}`")
            st.caption(f"Query: `{tool_call.get('query', 'N/A')}`")
            if tool_call.get("task_id"):
                st.caption(f"Task ID: `{tool_call['task_id']}`")
            if tool_call.get("round"):
                st.caption(f"Round: `{tool_call['round']}`")
            if tool_call.get("source_filter"):
                st.caption(f"Filter: `{tool_call['source_filter']}`")
            st.caption(f"Results returned: {tool_call.get('results_count', 0)}")
            if tool_call.get("error"):
                st.error(f"Error: {tool_call['error']}")
            st.divider()


def build_metrics_df(strategy_results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison dataframe for the available strategies."""
    rows = {
        "Search Calls": [],
        "Chunks Retrieved": [],
        "LLM Calls": [],
        "Search Time": [],
        "Generation Time": [],
        "Total Time": [],
    }

    data = {"Metric": list(rows.keys())}
    for label, result in strategy_results.items():
        data[label] = [
            result.get("search_calls", 0),
            result.get("chunks_retrieved", 0),
            result.get("llm_calls", 0),
            f"{result.get('search_time', 0.0)}s",
            f"{result.get('generation_time', 0.0)}s",
            f"{result.get('total_time', 0.0)}s",
        ]
    return pd.DataFrame(data)


def build_tool_rows(result: dict) -> list[dict]:
    """Convert tool call metadata into a dataframe-friendly list."""
    rows = []
    for index, tool_call in enumerate(result["tool_calls"], 1):
        rows.append({
            "#": index,
            "Round": tool_call.get("round") or "—",
            "Task": tool_call.get("task_id") or "—",
            "Objective": tool_call.get("objective") or "—",
            "Query": tool_call.get("query", "N/A"),
            "Source Filter": tool_call.get("source_filter") or "—",
            "Results": tool_call.get("results_count", 0),
        })
    return rows


def build_source_rows(result: dict) -> list[dict]:
    return [{"#": index, "Source": source} for index, source in enumerate(result["sources"], 1)]

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
        [
            "Compare All",
            "Normal RAG only",
            "Agentic RAG only",
            "Orchestrated RAG only",
            "Responses API only",
        ],
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
    orchestrated_result = None
    responses_result = None

    # Run the selected mode(s)
    if run_mode in ["Compare All", "Normal RAG only"]:
        with st.spinner("Running Normal RAG..."):
            rag_result = run_strategy("Normal RAG", normal_rag, active_question, active_index_name)

    if run_mode in ["Compare All", "Agentic RAG only"]:
        with st.spinner("Running Agentic RAG (agent is thinking & searching)..."):
            agent_result = run_strategy("Agentic RAG", agentic_rag, active_question, active_index_name)

    if run_mode in ["Compare All", "Orchestrated RAG only"]:
        with st.spinner("Running Orchestrated RAG (planner, executors, reviewer, synthesizer)..."):
            orchestrated_result = run_strategy("Orchestrated RAG", orchestrated_rag, active_question, active_index_name)

    if run_mode in ["Compare All", "Responses API only"]:
        with st.spinner("Running Responses API RAG..."):
            responses_result = run_strategy("Responses API RAG", responses_rag, active_question, active_index_name)

    # Save to per-question cache (overrides any previous run for this question)
    st.session_state["results_cache"][active_question] = {
        "rag_result": rag_result,
        "agent_result": agent_result,
        "orchestrated_result": orchestrated_result,
        "responses_result": responses_result,
        "run_mode": run_mode,
    }

# Load results for the current question from cache
cached = st.session_state["results_cache"].get(active_question, {})
rag_result = cached.get("rag_result")
agent_result = cached.get("agent_result")
orchestrated_result = cached.get("orchestrated_result")
responses_result = cached.get("responses_result")
run_mode = cached.get("run_mode")

if cached:
    st.caption("Showing cached results. Click **Run** to re-run.")

if rag_result or agent_result or orchestrated_result or responses_result:
    # --- Tabs ---
    tab_names = []
    if rag_result:
        tab_names.append("Normal RAG Answer")
    if agent_result:
        tab_names.append("Agentic RAG Answer")
    if orchestrated_result:
        tab_names.append("Orchestrated RAG Answer")
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
            render_tool_calls(agent_result, f"Agent Tool Calls ({agent_result['search_calls']} searches)")
        tab_idx += 1

    # --- Orchestrated RAG Answer Tab ---
    if orchestrated_result:
        with tabs[tab_idx]:
            if orchestrated_result.get("available") is False and orchestrated_result.get("error"):
                st.warning(orchestrated_result["error"])
            st.markdown(orchestrated_result["answer"])
            if orchestrated_result["plan_rounds"]:
                with st.expander("Planner / Reviewer Analysis"):
                    for round_info in orchestrated_result["plan_rounds"]:
                        st.markdown(f"**Round {round_info['round']}:** {round_info['plan_summary']}")
                        if round_info.get("tasks"):
                            task_df = pd.DataFrame(round_info["tasks"])
                            st.dataframe(task_df, use_container_width=True, hide_index=True)
                        review = round_info.get("review")
                        if review:
                            st.caption(f"Enough information: {review['enough_information']}")
                            if review.get("missing_information"):
                                st.write("Missing information:")
                                for item in review["missing_information"]:
                                    st.write(f"- {item}")
            render_tool_calls(orchestrated_result, f"Executor Calls ({orchestrated_result['search_calls']} searches)")
        tab_idx += 1

    # --- Responses API Answer Tab ---
    if responses_result:
        with tabs[tab_idx]:
            if responses_result.get("available") is False and responses_result.get("error"):
                st.warning(responses_result["error"])
            st.markdown(responses_result["answer"])
            render_tool_calls(responses_result, f"Responses Tool Calls ({responses_result['search_calls']} searches)")
            with st.expander("Sources Used"):
                for src in responses_result["sources"]:
                    st.write(f"- {src}")
        tab_idx += 1

    # --- Analytics Tab ---
    with tabs[tab_idx]:
        strategy_results = {}
        if rag_result:
            strategy_results["Normal RAG"] = rag_result
        if agent_result:
            strategy_results["Agentic RAG"] = agent_result
        if orchestrated_result:
            strategy_results["Orchestrated RAG"] = orchestrated_result
        if responses_result:
            strategy_results["Responses API"] = responses_result

        if len(strategy_results) > 1:
            st.subheader("Performance Comparison")
            st.dataframe(
                build_metrics_df(strategy_results),
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            for label, result in strategy_results.items():
                st.subheader(f"{label} Search Activity")
                tool_rows = build_tool_rows(result)
                if tool_rows:
                    st.dataframe(pd.DataFrame(tool_rows), use_container_width=True, hide_index=True)
                else:
                    st.write("No tool calls recorded.")

                if label == "Orchestrated RAG" and result.get("gaps"):
                    st.caption("Reviewer gaps after final round")
                    for gap in result["gaps"]:
                        st.write(f"- {gap}")

                st.subheader(f"{label} Sources")
                st.dataframe(pd.DataFrame(build_source_rows(result)), use_container_width=True, hide_index=True)
                st.divider()

        else:
            only_label, only_result = next(iter(strategy_results.items()))
            st.subheader(f"{only_label} Metrics")
            st.dataframe(build_metrics_df({only_label: only_result}), use_container_width=True, hide_index=True)
            st.divider()
            st.subheader(f"{only_label} Search Activity")
            tool_rows = build_tool_rows(only_result)
            if tool_rows:
                st.dataframe(pd.DataFrame(tool_rows), use_container_width=True, hide_index=True)
            else:
                st.write("No tool calls recorded.")
            st.divider()
            st.subheader("Sources")
            st.dataframe(pd.DataFrame(build_source_rows(only_result)), use_container_width=True, hide_index=True)
