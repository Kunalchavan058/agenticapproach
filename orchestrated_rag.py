"""
Orchestrated RAG with explicit planner, reviewer, synthesizer, and executor stages.

This strategy separates the LLM roles from orchestration code:
- Planner brain: creates targeted search tasks.
- Executor workers: run Azure AI Search retrieval for each task.
- Reviewer brain: decides whether follow-up searches are needed.
- Synthesizer brain: writes the final answer from collected evidence.
The LLM stages use the Azure OpenAI Responses API.
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI, BadRequestError, NotFoundError

load_dotenv(override=False)

SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_AI_SEARCH_KEY"]
OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_RESPONSES_ENDPOINT", os.environ["AZURE_OPENAI_ENDPOINT"])
OPENAI_KEY = os.environ.get("AZURE_OPENAI_RESPONSES_KEY", os.environ["AZURE_OPENAI_KEY"])
EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT", OPENAI_ENDPOINT)
EMBEDDING_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_KEY", OPENAI_KEY)
EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", os.environ["AZURE_AI_EMBEDDING_MODEL"])
CHAT_MODEL = os.environ.get("AZURE_OPENAI_RESPONSES_DEPLOYMENT", os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"])
PLANNER_MODEL = os.environ.get("AZURE_OPENAI_PLANNER_DEPLOYMENT", os.environ.get("AZURE_OPENAI_SMALL_MODEL_DEPLOYMENT", CHAT_MODEL))
SYNTHESIS_MODEL = os.environ.get("AZURE_OPENAI_SYNTHESIS_DEPLOYMENT", CHAT_MODEL)

DEFAULT_INDEX_NAME = "annual-reports-index"
TOP_K = 8
MAX_PLAN_TASKS = 4
MAX_EXECUTOR_ROUNDS = 2
MAX_PARALLEL_EXECUTORS = 10

PLANNER_INSTRUCTIONS = (
    "You are the planner brain for a document-analysis system with a search tool over indexed documents. "
    "Your job is only to create the smallest useful retrieval plan. Do not answer the user question.\n\n"
    "PLANNING RULES:\n"
    "1. Decompose the user request into the exact information needs that must be retrieved.\n"
    "2. Produce targeted search tasks, each with one query that combines subject, attribute, and any important constraint.\n"
    "3. Use source_filter only when the exact source file name is already known from the user request or prior evidence.\n"
    "4. Keep the plan compact. Prefer 1-4 tasks. Do not exceed 4 tasks.\n"
    "5. If prior evidence already covers an item, do not create a duplicate task for it.\n"
    "6. If information is still missing after a prior round, create only the smallest delta plan needed to close the gaps.\n"
    "7. Return JSON only."
)

REVIEWER_INSTRUCTIONS = (
    "You are the reviewer brain for a document-analysis system. You receive the user question, the completed search tasks, "
    "and the retrieved evidence summary. Decide whether the system has enough evidence to answer.\n\n"
    "REVIEW RULES:\n"
    "1. Mark enough_information true only if every requested sub-question can be answered from retrieved evidence.\n"
    "2. If evidence is missing, name the missing items and create at most 3 focused follow-up tasks.\n"
    "3. Do not ask for broad retries. Follow-up tasks must be meaningfully different from prior failed searches.\n"
    "4. Return JSON only."
)

SYNTHESIZER_INSTRUCTIONS = (
    "You are a general-purpose document analysis assistant working only from retrieved evidence.\n\n"
    "ANSWER RULES:\n"
    "1. Answer only from the provided evidence packets. Do not use outside knowledge.\n"
    "2. When the answer involves structured comparisons or numeric values, prefer a markdown table. Otherwise use the format that best fits the request.\n"
    "3. Use numbered references like [1], [2], [3] in the answer text to cite evidence packets.\n"
    "4. At the end of your answer, add a 'References' section listing each number with its source file and page when available.\n"
    "5. If information is not found after searching, say so explicitly and identify what was not found.\n"
    "6. Do not estimate, infer, back-calculate, or approximate missing values.\n"
    "7. Preserve the user's requested schema as closely as possible.\n"
    "8. Do not claim facts that are not supported by the retrieved evidence."
)

_responses_client = AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2025-03-01-preview",
)

_embedding_client = AzureOpenAI(
    azure_endpoint=EMBEDDING_ENDPOINT,
    api_key=EMBEDDING_KEY,
    api_version="2024-12-01-preview",
)

_executor_log: list[dict] = []
_executor_log_lock = Lock()
_executor_round_durations: list[float] = []


def _get_search_client(index_name: str) -> SearchClient:
    return SearchClient(SEARCH_ENDPOINT, index_name, AzureKeyCredential(SEARCH_KEY))


def _embed(text: str) -> list[float]:
    try:
        response = _embedding_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except NotFoundError as exc:
        raise RuntimeError(
            "Embedding deployment not found. Configure AZURE_OPENAI_EMBEDDING_DEPLOYMENT "
            "and optionally AZURE_OPENAI_EMBEDDING_ENDPOINT / AZURE_OPENAI_EMBEDDING_KEY "
            f"for the deployment that serves embeddings. Current embedding deployment: {EMBEDDING_MODEL}"
        ) from exc


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model did not return a JSON object: {text}")
    return json.loads(text[start:end + 1])


def _extract_output_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text.strip()

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "output_text" and getattr(content, "text", None):
                parts.append(content.text)

    return "\n".join(parts).strip()


def _create_response(**kwargs):
    try:
        return _responses_client.responses.create(**kwargs)
    except (NotFoundError, BadRequestError) as exc:
        message = str(exc)
        if "Responses API is not enabled in this region" in message or "Responses API is enabled only" in message:
            raise RuntimeError(
                "Responses API could not run on the current Azure OpenAI resource. "
                f"Configured endpoint: {OPENAI_ENDPOINT}. Azure reported that the Responses API is unavailable "
                "for this resource or region. Configure AZURE_OPENAI_RESPONSES_ENDPOINT / "
                "AZURE_OPENAI_RESPONSES_KEY / AZURE_OPENAI_RESPONSES_DEPLOYMENT for a supported resource."
            ) from exc
        raise


def _create_json_completion(model: str, system_prompt: str, user_payload: dict) -> dict:
    response = _create_response(
        model=model,
        instructions=system_prompt,
        input=[
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
        ],
        temperature=0.2,
    )
    content = _extract_output_text(response) or "{}"
    return _extract_json_object(content)


def _search_documents(
    query: str,
    top_k: int = TOP_K,
    source_filter: str | None = None,
    index_name: str = DEFAULT_INDEX_NAME,
) -> tuple[str, list[dict], list[str]]:
    started = time.time()
    safe_top_k = max(1, min(int(top_k), TOP_K))
    try:
        search_client = _get_search_client(index_name)
        query_embedding = _embed(query)
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=safe_top_k,
            fields="embedding",
        )

        filter_expr = None
        if source_filter:
            filter_expr = f"source_file eq '{source_filter}'"

        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=safe_top_k,
            filter=filter_expr,
            select=["content", "source_file", "page_number"],
        )

        output_parts: list[str] = []
        evidence_packets: list[dict] = []
        sources: list[str] = []
        for index, result in enumerate(results, 1):
            packet = {
                "source_file": result["source_file"],
                "page_number": result["page_number"],
                "content": result["content"],
            }
            evidence_packets.append(packet)
            output_parts.append(
                f"[Result {index} | {packet['source_file']} | Page {packet['page_number']}]\n{packet['content']}"
            )
            sources.append(f"{packet['source_file']} (p.{packet['page_number']})")

        if not output_parts:
            output_text = "No results found for this query."
        else:
            output_text = "\n\n---\n\n".join(output_parts)

        return output_text, evidence_packets, sources
    except Exception as exc:
        return f"Search error: {exc}", [], []
    finally:
        duration = round(time.time() - started, 2)
        with _executor_log_lock:
            _executor_log.append(
                {
                    "query": query,
                    "source_filter": source_filter,
                    "index_name": index_name,
                    "duration": duration,
                }
            )


def _normalize_task(task: dict, position: int) -> dict:
    return {
        "task_id": str(task.get("task_id") or f"task-{position}"),
        "objective": str(task.get("objective") or task.get("purpose") or f"Search task {position}"),
        "query": str(task.get("query") or "").strip(),
        "top_k": max(1, min(int(task.get("top_k", TOP_K)), TOP_K)),
        "source_filter": task.get("source_filter") or None,
    }


def _plan_search_tasks(query: str, previous_rounds: list[dict]) -> dict:
    payload = {
        "user_question": query,
        "prior_rounds": previous_rounds,
        "required_output_schema": {
            "plan_summary": "short string",
            "tasks": [
                {
                    "task_id": "task-1",
                    "objective": "what this task is trying to retrieve",
                    "query": "targeted search query",
                    "top_k": TOP_K,
                    "source_filter": None,
                }
            ],
        },
    }
    plan = _create_json_completion(PLANNER_MODEL, PLANNER_INSTRUCTIONS, payload)
    raw_tasks = plan.get("tasks", [])
    tasks = []
    for position, raw_task in enumerate(raw_tasks[:MAX_PLAN_TASKS], 1):
        task = _normalize_task(raw_task, position)
        if task["query"]:
            tasks.append(task)
    return {
        "plan_summary": str(plan.get("plan_summary") or "Targeted retrieval plan"),
        "tasks": tasks,
    }


def _review_progress(query: str, round_history: list[dict], evidence_packets: list[dict]) -> dict:
    payload = {
        "user_question": query,
        "round_history": round_history,
        "evidence_summary": [
            {
                "evidence_id": packet["evidence_id"],
                "task_id": packet["task_id"],
                "source_file": packet["source_file"],
                "page_number": packet["page_number"],
                "snippet": packet["content"][:300],
            }
            for packet in evidence_packets
        ],
        "required_output_schema": {
            "enough_information": True,
            "missing_information": ["missing item"],
            "follow_up_tasks": [
                {
                    "task_id": "task-followup-1",
                    "objective": "what remains missing",
                    "query": "meaningfully different follow-up query",
                    "top_k": TOP_K,
                    "source_filter": None,
                }
            ],
        },
    }
    review = _create_json_completion(PLANNER_MODEL, REVIEWER_INSTRUCTIONS, payload)
    follow_up_tasks = []
    for position, raw_task in enumerate(review.get("follow_up_tasks", [])[:MAX_PLAN_TASKS], 1):
        task = _normalize_task(raw_task, position)
        if task["query"]:
            follow_up_tasks.append(task)
    return {
        "enough_information": bool(review.get("enough_information")),
        "missing_information": [str(item) for item in review.get("missing_information", []) if str(item).strip()],
        "follow_up_tasks": follow_up_tasks,
    }


def _execute_task(task: dict, index_name: str, round_number: int) -> dict:
    started = time.time()
    output_text, packets, sources = _search_documents(
        query=task["query"],
        top_k=task["top_k"],
        source_filter=task["source_filter"],
        index_name=index_name,
    )
    duration = round(time.time() - started, 2)
    error = output_text if output_text.startswith("Search error:") else None
    evidence_packets = []
    for index, packet in enumerate(packets, 1):
        evidence_packets.append(
            {
                "evidence_id": None,
                "task_id": task["task_id"],
                "source_file": packet["source_file"],
                "page_number": packet["page_number"],
                "content": packet["content"],
            }
        )
    return {
        "task_id": task["task_id"],
        "objective": task["objective"],
        "query": task["query"],
        "source_filter": task["source_filter"],
        "top_k": task["top_k"],
        "round": round_number,
        "results_count": len(evidence_packets),
        "sources": sources,
        "output_text": output_text,
        "duration": duration,
        "error": error,
        "evidence_packets": evidence_packets,
    }


def _run_executor_round(tasks: list[dict], index_name: str, round_number: int) -> list[dict]:
    started = time.time()
    ordered_results: list[dict | None] = [None] * len(tasks)
    if len(tasks) == 1:
        ordered_results[0] = _execute_task(tasks[0], index_name, round_number)
    else:
        max_workers = min(len(tasks), MAX_PARALLEL_EXECUTORS)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_execute_task, task, index_name, round_number): position
                for position, task in enumerate(tasks)
            }
            for future in as_completed(future_map):
                ordered_results[future_map[future]] = future.result()

    _executor_round_durations.append(round(time.time() - started, 2))
    return [result for result in ordered_results if result is not None]


def _assign_evidence_ids(execution_results: list[dict], start_index: int) -> tuple[list[dict], int]:
    next_index = start_index
    all_packets: list[dict] = []
    for execution in execution_results:
        for packet in execution["evidence_packets"]:
            packet["evidence_id"] = next_index
            next_index += 1
            all_packets.append(packet)
    return all_packets, next_index


def _build_synthesis_input(query: str, evidence_packets: list[dict], missing_information: list[str]) -> list[dict]:
    evidence_blocks = []
    for packet in evidence_packets:
        evidence_blocks.append(
            f"[Evidence {packet['evidence_id']} | {packet['source_file']} | Page {packet['page_number']} | {packet['task_id']}]\n"
            f"{packet['content']}"
        )
    evidence_text = "\n\n---\n\n".join(evidence_blocks) if evidence_blocks else "No evidence retrieved."

    prompt = (
        f"Evidence Packets:\n{evidence_text}\n\n---\n\n"
        f"Question: {query}\n\n"
        "Write the final answer now."
    )
    if missing_information:
        prompt += (
            "\n\nReviewer unresolved items: "
            + "; ".join(missing_information)
            + ". Explicitly mark those items as not found if the evidence does not support them."
        )

    return [{"role": "user", "content": prompt}]


def _synthesize_answer(query: str, evidence_packets: list[dict], review: dict) -> str:
    response = _create_response(
        model=SYNTHESIS_MODEL,
        instructions=SYNTHESIZER_INSTRUCTIONS,
        input=_build_synthesis_input(query, evidence_packets, review.get("missing_information", [])),
        temperature=0.2,
    )
    return _extract_output_text(response) or "No answer generated."


def _run_orchestrated_rag(query: str, index_name: str = DEFAULT_INDEX_NAME) -> dict:
    global _executor_log, _executor_round_durations
    _executor_log = []
    _executor_round_durations = []

    started = time.time()
    round_history: list[dict] = []
    execution_log: list[dict] = []
    evidence_packets: list[dict] = []
    next_evidence_id = 1
    llm_calls = 0

    plan = _plan_search_tasks(query, previous_rounds=[])
    llm_calls += 1
    current_tasks = plan["tasks"]
    round_history.append({
        "round": 1,
        "plan_summary": plan["plan_summary"],
        "tasks": current_tasks,
    })

    review = {
        "enough_information": False,
        "missing_information": [],
        "follow_up_tasks": [],
    }

    round_number = 1
    while current_tasks and round_number <= MAX_EXECUTOR_ROUNDS:
        execution_results = _run_executor_round(current_tasks, index_name, round_number)
        packets, next_evidence_id = _assign_evidence_ids(execution_results, next_evidence_id)
        evidence_packets.extend(packets)
        execution_log.extend(execution_results)

        review = _review_progress(query, round_history, evidence_packets)
        llm_calls += 1
        round_history[-1]["review"] = {
            "enough_information": review["enough_information"],
            "missing_information": review["missing_information"],
        }

        if review["enough_information"] or round_number >= MAX_EXECUTOR_ROUNDS:
            break

        current_tasks = review["follow_up_tasks"]
        round_number += 1
        round_history.append({
            "round": round_number,
            "plan_summary": "Reviewer follow-up plan",
            "tasks": current_tasks,
        })

    answer = _synthesize_answer(query, evidence_packets, review)
    llm_calls += 1

    unique_sources: list[str] = []
    seen_sources: set[str] = set()
    tool_calls: list[dict] = []
    for execution in execution_log:
        tool_calls.append(
            {
                "task_id": execution["task_id"],
                "objective": execution["objective"],
                "query": execution["query"],
                "source_filter": execution["source_filter"],
                "results_count": execution["results_count"],
                "sources": execution["sources"],
                "duration": execution["duration"],
                "round": execution["round"],
                "error": execution["error"],
            }
        )
        for source in execution["sources"]:
            if source not in seen_sources:
                seen_sources.add(source)
                unique_sources.append(source)

    total_time = time.time() - started
    search_time = round(sum(_executor_round_durations), 2)
    generation_time = round(max(total_time - search_time, 0), 2)

    return {
        "answer": answer,
        "chunks_retrieved": len(evidence_packets),
        "search_calls": len(tool_calls),
        "sources": unique_sources,
        "search_time": search_time,
        "generation_time": generation_time,
        "total_time": round(total_time, 2),
        "tool_calls": tool_calls,
        "llm_calls": llm_calls,
        "strategy": "Orchestrated RAG",
        "plan_rounds": round_history,
        "gaps": review.get("missing_information", []),
        "available": True,
    }


def ask_with_metadata(query: str, index_name: str = DEFAULT_INDEX_NAME) -> dict:
    return _run_orchestrated_rag(query, index_name=index_name)


def ask(query: str, index_name: str = DEFAULT_INDEX_NAME) -> str:
    return ask_with_metadata(query, index_name=index_name)["answer"]


def main():
    print("=== Orchestrated RAG ===")
    print("Ask questions about the indexed documents. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        print("\nOrchestrator running planner, executors, and synthesizer...\n")
        result = _run_orchestrated_rag(query)
        print(result["answer"])
        print(
            f"\n[{result['search_calls']} executor searches, {result['chunks_retrieved']} chunks, {result['total_time']}s]\n"
        )


if __name__ == "__main__":
    main()