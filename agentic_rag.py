"""
Agentic RAG using OpenAI tool calling.

The LLM has a search tool it can call multiple times with different queries.
It autonomously decides how to decompose complex questions, search iteratively,
and synthesize a complete answer from all retrieved information.
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
from openai import AzureOpenAI, NotFoundError

load_dotenv(override=False)

# --- Configuration ---
SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_AI_SEARCH_KEY"]
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT", OPENAI_ENDPOINT)
EMBEDDING_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_KEY", OPENAI_KEY)
EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", os.environ["AZURE_AI_EMBEDDING_MODEL"])
CHAT_MODEL = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]
DEFAULT_INDEX_NAME = "annual-reports-index"
TOP_K = 8
MAX_TOOL_ROUNDS = 8
MAX_PARALLEL_SEARCHES = 4
MAX_STALLED_ROUNDS = 2

# --- Clients ---
_chat_client = AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-12-01-preview",
)

_embedding_client = AzureOpenAI(
    azure_endpoint=EMBEDDING_ENDPOINT,
    api_key=EMBEDDING_KEY,
    api_version="2024-12-01-preview",
)



def _get_search_client(index_name: str) -> SearchClient:
    """Create a search client for the requested index."""
    return SearchClient(
        SEARCH_ENDPOINT, index_name, AzureKeyCredential(SEARCH_KEY)
    )


def _embed(text: str) -> list[float]:
    """Generate embedding for a single text."""
    try:
        response = _embedding_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except NotFoundError as exc:
        raise RuntimeError(
            "Embedding deployment not found. Configure AZURE_OPENAI_EMBEDDING_DEPLOYMENT "
            "and optionally AZURE_OPENAI_EMBEDDING_ENDPOINT / AZURE_OPENAI_EMBEDDING_KEY "
            f"for the deployment that serves embeddings. Current embedding deployment: {EMBEDDING_MODEL}"
        ) from exc


def search_documents(
    query: str,
    top_k: int = TOP_K,
    source_filter: str | None = None,
    index_name: str = DEFAULT_INDEX_NAME,
) -> str:
    """Search the document index using hybrid search (keyword + vector)."""
    _t0 = time.time()
    try:
        safe_top_k = max(1, min(int(top_k), TOP_K))
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

        output_parts = []
        sources = []
        for i, result in enumerate(results, 1):
            output_parts.append(
                f"[Result {i} | {result['source_file']} | Page {result['page_number']}]\n"
                f"{result['content']}"
            )
            sources.append(f"{result['source_file']} (p.{result['page_number']})")

        with _tool_call_log_lock:
            _tool_call_log.append({
                "query": query,
                "index_name": index_name,
                "source_filter": source_filter,
                "results_count": len(output_parts),
                "sources": sources[:5],
                "duration": round(time.time() - _t0, 2),
            })

        if not output_parts:
            return "No results found for this query."

        return "\n\n---\n\n".join(output_parts)
    except Exception as e:
        with _tool_call_log_lock:
            _tool_call_log.append({"query": query, "index_name": index_name, "error": str(e), "duration": round(time.time() - _t0, 2)})
        return f"Search error: {e}"


# Track tool calls for external consumers (e.g. the UI)
_tool_call_log: list[dict] = []
_tool_call_log_lock = Lock()
_tool_round_durations: list[float] = []

# Tool definition for OpenAI function calling
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Search the indexed documents using hybrid search (keyword + vector). "
            "Use this tool to find specific information from the available documents. "
            "You can call this multiple times with different queries to gather all needed information. "
            "Use targeted, specific queries for best results. "
            "Use source_filter to narrow results to a specific source file when the exact file name is known."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — be specific and targeted",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (1-8)",
                    "default": 8,
                },
                "source_filter": {
                    "type": "string",
                    "description": "Optional: exact source file name to narrow the search to a single document.",
                },
            },
            "required": ["query"],
        },
    },
}]

AGENT_INSTRUCTIONS = (
    "You are a general-purpose document analysis assistant with access to a search tool over indexed documents.\n\n"
    "PLAN-AND-EXECUTE POLICY:\n"
    "1. ALWAYS use the search_documents tool to find information. NEVER answer from memory.\n"
    "2. For every non-trivial question, first decompose it internally into a compact task ledger of the exact information needs, such as entities, attributes, constraints, sections, time ranges, or comparison dimensions.\n"
    "3. When the request has multiple parts, internally track each task as resolved, unresolved, or unsupported. Treat partial completion as incomplete.\n"
    "4. For broad or complex documents, plan retrieval in stages: first locate the most relevant section, table, or passage cluster; then extract the exact facts needed from that area.\n"
    "5. Group related tasks that can likely be answered from the same evidence, but split tasks apart when a targeted search will improve precision.\n"
    "6. If multiple tasks are independent, issue multiple search_documents tool calls in the SAME turn so they can run in parallel.\n"
    "7. After every tool round, compare the evidence collected so far against the task ledger and identify the smallest set of remaining gaps. Use the next round only for those gaps.\n"
    "8. Use source_filter only when you know the exact source file name and want to constrain the search to one document.\n"
    "9. Keep search queries targeted and information-dense. Prefer queries that combine the subject, the needed attribute, and any important constraint or scope.\n"
    "10. On large documents, prefer smart navigation over brute force. Use broad locator searches to find the right region, then focused searches to extract missing details.\n"
    "11. If a follow-up search is substantially similar to a previous one and did not add useful evidence, do not keep repeating it. Try a meaningfully different angle or mark the item as not found if needed.\n"
    "12. Before giving the final answer, verify that every requested sub-question has either a supported answer or an explicit not found outcome after targeted searching.\n"
    "13. Do not reveal your hidden reasoning or internal plan. Execute it through efficient tool usage and then return the final answer.\n\n"
    "ANSWER RULES:\n"
    "1. When the answer involves structured comparisons or numeric values, prefer a markdown table. Otherwise use the format that best fits the request.\n"
    "2. Use numbered references like [1], [2], [3] in the answer text to cite sources. "
    "At the end of your answer, add a 'References' section listing each number with its source file and page "
    "when available. Do NOT write the full source file name inline in the answer body — only use the number.\n"
    "3. If information is not found after thorough searching, say so explicitly and identify which requested item was not found.\n"
    "4. For multi-entity or multi-metric requests, ensure the final answer covers every requested row and column from the user's ask.\n"
    "5. Answer only the question that was asked. Do not add extra metrics, extra columns, extra sections, or related facts unless the user explicitly requested them.\n"
    "6. Do not estimate, infer, back-calculate, or approximate missing values. If a requested value is not supported by retrieved evidence, mark it as not found.\n"
    "7. Preserve the user's requested schema as closely as possible. If the user named specific fields, use only those fields in the final table or structure.\n"
    "8. Do not claim facts that are not supported by the retrieved context."
)


def _tool_call_signature(tool_call, index_name: str) -> tuple[tuple[str, str | int | None], ...]:
    """Build a stable signature for one tool call so repeated rounds can be detected."""
    args = json.loads(tool_call.function.arguments)
    args.setdefault("index_name", index_name)
    normalized = {
        "index_name": args.get("index_name"),
        "query": args.get("query"),
        "source_filter": args.get("source_filter"),
        "top_k": args.get("top_k", TOP_K),
    }
    return tuple(sorted(normalized.items()))


def _generate_final_answer(messages: list[dict]) -> str:
    """Force a final answer using only the evidence collected so far."""
    final_messages = list(messages)
    final_messages.append({
        "role": "system",
        "content": (
            "Produce the final answer using only the evidence already retrieved. "
            "Answer only the user's requested fields. Do not add extra columns, extra metrics, or derived values. "
            "If a requested item is unsupported, mark it as not found instead of estimating it."
        ),
    })
    response = _chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=final_messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or "No answer generated."


def _execute_tool_call(tool_call, index_name: str) -> tuple[str, str]:
    """Execute one search tool call and return the tool id with its output."""
    args = json.loads(tool_call.function.arguments)
    args.setdefault("index_name", index_name)
    return tool_call.id, search_documents(**args)


def _run_parallel_tool_calls(tool_calls, index_name: str) -> list[tuple[str, str]]:
    """Execute one round of independent tool calls concurrently while preserving order."""
    started = time.time()
    ordered_results: list[tuple[str, str] | None] = [None] * len(tool_calls)

    if len(tool_calls) == 1:
        ordered_results[0] = _execute_tool_call(tool_calls[0], index_name)
    else:
        max_workers = min(len(tool_calls), MAX_PARALLEL_SEARCHES)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_execute_tool_call, tool_call, index_name): position
                for position, tool_call in enumerate(tool_calls)
            }
            for future in as_completed(future_map):
                ordered_results[future_map[future]] = future.result()

    _tool_round_durations.append(round(time.time() - started, 2))
    return [result for result in ordered_results if result is not None]


def _run_agentic_rag(query: str, index_name: str = DEFAULT_INDEX_NAME) -> dict:
    """Run agentic RAG with OpenAI tool calling loop."""
    global _tool_call_log, _tool_round_durations
    _tool_call_log = []
    _tool_round_durations = []

    start = time.time()

    messages = [
        {"role": "system", "content": AGENT_INSTRUCTIONS},
        {"role": "user", "content": query},
    ]
    answer = ""
    tool_rounds = 0
    stalled_rounds = 0
    llm_calls = 0
    previous_round_signatures: tuple[tuple[tuple[str, str | int | None], ...], ...] | None = None

    while tool_rounds < MAX_TOOL_ROUNDS:
        response = _chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            parallel_tool_calls=True,
            temperature=0.3,
        )
        llm_calls += 1

        choice = response.choices[0]

        if choice.message.tool_calls:
            # Append the assistant message with tool calls
            messages.append(choice.message)
            current_round_signatures = tuple(
                _tool_call_signature(tool_call, index_name) for tool_call in choice.message.tool_calls
            )
            round_started_at = len(_tool_call_log)
            for tool_call_id, result in _run_parallel_tool_calls(choice.message.tool_calls, index_name):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                })

            round_logs = _tool_call_log[round_started_at:]
            round_chunks = sum(log.get("results_count", 0) for log in round_logs)
            is_repeated_round = current_round_signatures == previous_round_signatures

            if is_repeated_round or round_chunks == 0:
                stalled_rounds += 1
            else:
                stalled_rounds = 0

            previous_round_signatures = current_round_signatures
            tool_rounds += 1

            if stalled_rounds >= MAX_STALLED_ROUNDS:
                messages.append({
                    "role": "system",
                    "content": (
                        "Search progress has stalled. Do not call tools again. "
                        "Produce the best final answer from the evidence already gathered, and explicitly "
                        "mark unresolved items as not found after searching. Do not add any fields or metrics "
                        "that were not explicitly requested by the user."
                    ),
                })
                answer = _generate_final_answer(messages)
                llm_calls += 1
                break
        else:
            # Final text response
            answer = choice.message.content or ""
            break

    if not answer:
        messages.append({
            "role": "system",
            "content": (
                "You have reached the tool-use budget. Produce the best final answer from the evidence already "
                "retrieved. Explicitly mark anything still unresolved as not found after searching. Do not add "
                "extra fields, extra metrics, or derived values beyond the user's request."
            ),
        })
        answer = _generate_final_answer(messages)
        llm_calls += 1

    total_time = time.time() - start
    total_chunks = sum(tc.get("results_count", 0) for tc in _tool_call_log)
    search_time = round(sum(_tool_round_durations), 2)
    generation_time = round(max(total_time - search_time, 0), 2)

    return {
        "answer": answer,
        "chunks_retrieved": total_chunks,
        "search_calls": len(_tool_call_log),
        "llm_calls": llm_calls,
        "tool_calls": list(_tool_call_log),
        "search_time": search_time,
        "generation_time": generation_time,
        "total_time": round(total_time, 2),
    }


def ask_with_metadata(query: str, index_name: str = DEFAULT_INDEX_NAME) -> dict:
    """Run agentic RAG, returning answer + metadata."""
    return _run_agentic_rag(query, index_name=index_name)


def main():
    print("=== Agentic RAG (Tool Calling) ===")
    print("Ask questions about the indexed documents. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        print("\nAgent thinking...\n")
        result = _run_agentic_rag(query)
        print(result["answer"])
        print(f"\n[{result['search_calls']} searches, {result['chunks_retrieved']} chunks, {result['total_time']}s]\n")


if __name__ == "__main__":
    main()
