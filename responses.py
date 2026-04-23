"""Responses API agentic RAG using native function-calling items.

The model decides when and how often to call the search tool. The application
executes those tool calls against Azure AI Search and feeds the outputs back to
the Responses API until the model returns a final answer.
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

load_dotenv()

# --- Configuration ---
SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_AI_SEARCH_KEY"]
OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_RESPONSES_ENDPOINT", os.environ["AZURE_OPENAI_ENDPOINT"])
OPENAI_KEY = os.environ.get("AZURE_OPENAI_RESPONSES_KEY", os.environ["AZURE_OPENAI_KEY"])
EMBEDDING_ENDPOINT = os.environ.get("AZURE_OPENAI_EMBEDDING_ENDPOINT", os.environ["AZURE_OPENAI_ENDPOINT"])
EMBEDDING_KEY = os.environ.get("AZURE_OPENAI_EMBEDDING_KEY", os.environ["AZURE_OPENAI_KEY"])
EMBEDDING_MODEL = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", os.environ["AZURE_AI_EMBEDDING_MODEL"])
CHAT_MODEL = os.environ.get("AZURE_OPENAI_RESPONSES_DEPLOYMENT", os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"])

DEFAULT_INDEX_NAME = "annual-reports-index"
TOP_K = 12
MAX_TOOL_ROUNDS = 8
MAX_PARALLEL_SEARCHES = 4
MAX_STALLED_ROUNDS = 2

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
    "10. Before giving the final answer, verify that every requested sub-question has either a supported answer or an explicit not found outcome after targeted searching.\n"
    "11. If a follow-up search is substantially similar to a previous one and did not add useful evidence, do not keep repeating it. Try a meaningfully different angle or mark the item as not found if needed.\n"
    "12. Do not reveal your hidden reasoning or internal plan. Execute it through efficient tool usage and then return the final answer.\n\n"
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

TOOLS = [
    {
        "type": "function",
        "name": "search_documents",
        "description": (
            "Search the indexed documents using hybrid search over keyword and vector retrieval. "
            "Use this tool to find facts from one or more documents. For broad questions, make several targeted searches. "
            "source_filter should be null or one exact source file name when you want to narrow to a single document."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A specific search query for the needed fact or section.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return, usually between 1 and 10.",
                },
                "source_filter": {
                    "type": ["string", "null"],
                    "description": "Exact source file name to filter to one document, or null.",
                },
            },
            "required": ["query", "top_k", "source_filter"],
            "additionalProperties": False,
        },
    }
]

_tool_call_log: list[dict] = []
_tool_call_log_lock = Lock()
_tool_round_durations: list[float] = []


def get_responses_client() -> AzureOpenAI:
    """Create an Azure OpenAI client for Responses API calls."""
    return AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_KEY,
        api_version="2025-03-01-preview",
    )


def get_embedding_client() -> AzureOpenAI:
    """Create an Azure OpenAI client for embedding generation."""
    return AzureOpenAI(
        azure_endpoint=EMBEDDING_ENDPOINT,
        api_key=EMBEDDING_KEY,
        api_version="2024-12-01-preview",
    )


def embed_query(client: AzureOpenAI, query: str) -> list[float]:
    """Generate embedding for the user query."""
    try:
        response = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except NotFoundError as exc:
        raise RuntimeError(
            "Embedding deployment not found. Configure AZURE_OPENAI_EMBEDDING_DEPLOYMENT "
            "and optionally AZURE_OPENAI_EMBEDDING_ENDPOINT / AZURE_OPENAI_EMBEDDING_KEY "
            f"for the deployment that serves embeddings. Current embedding deployment: {EMBEDDING_MODEL}"
        ) from exc


def search_documents(
    search_client: SearchClient,
    embedding_client: AzureOpenAI,
    query: str,
    top_k: int = TOP_K,
    source_filter: str | None = None,
) -> str:
    """Search the document index using hybrid search (keyword + vector)."""
    started = time.time()
    safe_top_k = max(1, min(int(top_k), 15))
    try:
        query_embedding = embed_query(embedding_client, query)
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
        for index, result in enumerate(results, 1):
            output_parts.append(
                f"[Result {index} | {result['source_file']} | Page {result['page_number']}]\n"
                f"{result['content']}"
            )
            sources.append(f"{result['source_file']} (p.{result['page_number']})")

        with _tool_call_log_lock:
            _tool_call_log.append(
                {
                    "query": query,
                    "source_filter": source_filter,
                    "results_count": len(output_parts),
                    "sources": sources[:5],
                    "duration": round(time.time() - started, 2),
                }
            )

        if not output_parts:
            return "No results found for this query."

        return "\n\n---\n\n".join(output_parts)
    except Exception as exc:
        with _tool_call_log_lock:
            _tool_call_log.append(
                {
                    "query": query,
                    "source_filter": source_filter,
                    "error": str(exc),
                    "duration": round(time.time() - started, 2),
                }
            )
        return f"Search error: {exc}"


def _extract_output_text(response) -> str:
    """Normalize the Responses API payload into plain text."""
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


def _generate_final_answer(client: AzureOpenAI, input_items: list) -> str:
    """Force a focused final answer using the collected evidence."""
    final_input = list(input_items)
    final_input.append({
        "role": "user",
        "content": (
            "Search complete. Produce the final answer now using ONLY the evidence already retrieved. "
            "Answer all parts of the user question. Do not add extra columns, extra metrics, or derived values. "
            "If a requested item is unsupported, mark it as not found instead of estimating it."
        )
    })
    
    response = _create_response(
        client,
        model=CHAT_MODEL,
        instructions=AGENT_INSTRUCTIONS,
        input=final_input,
        temperature=0.2, # Lower temperature for synthesis
    )
    return _extract_output_text(response)


def _create_response(client: AzureOpenAI, **kwargs):
    """Call the Azure OpenAI Responses API with consistent error handling."""
    try:
        return client.responses.create(**kwargs)
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


def ask(query: str, index_name: str = DEFAULT_INDEX_NAME) -> str:
    """Run the Responses-based agentic RAG pipeline."""
    result = ask_with_metadata(query, index_name=index_name)
    return result["answer"]


def _execute_function_call(tool_call, search_client: SearchClient, embedding_client: AzureOpenAI) -> dict:
    """Execute one Responses API function call."""
    args = json.loads(tool_call.arguments)
    result = search_documents(
        search_client,
        embedding_client,
        query=args["query"],
        top_k=args["top_k"],
        source_filter=args["source_filter"],
    )
    return {
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": result,
    }


def _run_parallel_function_calls(function_calls, search_client: SearchClient, embedding_client: AzureOpenAI) -> list[dict]:
    """Execute one round of independent Responses API function calls concurrently."""
    started = time.time()
    ordered_outputs: list[dict | None] = [None] * len(function_calls)

    if len(function_calls) == 1:
        ordered_outputs[0] = _execute_function_call(function_calls[0], search_client, embedding_client)
    else:
        max_workers = min(len(function_calls), MAX_PARALLEL_SEARCHES)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_execute_function_call, tool_call, search_client, embedding_client): position
                for position, tool_call in enumerate(function_calls)
            }
            for future in as_completed(future_map):
                ordered_outputs[future_map[future]] = future.result()

    _tool_round_durations.append(round(time.time() - started, 2))
    return [output for output in ordered_outputs if output is not None]


def _get_call_signatures(function_calls) -> set:
    """Build a set of tool call signatures for a round."""
    sigs = set()
    for call in function_calls:
        args = json.loads(call.arguments)
        # Sort keys to ensure stable signature
        sig = tuple(sorted(args.items()))
        sigs.add(sig)
    return sigs


def ask_with_metadata(query: str, index_name: str = DEFAULT_INDEX_NAME) -> dict:
    """Run native Responses tool-calling RAG and return answer + metadata."""
    global _tool_call_log, _tool_round_durations
    _tool_call_log = []
    _tool_round_durations = []

    started = time.time()
    embedding_client = get_embedding_client()
    responses_client = get_responses_client()
    search_client = SearchClient(SEARCH_ENDPOINT, index_name, AzureKeyCredential(SEARCH_KEY))

    input_items: list = [{"role": "user", "content": query}]
    response = None
    tool_rounds = 0
    stalled_rounds = 0
    previous_signatures = None

    while tool_rounds < MAX_TOOL_ROUNDS:
        response = _create_response(
            responses_client,
            model=CHAT_MODEL,
            instructions=AGENT_INSTRUCTIONS,
            tools=TOOLS,
            input=input_items,
            temperature=0.3,
            parallel_tool_calls=True,
        )

        function_calls = [
            item for item in getattr(response, "output", []) or []
            if getattr(item, "type", None) == "function_call"
        ]

        if not function_calls:
            break

        current_signatures = _get_call_signatures(function_calls)
        input_items.extend(response.output)
        
        tool_outputs = _run_parallel_function_calls(function_calls, search_client, embedding_client)
        input_items.extend(tool_outputs)
        
        # Check for stalls (repeated queries or no results)
        total_results = sum(getattr(item, "results_count", 0) for item in _tool_call_log[-len(function_calls):])
        if current_signatures == previous_signatures or total_results == 0:
            stalled_rounds += 1
        else:
            stalled_rounds = 0
            
        previous_signatures = current_signatures
        tool_rounds += 1
        
        if stalled_rounds >= MAX_STALLED_ROUNDS:
            break

    total_time = time.time() - started
    
    # If we exited due to no more calls or budget/stalls, do a final synthesis
    answer = _generate_final_answer(responses_client, input_items)
    
    if not answer:
        answer = "No answer generated."

    sources: list[str] = []
    seen_sources: set[str] = set()
    for tool_call in _tool_call_log:
        for source in tool_call.get("sources", []):
            if source not in seen_sources:
                seen_sources.add(source)
                sources.append(source)

    search_time = round(sum(_tool_round_durations), 2)
    generation_time = round(max(total_time - search_time, 0), 2)
    chunks_retrieved = sum(call.get("results_count", 0) for call in _tool_call_log)

    if tool_rounds > MAX_TOOL_ROUNDS:
        answer = (
            "The Responses API agent exceeded the configured tool-call limit before reaching a final answer.\n\n"
            "Try a narrower question, or increase MAX_TOOL_ROUNDS in responses.py."
        )

    return {
        "answer": answer,
        "chunks_retrieved": chunks_retrieved,
        "search_calls": len(_tool_call_log),
        "sources": sources,
        "search_time": search_time,
        "generation_time": generation_time,
        "total_time": round(total_time, 2),
        "response_id": getattr(response, "id", None) if response else None,
        "available": True,
        "tool_calls": list(_tool_call_log),
    }


def main():
    print("=== Document Responses API RAG ===")
    print("Ask questions about the indexed documents. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            break
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        answer = ask(query)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()