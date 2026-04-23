"""
Agentic RAG using OpenAI tool calling.

The LLM has a search tool it can call multiple times with different queries.
It autonomously decides how to decompose complex questions, search iteratively,
and synthesize a complete answer from all retrieved information.
"""

import json
import os
import time

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
MAX_TOOL_ROUNDS = 8

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
    top_k: int = 10,
    source_filter: str | None = None,
    index_name: str = DEFAULT_INDEX_NAME,
) -> str:
    """Search the document index using hybrid search (keyword + vector)."""
    _t0 = time.time()
    try:
        search_client = _get_search_client(index_name)
        query_embedding = _embed(query)

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )

        filter_expr = None
        if source_filter:
            filter_expr = f"source_file eq '{source_filter}'"

        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            top=top_k,
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
        _tool_call_log.append({"query": query, "index_name": index_name, "error": str(e), "duration": round(time.time() - _t0, 2)})
        return f"Search error: {e}"


# Track tool calls for external consumers (e.g. the UI)
_tool_call_log: list[dict] = []

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
                    "description": "Number of results to return (1-20)",
                    "default": 10,
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
    "2. For every non-trivial question, first decompose it into smaller factual tasks internally before deciding on tool calls.\n"
    "3. Prefer a short plan with atomic tasks such as entities, attributes, time periods, sections, or comparison dimensions.\n"
    "4. If multiple tasks are independent, issue multiple search_documents tool calls in the SAME turn so they can run in parallel.\n"
    "5. Use follow-up searches only for unresolved gaps after reviewing previous tool results. Avoid redundant searches.\n"
    "6. Use source_filter only when you know the exact source file name and want to constrain the search to one document.\n"
    "7. Keep search queries targeted and specific. Do not send one broad query when several narrower queries would retrieve better evidence.\n"
    "8. Do not reveal your hidden reasoning or internal plan. Execute it through efficient tool usage and then return the final answer.\n\n"
    "ANSWER RULES:\n"
    "1. When the answer involves structured comparisons or numeric values, prefer a markdown table. Otherwise use the format that best fits the request.\n"
    "2. Use numbered references like [1], [2], [3] in the answer text to cite sources. "
    "At the end of your answer, add a 'References' section listing each number with its source file and page "
    "when available. Do NOT write the full source file name inline in the answer body — only use the number.\n"
    "3. If information is not found after thorough searching, say so explicitly.\n"
    "4. Do not stop after partial coverage if the user asked for multiple items.\n"
    "5. Do not claim facts that are not supported by the retrieved context."
)


def _run_agentic_rag(query: str, index_name: str = DEFAULT_INDEX_NAME) -> dict:
    """Run agentic RAG with OpenAI tool calling loop."""
    global _tool_call_log
    _tool_call_log = []

    start = time.time()

    messages = [
        {"role": "system", "content": AGENT_INSTRUCTIONS},
        {"role": "user", "content": query},
    ]
    tool_rounds = 0

    while tool_rounds <= MAX_TOOL_ROUNDS:
        response = _chat_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            parallel_tool_calls=True,
            temperature=0.3,
        )

        choice = response.choices[0]

        if choice.message.tool_calls:
            # Append the assistant message with tool calls
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                args.setdefault("index_name", index_name)
                result = search_documents(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            tool_rounds += 1
        else:
            # Final text response
            answer = choice.message.content or ""
            break

    if tool_rounds > MAX_TOOL_ROUNDS:
        answer = (
            "The agent exceeded the configured tool-call limit before reaching a final answer.\n\n"
            "Try a narrower question, or increase MAX_TOOL_ROUNDS in agentic_rag.py."
        )

    total_time = time.time() - start
    total_chunks = sum(tc.get("results_count", 0) for tc in _tool_call_log)
    search_time = round(sum(tc.get("duration", 0) for tc in _tool_call_log), 2)
    generation_time = round(total_time - search_time, 2)

    return {
        "answer": answer,
        "chunks_retrieved": total_chunks,
        "search_calls": len(_tool_call_log),
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
