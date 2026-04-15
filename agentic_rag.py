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
from openai import AzureOpenAI

load_dotenv(override=False)

# --- Configuration ---
SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_AI_SEARCH_KEY"]
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
EMBEDDING_MODEL = os.environ["AZURE_AI_EMBEDDING_MODEL"]
CHAT_MODEL = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]
INDEX_NAME = "annual-reports-index"

# --- Clients ---
_openai_client = AzureOpenAI(
    azure_endpoint=OPENAI_ENDPOINT,
    api_key=OPENAI_KEY,
    api_version="2024-12-01-preview",
)

_search_client = SearchClient(
    SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY)
)


def _embed(text: str) -> list[float]:
    """Generate embedding for a single text."""
    response = _openai_client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def search_annual_reports(query: str, top_k: int = 10, source_filter: str | None = None) -> str:
    """Search the annual reports index using hybrid search (keyword + vector)."""
    _t0 = time.time()
    try:
        query_embedding = _embed(query)

        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )

        filter_expr = None
        if source_filter:
            filter_expr = f"source_file eq '{source_filter}'"

        results = _search_client.search(
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
            "source_filter": source_filter,
            "results_count": len(output_parts),
            "sources": sources[:5],
            "duration": round(time.time() - _t0, 2),
        })

        if not output_parts:
            return "No results found for this query."

        return "\n\n---\n\n".join(output_parts)
    except Exception as e:
        _tool_call_log.append({"query": query, "error": str(e), "duration": round(time.time() - _t0, 2)})
        return f"Search error: {e}"


# Track tool calls for external consumers (e.g. the UI)
_tool_call_log: list[dict] = []

# Tool definition for OpenAI function calling
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_annual_reports",
        "description": (
            "Search the annual reports index using hybrid search (keyword + vector). "
            "Use this tool to find specific information from company annual reports. "
            "You can call this multiple times with different queries to gather all needed information. "
            "Use targeted, specific queries for best results. "
            "Use source_filter to narrow results to a specific company file."
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
                    "description": "Optional: exact source file name to filter (e.g., 'Apollo_Hospitals_Delhi_Annual_Report_2024-25.pdf')",
                },
            },
            "required": ["query"],
        },
    },
}]

AGENT_INSTRUCTIONS = (
    "You are an expert financial analyst assistant with access to annual reports "
    "from six Indian listed companies for FY 2024-25.\n\n"
    "AVAILABLE SOURCE FILES:\n"
    "- Apollo_Hospitals_Delhi_Annual_Report_2024-25.pdf\n"
    "- Data_Patterns_Annual_Report_2024-25.pdf\n"
    "- IndiGo_Airlines_Annual_Report_2024-25.pdf\n"
    "- Indigo_Paints_Annual_Report_2024-25.pdf\n"
    "- KPEL_Annual_Report_2024-25.pdf\n"
    "- Oracle_Financial_Services_Annual_Report_2024-25.pdf\n\n"
    "IMPORTANT RULES:\n"
    "1. ALWAYS use the search_annual_reports tool to find information. NEVER answer from memory.\n"
    "2. For complex questions involving multiple companies or metrics, break them down and search SEPARATELY "
    "for each company and each metric. Make multiple targeted searches.\n"
    "3. If your first search doesn't return enough information, search again with different queries.\n"
    "4. When comparing companies, search for each company individually using source_filter with the exact file name.\n"
    "5. When the answer involves numerical data or comparisons, present a markdown table FIRST with all found values, "
    "then add any additional commentary or analysis below the table.\n"
    "6. Use numbered references like [1], [2], [3] in the answer text to cite sources. "
    "At the end of your answer, add a 'References' section listing each number with its source file and page "
    "(e.g., [1] Apollo_Hospitals_Delhi_Annual_Report_2024-25.pdf, Page 61). "
    "Do NOT write the full source file name inline in the answer body — only use the number.\n"
    "7. If information is not found after thorough searching, say so explicitly.\n"
    "8. Do NOT pass source_filter unless you use an exact file name from the list above."
)


def _run_agentic_rag(query: str) -> dict:
    """Run agentic RAG with OpenAI tool calling loop."""
    global _tool_call_log
    _tool_call_log = []

    start = time.time()

    messages = [
        {"role": "system", "content": AGENT_INSTRUCTIONS},
        {"role": "user", "content": query},
    ]

    while True:
        response = _openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
            temperature=0.3,
        )

        choice = response.choices[0]

        if choice.message.tool_calls:
            # Append the assistant message with tool calls
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                args = json.loads(tool_call.function.arguments)
                result = search_annual_reports(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            # Final text response
            answer = choice.message.content or ""
            break

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


def ask_with_metadata(query: str) -> dict:
    """Run agentic RAG, returning answer + metadata."""
    return _run_agentic_rag(query)


def main():
    print("=== Agentic RAG (Tool Calling) ===")
    print("Ask questions about the annual reports. Type 'quit' to exit.\n")

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
