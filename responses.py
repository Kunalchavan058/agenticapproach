"""Responses API agentic RAG using native function-calling items.

The model decides when and how often to call the search tool. The application
executes those tool calls against Azure AI Search and feeds the outputs back to
the Responses API until the model returns a final answer.
"""

import json
import os
import time

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

INDEX_NAME = "annual-reports-index"
TOP_K = 8
MAX_TOOL_ROUNDS = 8

AGENT_INSTRUCTIONS = (
    "You are a general-purpose document analysis assistant with access to a search tool over indexed documents.\n\n"
    "IMPORTANT RULES:\n"
    "1. ALWAYS use the search_documents tool before answering. Never answer from memory.\n"
    "2. For complex questions, perform multiple targeted searches. Break the problem down by topic, entity, section, or document when useful.\n"
    "3. If your first searches are incomplete, search again with narrower queries.\n"
    "4. Use source_filter only when you know the exact source file name for a specific document.\n"
    "5. When the answer involves comparisons or structured values, use a markdown table. Otherwise choose the most suitable format.\n"
    "6. If part of the requested information is missing after searching, say what was not found explicitly.\n"
    "7. Use numbered references like [1], [2], [3] in the answer text and add a References section with file name and page.\n"
    "8. Do not stop after partial coverage if the user asked for multiple items, and do not claim facts not supported by the retrieved context."
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
    safe_top_k = max(1, min(int(top_k), 10))
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


def ask(query: str) -> str:
    """Run the Responses-based agentic RAG pipeline."""
    result = ask_with_metadata(query)
    return result["answer"]


def ask_with_metadata(query: str) -> dict:
    """Run native Responses tool-calling RAG and return answer + metadata."""
    global _tool_call_log
    _tool_call_log = []

    started = time.time()
    embedding_client = get_embedding_client()
    responses_client = get_responses_client()
    search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY))

    input_items: list = [{"role": "user", "content": query}]
    response = None
    tool_rounds = 0

    while tool_rounds <= MAX_TOOL_ROUNDS:
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

        input_items.extend(response.output)
        tool_outputs = []
        for tool_call in function_calls:
            args = json.loads(tool_call.arguments)
            result = search_documents(
                search_client,
                embedding_client,
                query=args["query"],
                top_k=args["top_k"],
                source_filter=args["source_filter"],
            )
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": result,
                }
            )

        input_items.extend(tool_outputs)
        tool_rounds += 1

    total_time = time.time() - started
    answer = _extract_output_text(response)
    if not answer:
        answer = "No answer generated."

    sources: list[str] = []
    seen_sources: set[str] = set()
    for tool_call in _tool_call_log:
        for source in tool_call.get("sources", []):
            if source not in seen_sources:
                seen_sources.add(source)
                sources.append(source)

    search_time = round(sum(call.get("duration", 0) for call in _tool_call_log), 2)
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