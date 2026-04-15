"""
RAG script: Ask questions over the annual reports search index.

Uses hybrid search (keyword + vector) to retrieve relevant chunks,
then sends them as context to gpt-5.4 for answer generation.
"""

import os

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery, VectorizedQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# --- Configuration ---
SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_AI_SEARCH_KEY"]
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
EMBEDDING_MODEL = os.environ["AZURE_AI_EMBEDDING_MODEL"]
CHAT_MODEL = os.environ["AZURE_AI_MODEL_DEPLOYMENT_NAME"]

INDEX_NAME = "annual-reports-index"
TOP_K = 15 # number of chunks to retrieve


def get_openai_client() -> AzureOpenAI:
    """Create an Azure OpenAI client using API key."""
    return AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_KEY,
        api_version="2024-12-01-preview",
    )


def embed_query(client: AzureOpenAI, query: str) -> list[float]:
    """Generate embedding for the user query."""
    response = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def hybrid_search(search_client: SearchClient, openai_client: AzureOpenAI, query: str) -> list[dict]:
    """Perform hybrid search (keyword + vector) on the index."""
    query_embedding = embed_query(openai_client, query)

    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=TOP_K,
        fields="embedding",
    )

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=TOP_K,
        select=["content", "source_file", "page_number", "chunk_index"],
    )

    chunks = []
    for result in results:
        chunks.append({
            "content": result["content"],
            "source_file": result["source_file"],
            "page_number": result["page_number"],
            "score": result["@search.score"],
        })

    return chunks


def build_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """Build the chat messages with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source_file']}, Page {chunk['page_number']}]\n{chunk['content']}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    system_message = (
        "You are a helpful financial analyst assistant. Answer the user's question "
        "based ONLY on the provided context from annual reports. "
        "If the context doesn't contain enough information to answer, say so clearly.\n\n"
        "RESPONSE FORMAT:\n"
        "- When the answer involves numerical data or comparisons, present a markdown table FIRST with all found values.\n"
        "- After the table, add any additional commentary or analysis.\n\n"
        "CITATION FORMAT:\n"
        "- Use numbered references like [1], [2], [3] in the answer text to cite sources.\n"
        "- At the end of your answer, add a 'References' section listing each number with its source file and page.\n"
        "- Example: [1] Apollo_Hospitals_Delhi_Annual_Report_2024-25.pdf, Page 61\n"
        "- Do NOT write the full source file name inline in the answer body — only use the number."
    )

    user_message = f"Context:\n{context_text}\n\n---\n\nQuestion: {query}"

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def ask(query: str) -> str:
    """Full RAG pipeline: search -> build prompt -> generate answer."""
    result = ask_with_metadata(query)
    return result["answer"]


def ask_with_metadata(query: str) -> dict:
    """Full RAG pipeline returning answer + metadata for the UI."""
    import time
    start = time.time()

    openai_client = get_openai_client()
    search_client = SearchClient(
        SEARCH_ENDPOINT, INDEX_NAME, AzureKeyCredential(SEARCH_KEY)
    )

    # 1. Retrieve relevant chunks
    chunks = hybrid_search(search_client, openai_client, query)
    search_time = time.time() - start

    # 2. Build prompt with context
    messages = build_prompt(query, chunks)

    # 3. Generate answer
    gen_start = time.time()
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
    )
    gen_time = time.time() - gen_start
    total_time = time.time() - start

    return {
        "answer": response.choices[0].message.content,
        "chunks_retrieved": len(chunks),
        "search_calls": 1,
        "sources": [f"{c['source_file']} (p.{c['page_number']})" for c in chunks],
        "search_time": round(search_time, 2),
        "generation_time": round(gen_time, 2),
        "total_time": round(total_time, 2),
    }


def main():
    print("=== Annual Reports RAG ===")
    print("Ask questions about the annual reports. Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if not query or query.lower() in ("quit", "exit", "q"):
            break

        answer = ask(query)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
