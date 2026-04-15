"""
Azure AI Search Agentic Retrieval — built-in knowledge base pipeline.

This uses Azure AI Search's native agentic retrieval feature:
  1. Updates the existing index with a semantic configuration (required).
  2. Creates a knowledge source wrapping the existing index.
  3. Creates a knowledge base that connects the knowledge source to an LLM.
  4. Queries via the retrieve action — Azure AI Search handles query planning,
     sub-query generation, semantic reranking, and answer synthesis internally.

Prerequisites:
  - Azure AI Search Basic tier or higher with semantic ranker enabled.
  - azure-search-documents>=11.7.0b2 (preview).
  - The search service's managed identity needs "Cognitive Services User" role
    on the Azure OpenAI resource (or use API keys).

Run:  uv run python agentic_retrieval.py
"""

import os
import time

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    KnowledgeBase,
    KnowledgeBaseAzureOpenAIModel,
    KnowledgeRetrievalLowReasoningEffort,
    KnowledgeRetrievalOutputMode,
    KnowledgeSourceReference,
    SearchIndex,
    SearchIndexKnowledgeSource,
    SearchIndexKnowledgeSourceParameters,
    SearchIndexFieldReference,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    AzureOpenAIVectorizerParameters,
)
from azure.search.documents.knowledgebases import KnowledgeBaseRetrievalClient
from azure.search.documents.knowledgebases.models import (
    KnowledgeBaseMessage,
    KnowledgeBaseMessageTextContent,
    KnowledgeBaseRetrievalRequest,
    SearchIndexKnowledgeSourceParams,
)
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_AI_SEARCH_KEY"]
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
EMBEDDING_MODEL = os.environ["AZURE_AI_EMBEDDING_MODEL"]
INDEX_NAME = "annual-reports-index"

# Agentic retrieval only supports specific models — read from .env
AGENTIC_DEPLOYMENT_NAME = os.environ["AZURE_AGENTIC_RETRIEVAL_DEPLOYMENT"]
AGENTIC_MODEL_NAME = AGENTIC_DEPLOYMENT_NAME

KNOWLEDGE_SOURCE_NAME = "annual-reports-ks"
KNOWLEDGE_BASE_NAME = "annual-reports-kb"
SEMANTIC_CONFIG_NAME = "annual-reports-semantic"


def get_index_client() -> SearchIndexClient:
    return SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))


# ============================================================
# Step 1: Add semantic configuration to the existing index
# ============================================================
def update_index_with_semantic_config(index_client: SearchIndexClient) -> None:
    """Add a semantic configuration to the existing index (no re-indexing needed)."""
    existing_index: SearchIndex = index_client.get_index(INDEX_NAME)

    semantic_config = SemanticConfiguration(
        name=SEMANTIC_CONFIG_NAME,
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")],
        ),
    )

    existing_index.semantic_search = SemanticSearch(
        configurations=[semantic_config],
    )

    # Add a description so the LLM can decide whether this index is relevant
    existing_index.description = (
        "Annual reports (FY 2024-25) for six Indian listed companies: "
        "Apollo Hospitals Delhi, Data Patterns, IndiGo Airlines, Indigo Paints, "
        "KPEL, and Oracle Financial Services. Contains chunked text with embeddings."
    )

    index_client.create_or_update_index(existing_index)
    print(f"[OK] Semantic config '{SEMANTIC_CONFIG_NAME}' added to index '{INDEX_NAME}'.")


# ============================================================
# Step 2: Create a knowledge source wrapping the existing index
# ============================================================
def create_knowledge_source(index_client: SearchIndexClient) -> None:
    """Create a search index knowledge source pointing to our index."""
    knowledge_source = SearchIndexKnowledgeSource(
        name=KNOWLEDGE_SOURCE_NAME,
        description=(
            "Annual reports (FY 2024-25) for Apollo Hospitals Delhi, Data Patterns, "
            "IndiGo Airlines, Indigo Paints, KPEL, and Oracle Financial Services."
        ),
        search_index_parameters=SearchIndexKnowledgeSourceParameters(
            search_index_name=INDEX_NAME,
            semantic_configuration_name=SEMANTIC_CONFIG_NAME,
            source_data_fields=[
                SearchIndexFieldReference(name="source_file"),
                SearchIndexFieldReference(name="page_number"),
            ],
            search_fields=[
                SearchIndexFieldReference(name="content"),
            ],
        ),
    )

    index_client.create_or_update_knowledge_source(knowledge_source)
    print(f"[OK] Knowledge source '{KNOWLEDGE_SOURCE_NAME}' created.")


# ============================================================
# Step 3: Create a knowledge base (connects source + LLM)
# ============================================================
def create_knowledge_base(index_client: SearchIndexClient) -> None:
    """Create or update the knowledge base with LLM connection."""
    aoai_params = AzureOpenAIVectorizerParameters(
        resource_url=OPENAI_ENDPOINT.rstrip("/"),
        deployment_name=AGENTIC_DEPLOYMENT_NAME,
        model_name=AGENTIC_MODEL_NAME,
        api_key=OPENAI_KEY,
    )

    knowledge_base = KnowledgeBase(
        name=KNOWLEDGE_BASE_NAME,
        description=(
            "Knowledge base for querying annual reports from six Indian companies. "
            "Supports financial analysis, comparisons, and detailed report queries."
        ),
        retrieval_instructions=(
            "Search the annual reports knowledge source for any financial data, "
            "company information, risk factors, board details, or other corporate "
            "disclosures. Always cite the source file and page number."
        ),
        answer_instructions=(
            "You are an expert financial analyst. Provide detailed, accurate answers "
            "based on the retrieved documents. Use numbered references like [1], [2] "
            "in your answer. At the end, list a References section mapping each number "
            "to its source file and page. If information is not found, say so clearly."
        ),
        output_mode=KnowledgeRetrievalOutputMode.ANSWER_SYNTHESIS,
        knowledge_sources=[
            KnowledgeSourceReference(name=KNOWLEDGE_SOURCE_NAME),
        ],
        models=[KnowledgeBaseAzureOpenAIModel(azure_open_ai_parameters=aoai_params)],
        retrieval_reasoning_effort=KnowledgeRetrievalLowReasoningEffort(),
    )

    index_client.create_or_update_knowledge_base(knowledge_base)
    print(f"[OK] Knowledge base '{KNOWLEDGE_BASE_NAME}' created.")


# ============================================================
# Step 4: Query the knowledge base
# ============================================================
def query(question: str) -> dict:
    """Send a question to the knowledge base and return the response + metadata."""
    credential = AzureKeyCredential(SEARCH_KEY)
    kb_client = KnowledgeBaseRetrievalClient(
        endpoint=SEARCH_ENDPOINT,
        knowledge_base_name=KNOWLEDGE_BASE_NAME,
        credential=credential,
    )

    request = KnowledgeBaseRetrievalRequest(
        messages=[
            KnowledgeBaseMessage(
                role="user",
                content=[KnowledgeBaseMessageTextContent(text=question)],
            ),
        ],
        knowledge_source_params=[
            SearchIndexKnowledgeSourceParams(
                knowledge_source_name=KNOWLEDGE_SOURCE_NAME,
                include_references=True,
                include_reference_source_data=True,
            ),
        ],
        include_activity=True,
    )

    start = time.time()
    result = kb_client.retrieve(request)
    total_time = round(time.time() - start, 2)

    # Extract the answer text
    answer = ""
    if result.response and result.response[0].content:
        content = result.response[0].content[0]
        if hasattr(content, "text"):
            answer = content.text

    # Extract activity / search details if available
    activity = []
    if hasattr(result, "activity") and result.activity:
        for act in result.activity:
            activity.append(str(act))

    # Extract references
    references = []
    if hasattr(result, "references") and result.references:
        for ref in result.references:
            references.append(str(ref))

    return {
        "answer": answer,
        "total_time": total_time,
        "activity": activity,
        "references": references,
    }


# ============================================================
# Setup: run once to create knowledge source + knowledge base
# ============================================================
def setup():
    """One-time setup: add semantic config, create knowledge source and knowledge base."""
    index_client = get_index_client()

    print("\n--- Step 1: Adding semantic configuration to index ---")
    update_index_with_semantic_config(index_client)

    print("\n--- Step 2: Creating knowledge source ---")
    create_knowledge_source(index_client)

    print("\n--- Step 3: Creating knowledge base ---")
    create_knowledge_base(index_client)

    print("\n[DONE] Setup complete. You can now query.\n")


# ============================================================
# Interactive mode
# ============================================================
def main():
    import sys

    if "--setup" in sys.argv:
        setup()
        if "--query" not in sys.argv:
            return

    print("=== Azure AI Search Agentic Retrieval ===")
    print("Ask questions about the annual reports. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except EOFError:
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break

        print("\nSearching & synthesizing...\n")
        result = query(question)

        print(f"Answer ({result['total_time']}s):\n")
        print(result["answer"])

        if result["activity"]:
            print(f"\n--- Activity ({len(result['activity'])} steps) ---")
            for a in result["activity"]:
                print(f"  {a}")

        if result["references"]:
            print(f"\n--- References ({len(result['references'])}) ---")
            for r in result["references"]:
                print(f"  {r}")

        print()


if __name__ == "__main__":
    main()
