"""
Script to create an Azure AI Search hybrid index from PDF documents.

Steps:
1. Extract text from PDFs using Azure Document Intelligence (cached locally as JSON)
2. Chunk the extracted text (1000 chars, 200 char overlap)
3. Generate embeddings using text-embedding-3-large via Azure OpenAI
4. Create/update the search index in Azure AI Search
5. Upload chunked documents with embeddings
"""

import hashlib
import json
import os
from pathlib import Path

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

# --- Configuration ---
SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_AI_SEARCH_KEY"]
DOC_INTEL_ENDPOINT = os.environ["AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"]
DOC_INTEL_KEY = os.environ["AZURE_DOCUMENT_INTELLIGENCE_KEY"]
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_KEY = os.environ["AZURE_OPENAI_KEY"]
EMBEDDING_MODEL = os.environ["AZURE_AI_EMBEDDING_MODEL"]

INDEX_NAME = "annual-reports-index"
PDF_DIR = Path("pdfs")
CACHE_DIR = Path("doc_intel_cache")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_DIMENSIONS = 3072  # text-embedding-3-large

# Set to process only one PDF for testing; set to None to process all
TEST_SINGLE_PDF = None


def get_embedding_client() -> AzureOpenAI:
    """Create an Azure OpenAI client using API key."""
    return AzureOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_KEY,
        api_version="2024-12-01-preview",
    )


def generate_embeddings(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract text from PDF using Azure Document Intelligence, with local JSON caching."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{pdf_path.stem}.json"

    # Return cached result if available
    if cache_file.exists():
        print(f"  Using cached extraction: {cache_file.name}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Call Document Intelligence
    credential = AzureKeyCredential(DOC_INTEL_KEY)
    client = DocumentIntelligenceClient(DOC_INTEL_ENDPOINT, credential)

    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-read",
            body=f,
        )

    result = poller.result()

    pages = []
    if result.pages:
        for page in result.pages:
            page_text = ""
            if page.lines:
                page_text = "\n".join(line.content for line in page.lines)
            pages.append({
                "page_number": page.page_number,
                "text": page_text,
            })

    # Save to cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"  Cached extraction to: {cache_file.name}")

    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap

    return chunks


def create_search_index(index_client: SearchIndexClient) -> None:
    """Create or update the search index with hybrid (keyword + vector) fields."""
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SimpleField(name="source_file", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="default-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="default-hnsw"),
        ],
        profiles=[
            VectorSearchProfile(
                name="default-vector-profile",
                algorithm_configuration_name="default-hnsw",
            ),
        ],
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
    )

    index_client.create_or_update_index(index)
    print(f"Index '{INDEX_NAME}' created/updated successfully.")


def generate_chunk_id(source_file: str, page_number: int, chunk_index: int) -> str:
    """Generate a deterministic, URL-safe ID for a chunk."""
    raw = f"{source_file}_{page_number}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def main():
    # --- 1. Create the search index ---
    search_credential = AzureKeyCredential(SEARCH_KEY)
    index_client = SearchIndexClient(SEARCH_ENDPOINT, search_credential)
    create_search_index(index_client)

    # --- 2. Initialize clients ---
    search_client = SearchClient(SEARCH_ENDPOINT, INDEX_NAME, search_credential)
    embedding_client = get_embedding_client()

    # --- 3. Process PDFs ---
    if TEST_SINGLE_PDF:
        pdf_files = [PDF_DIR / TEST_SINGLE_PDF]
        print(f"Testing with single PDF: {TEST_SINGLE_PDF}")
    else:
        pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF(s) to process.")

    all_documents = []

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        # Extract text per page
        pages = extract_text_from_pdf(pdf_path)
        print(f"  Extracted {len(pages)} pages.")

        # Chunk each page
        for page_info in pages:
            chunks = chunk_text(page_info["text"])
            for chunk_idx, chunk_text_content in enumerate(chunks):
                doc_id = generate_chunk_id(pdf_path.name, page_info["page_number"], chunk_idx)
                all_documents.append({
                    "id": doc_id,
                    "content": chunk_text_content,
                    "source_file": pdf_path.name,
                    "page_number": page_info["page_number"],
                    "chunk_index": chunk_idx,
                    "_chunk_text": chunk_text_content,  # temporary, for embedding
                })

    print(f"\nTotal chunks to embed and upload: {len(all_documents)}")

    # --- 4. Generate embeddings in batches ---
    BATCH_SIZE = 16
    for i in range(0, len(all_documents), BATCH_SIZE):
        batch = all_documents[i : i + BATCH_SIZE]
        texts = [doc["_chunk_text"] for doc in batch]
        embeddings = generate_embeddings(embedding_client, texts)

        for doc, emb in zip(batch, embeddings):
            doc["embedding"] = emb
            del doc["_chunk_text"]

        print(f"  Embedded batch {i // BATCH_SIZE + 1}/{(len(all_documents) + BATCH_SIZE - 1) // BATCH_SIZE}")

    # --- 5. Upload to Azure AI Search in batches ---
    UPLOAD_BATCH_SIZE = 100
    for i in range(0, len(all_documents), UPLOAD_BATCH_SIZE):
        batch = all_documents[i : i + UPLOAD_BATCH_SIZE]
        result = search_client.upload_documents(documents=batch)
        succeeded = sum(1 for r in result if r.succeeded)
        print(f"  Uploaded batch {i // UPLOAD_BATCH_SIZE + 1}: {succeeded}/{len(batch)} succeeded")

    print(f"\nDone! Index '{INDEX_NAME}' populated with {len(all_documents)} chunks.")


if __name__ == "__main__":
    main()
