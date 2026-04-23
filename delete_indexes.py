import os
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

SEARCH_ENDPOINT = os.environ["AZURE_AI_SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["AZURE_AI_SEARCH_KEY"]

index_client = SearchIndexClient(endpoint=SEARCH_ENDPOINT, credential=AzureKeyCredential(SEARCH_KEY))

indexes_to_delete = ["annual-reports-index", "aragdoc"]

for index_name in indexes_to_delete:
    try:
        print(f"Deleting index: {index_name}")
        index_client.delete_index(index_name)
        print(f"Successfully deleted: {index_name}")
    except Exception as e:
        print(f"Error deleting {index_name}: {e}")
