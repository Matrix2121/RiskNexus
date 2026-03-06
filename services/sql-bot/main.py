import chromadb
from typing import List
from shared.models import WorkerRequest, WorkerResponse

# Initialize ChromaDB Client
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)

async def _search_schema(query: str) -> List[str]:
    """Retrieves relevant DDLs using the official client."""
    collection = chroma_client.get_collection(name="schema_sql_core")
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    # The client returns a list of lists for 'documents'
    return results.get("documents", [[]])[0]

# ... existing _generate_sql logic remains similar ...

@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest):
    ddls = await _search_schema(req.query)
    sql = await _generate_sql(ddls, req.query)
    
    # Mock result for PoC
    dummy_result = [{"id": 1, "value": "mock_data"}]
    return WorkerResponse(
        status="SUCCESS",
        data={"rows": dummy_result},
        citations=["schema_sql_core"],
        message=f"Generated SQL: {sql}"
    )