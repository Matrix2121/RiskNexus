import chromadb
from shared.models import WorkerRequest, WorkerResponse

chroma_client = chromadb.HttpClient(host="chromadb", port=8000)

async def _search_schema(query: str) -> List[str]:
    """Retrieves graph ontology using the official client."""
    collection = chroma_client.get_collection(name="schema_graph_entities")
    results = collection.query(
        query_texts=[query],
        n_results=5
    )
    return results.get("documents", [[]])[0]

# ... existing _generate_cypher logic remains similar ...

@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest):
    schemas = await _search_schema(req.query)
    cypher = await _generate_cypher(schemas, req.query)
    
    return WorkerResponse(
        status="SUCCESS",
        data={"nodes": [], "edges": []},
        citations=[cypher] if cypher else [],
        message="Cypher generated and returned mock graph data"
    )