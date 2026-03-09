from __future__ import annotations
import os
from typing import Any, Dict, List
import chromadb
from fastapi import FastAPI
import google.genai as genai
from shared.models import WorkerRequest, WorkerResponse

app = FastAPI(title="Graph Bot")
client = genai.Client()
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)


async def _search_schema(query: str) -> List[str]:
    collection = chroma_client.get_collection(name="schema_graph_entities")
    results = collection.query(query_texts=[query], n_results=5)
    return results.get("documents", [[]])[0]


@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest):
    schemas = await _search_schema(req.query)
    # logic for cypher generation...
    return WorkerResponse(status="SUCCESS", data={}, message="Graph data retrieved")
