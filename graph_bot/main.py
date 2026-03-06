from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException

import genai
from shared.models import WorkerRequest, WorkerResponse


CHROMA_URL = os.getenv("CHROMA_URL", "http://chromadb:8000")

client = genai.Client()
app = FastAPI(title="Graph Bot")


async def _search_schema(query: str) -> List[str]:
    url = f"{CHROMA_URL}/collections/schema_graph_entities/query"
    payload = {"query": query, "n_results": 5}
    async with httpx.AsyncClient() as hc:
        resp = await hc.post(url, json=payload)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    return [d.get("document", "") for d in data.get("results", [])]


async def _generate_cypher(schemas: List[str], user_query: str) -> str:
    prompt = (
        "You are an expert Cypher query writer. Given the following graph schema descriptions delimited by ``` "
        "and a natural language question, produce a single Cypher query that answers the question. Only output the Cypher.\n\n"
        "Schemas:\n```\n" +
        "\n```\n".join(schemas) + "\n```\n\nQuestion:\n" + user_query
    )
    response = client.responses.create(model="gemini-2.5-flash", input=prompt)
    cypher_text = response.output_text if hasattr(
        response, "output_text") else ""
    if not cypher_text and response.output:
        cypher_text = "".join(str(item) for item in response.output)
    return cypher_text.strip()


@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest):
    schemas = await _search_schema(req.query)
    cypher = await _generate_cypher(schemas, req.query)
    # mock result
    dummy = {"nodes": [{"id": 1, "label": "Example"}], "relationships": []}
    resp = WorkerResponse(
        status="SUCCESS",
        data=dummy,
        citations=[cypher] if cypher else [],
        message="Cypher generated and returned mock graph data" if cypher else "No Cypher generated",
    )
    return resp


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
