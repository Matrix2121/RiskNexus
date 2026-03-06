from __future__ import annotations

import os
import json
import uuid
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import genai


class TextPayload(BaseModel):
    text: str


CHROMA_URL = os.getenv("CHROMA_URL", "http://chromadb:8000")

# initialize Google GenAI client using environment variable for API key
client = genai.Client()

app = FastAPI(title="Embedding Bot")


async def _add_to_chroma(collection: str, text: str, metadata: Dict[str, Any], embedding: list[float]):
    url = f"{CHROMA_URL}/collections/{collection}/add"
    payload = {
        "ids": [str(uuid.uuid4())],
        "documents": [text],
        "metadatas": [metadata],
        "embeddings": [embedding],
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return resp.json()


async def _generate_metadata(text: str) -> Dict[str, Any]:
    prompt = (
        "Given the following text, return a JSON object with two keys: "
        "`summary` (a brief natural-language summary) and `tags` (a list of relevant short keywords). "
        "The output must be valid JSON and nothing else.\nText:\n" + text
    )
    response = client.responses.create(
        model="gemini-2.5-flash-lite",
        input=prompt,
    )
    # the SDK returns nested output; extract the content
    # depending on SDK version this might differ; we'll assume the first text output is in 'output_text'
    raw = response.output_text if hasattr(response, "output_text") else ""
    if not raw and response.output:
        # fallback path, iterate through output
        raw = "".join(str(item) for item in response.output)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # if LLM didn't obey, try to strip surrounding backticks
        cleaned = raw.strip().strip("`\n ")
        return json.loads(cleaned)


async def _embed(text: str) -> list[float]:
    emb = client.embeddings.create(model="gemini-embedding-001", input=text)
    # result contains data list
    return emb.data[0].embedding


@app.post("/embed/{collection}")
async def embed_generic(collection: str, payload: TextPayload):
    if collection not in {"documents", "sql-schemas", "graph-schemas"}:
        raise HTTPException(status_code=404, detail="Unknown collection")
    metadata = await _generate_metadata(payload.text)
    vector = await _embed(payload.text)
    return await _add_to_chroma(collection, payload.text, metadata, vector)


# convenience wrappers for the three endpoints
@app.post("/embed/documents")
async def embed_documents(payload: TextPayload):
    return await embed_generic("documents", payload)


@app.post("/embed/sql-schemas")
async def embed_sql_schemas(payload: TextPayload):
    return await embed_generic("sql-schemas", payload)


@app.post("/embed/graph-schemas")
async def embed_graph_schemas(payload: TextPayload):
    return await embed_generic("graph-schemas", payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
