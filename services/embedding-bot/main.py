from __future__ import annotations

import os
import json
import uuid
from typing import Any, Dict

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import google.genai as genai

# Models


class TextPayload(BaseModel):
    text: str


app = FastAPI(title="Embedding Bot")
client = genai.Client()

# optional LangSmith tracing
try:
    from langchain import LangSmithTracer
    tracer = LangSmithTracer()
    tracer.service = "embedding-bot"
except ImportError:
    tracer = None

# Initialize ChromaDB HTTP Client
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)

# Defined collections from the spec
REQUIRED_COLLECTIONS = [
    "documents",
    "sql-schemas",
    "graph-schemas",
    "docs_regulations",
    "docs_contracts",
    "schema_sql_core",
    "schema_graph_entities"
]


@app.on_event("startup")
async def initialize_collections():
    """Ensures all required collections exist on startup."""
    for collection_name in REQUIRED_COLLECTIONS:
        try:
            chroma_client.get_or_create_collection(name=collection_name)
            print(f"Collection '{collection_name}' is ready.")
        except Exception as e:
            print(f"Failed to initialize collection {collection_name}: {e}")


async def _generate_metadata(text: str) -> Dict[str, Any]:
    prompt = (
        "Given the following text, return a JSON object with two keys: "
        "`summary` (a brief natural-language summary) and `tags` (a list of relevant short keywords). "
        "The output must be valid JSON and nothing else.\nText:\n" + text
    )
    if tracer is not None:
        with tracer.as_default():
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite", contents=prompt)
    else:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt)
    raw = response.text or ""
    try:
        return json.loads(raw.strip().strip("```json").strip("```"))
    except Exception:
        return {"summary": "Extraction failed", "tags": []}


async def _embed(text: str) -> list[float]:
    if tracer is not None:
        with tracer.as_default():
            emb = client.embeddings.create(
                model="gemini-embedding-001", input=text)
    else:
        emb = client.embeddings.create(
            model="gemini-embedding-001", input=text)
    return emb.data[0].embedding


@app.post("/embed/{collection_name}")
async def embed_generic(collection_name: str, payload: TextPayload):
    if collection_name not in REQUIRED_COLLECTIONS:
        raise HTTPException(status_code=404, detail="Unknown collection")

    metadata = await _generate_metadata(payload.text)
    vector = await _embed(payload.text)

    # Use client to add data
    collection = chroma_client.get_collection(name=collection_name)
    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[payload.text],
        metadatas=[metadata],
        embeddings=[vector]
    )
    return {"status": "success", "metadata": metadata}

# (Existing convenience wrappers /embed/documents, etc. would follow here)
