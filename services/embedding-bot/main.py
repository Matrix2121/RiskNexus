from __future__ import annotations

import io
import os
import json
import uuid
import asyncio
from typing import Any, Dict, List, Optional

import chromadb
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel

import google.genai as genai
from langsmith import wrappers, traceable, tracing_context

# LangChain imports for advanced metadata
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.sql_schema import RiskSQLSchema
from shared.graph_schema import RiskGraphSchema

# Models


class TextPayload(BaseModel):
    text: str
    source_metadata: Optional[Dict[str, Any]] = None


class DocumentMetadata(BaseModel):
    section_title: str
    effective_date: str
    topic_summary: str


app = FastAPI(title="Embedding Bot")
raw_client = genai.Client()
client = wrappers.wrap_gemini(raw_client)

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
    "sql_schema",
    "graph_schema",
    "regulations",
    "documents"
]

@app.post("/sync-schemas")
async def sync_schemas():
    """Reads the static Python schema definitions and embeds them into ChromaDB."""
    try:
        sql_schema = RiskSQLSchema()
        graph_schema = RiskGraphSchema()
        
        sql_docs = sql_schema.get_documents_for_embedding()
        graph_docs = graph_schema.get_documents_for_embedding()
        
        # 1. Запис на SQL схемата
        sql_collection = chroma_client.get_collection(name="sql_schema")
        for doc in sql_docs:
            vector = await _embed(doc["text"])
            # Използваме upsert вместо add, за да презаписваме старите данни, ако натиснем бутона пак
            sql_collection.upsert(
                ids=[doc["id"]],
                documents=[doc["text"]],
                metadatas=[doc["metadata"]],
                embeddings=[vector]
            )
            
        # 2. Запис на Graph схемата
        graph_collection = chroma_client.get_collection(name="graph_schema")
        for doc in graph_docs:
            vector = await _embed(doc["text"])
            graph_collection.upsert(
                ids=[doc["id"]],
                documents=[doc["text"]],
                metadatas=[doc["metadata"]],
                embeddings=[vector]
            )
            
        return {"status": "success", "message": "Schemas successfully synced to ChromaDB."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def initialize_collections():
    """Ensures all required collections exist on startup."""
    for collection_name in REQUIRED_COLLECTIONS:
        try:
            chroma_client.get_or_create_collection(name=collection_name)
            print(f"Collection '{collection_name}' is ready.")
        except Exception as e:
            print(f"Failed to initialize collection {collection_name}: {e}")


@traceable(run_type="llm", name="Metadata-Generation")
async def _generate_metadata(text: str) -> Dict[str, Any]:
    # use LangChain structured output to build DocumentMetadata
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    chain = llm.with_structured_output(DocumentMetadata)
    
    try:
        if tracer is not None:
            with tracer.as_default():
                result = await chain.ainvoke(text)
        else:
            result = await chain.ainvoke(text)
            
        # Pydantic v1/v2 compatibility
        if hasattr(result, "model_dump"):
            return result.model_dump()
        return result.dict()
    except Exception as e:
        # Fallback just in case the LLM refuses to answer or hits a rate limit
        print(f"Metadata extraction failed: {e}")
        return {
            "section_title": "unknown", 
            "effective_date": "unknown", 
            "topic_summary": "unknown"
        }


@traceable(run_type="embedding", name="Embedding-Call")
async def _embed(text: str) -> list[float]:
    if tracer is not None:
        with tracer.as_default():
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text
            )
    else:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
    return response.embeddings[0].values


@traceable(run_type="chain", name="Process-File-Upload")
async def _process_upload(collection_name: str, file: UploadFile) -> Dict[str, Any]:
    if collection_name not in REQUIRED_COLLECTIONS:
        raise HTTPException(status_code=404, detail="Unknown collection")

    contents = await file.read()
    if file.filename.lower().endswith(".txt"):
        text = contents.decode("utf-8", errors="ignore")
    elif file.filename.lower().endswith(".pdf"):
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(contents))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    base_metadata = {"filename": file.filename}
    # split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []
    embs: List[List[float]] = []

    for chunk in chunks:
        chunk_meta = await _generate_metadata(chunk)
        # merge with base metadata
        merged = {**{k: str(v) for k, v in base_metadata.items()},
                  **{k: str(v) for k, v in chunk_meta.items()}}
        embedding = await _embed(chunk)
        ids.append(str(uuid.uuid4()))
        docs.append(chunk)
        metas.append(merged)
        embs.append(embedding)

    collection = chroma_client.get_collection(name=collection_name)
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embs
    )
    return {"status": "success", "metadata_count": len(ids)}


@app.post("/upload/{collection_name}")
async def upload_file(collection_name: str, request: Request, file: UploadFile = File(...)):
    with tracing_context(parent=dict(request.headers)):
        return await _process_upload(collection_name, file)


@app.post("/embed/{collection_name}")
async def embed_generic(collection_name: str, payload: TextPayload, request: Request):
    with tracing_context(parent=request.headers):
        if collection_name not in REQUIRED_COLLECTIONS:
            raise HTTPException(status_code=404, detail="Unknown collection")

        metadata = await _generate_metadata(payload.text)
        if payload.source_metadata:
            metadata.update(payload.source_metadata)
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
