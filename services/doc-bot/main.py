from __future__ import annotations

import os
from typing import List, Dict, Any

import chromadb
from fastapi import FastAPI, HTTPException

import google.genai as genai
from shared.models import WorkerRequest, WorkerResponse
from fastapi import FastAPI, HTTPException, Request
from langsmith import wrappers, traceable, tracing_context

# Initialize the official ChromaDB HTTP Client
# 'chromadb' is the hostname of the service in your docker-compose.yml
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)

# wrap genai client for usage metadata
raw_client = genai.Client()
client = wrappers.wrap_gemini(raw_client)

app = FastAPI(title="Documentation Bot")

# optional LangSmith tracer
try:
    from langchain import LangSmithTracer
    tracer = LangSmithTracer()
    tracer.service = "doc-bot"
except ImportError:
    tracer = None

@traceable(run_type="embedding", name="Embed-Query")
async def _embed_query(text: str) -> list[float]:
    """Converts the user's question into a Gemini vector."""
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


@traceable(run_type="retriever", name="ChromaDB-Search")
async def _search_docs(query: str, collection_name: str = "documents") -> Dict[str, Any]:
    """Searches ChromaDB using Gemini embeddings."""
    try:
        collection = chroma_client.get_collection(name=collection_name)

        # 1. Embed the search query using Gemini first
        query_vector = await _embed_query(query)

        # 2. Search using 'query_embeddings' instead of 'query_texts'
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=5,
            include=["documents", "metadatas"]
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"ChromaDB Query Failed: {str(e)}")

@traceable(run_type="llm", name="Doc-Bot-Answer")
async def _answer_from_chunks(chunks: List[str], user_query: str) -> str:
    if not chunks:
        return "I don't know."

    prompt = (
        "You are a compliance assistant. Answer the question using ONLY the provided chunks. "
        "Include inline citations in your response corresponding to the sources (e.g., '... [Source: policy.pdf]'). "
        "If the answer is not contained in the chunks, say 'I don't know'.\n\n" +
        "\n---\n".join(chunks) + "\n\nQuestion:\n" + user_query
    )

    if tracer is not None:
        with tracer.as_default():
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt)
    else:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
    answer = response.text or ""
    return answer.strip()


@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest, request: Request):
    with tracing_context(parent=request.headers):
        # determine collection dynamically
        collection = req.context.get("collection") if req.context else None
        if not collection:
            collection = "documents"
        # Step 1: Search ChromaDB using the new client logic
        results = await _search_docs(req.query, collection)

        # Check if we actually got documents back
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if not docs:
            return WorkerResponse(status="SUCCESS", data={"answer": ""}, citations=[], message="no relevant chunks")

        # build formatted chunks with source info and new metadata
        chunks: List[str] = []
        for idx, (text, md) in enumerate(zip(docs, metas), start=1):
            filename = md.get("filename", "Unknown Source")
            section = md.get("section_title", "N/A")
            date = md.get("effective_date", "N/A")
            summary_metadata = md.get("topic_summary", "")
            chunks.append(
                f"Chunk {idx}:\n"
                f"- Source file: {filename}\n"
                f"- Section: {section}\n"
                f"- Date: {date}\n"
                f"- Summary: {summary_metadata}\n"
                f"Text:\n{text}"
            )

        # Step 2: Generate answer
        answer = await _answer_from_chunks(chunks, req.query)

        # optionally build a simple citation list as before
        citations = []
        for md in metas:
            file = md.get("filename", "Unknown Source")
            page = md.get("page", "N/A")
            citations.append(f"{file} (Page {page})")

        return WorkerResponse(
            status="SUCCESS",
            data={"answer": answer},
            citations=citations,
            message="Generated response from internal documentation."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
