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


async def _search_docs(query: str) -> Dict[str, Any]:
    """Uses the official client to query the collection by name."""
    try:
        # Step 1: Get the collection object (Client handles UUID lookup automatically)
        collection = chroma_client.get_collection(name="docs_regulations")

        # Step 2: Perform the semantic search
        results = collection.query(
            query_texts=[query],
            n_results=5,
            include=["documents", "metadatas"]
        )
        return results
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"ChromaDB Query Failed: {str(e)}")


@traceable(run_type="llm", name="Doc-Bot-Answer")
async def _answer_from_chunks(results: Dict[str, Any], user_query: str) -> str:
    # results['documents'] is a list of lists; we take the first list [0]
    texts = results.get("documents", [[]])[0]

    if not texts:
        return "I don't know."

    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the following text chunks. "
        "If the answer is not contained in the chunks, say 'I don't know'.\n\n" +
        "\n---\n".join(texts) + "\n\nQuestion:\n" + user_query
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
        # Step 1: Search ChromaDB using the new client logic
        results = await _search_docs(req.query)

        # Check if we actually got documents back
        if not results.get("documents") or not results["documents"][0]:
            return WorkerResponse(status="SUCCESS", data={"answer": ""}, citations=[], message="no relevant chunks")

        # Step 2: Generate answer
        answer = await _answer_from_chunks(results, req.query)

        # Step 3: Build citations from the structured metadata returned by the client
        citations = []
        # metadatas is also a list of lists
        for md in results.get("metadatas", [[]])[0]:
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
