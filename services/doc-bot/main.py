from __future__ import annotations

import os
from typing import List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException

import genai
from shared.models import WorkerRequest, WorkerResponse


CHROMA_URL = os.getenv("CHROMA_URL", "http://chromadb:8000")
client = genai.Client()
app = FastAPI(title="Documentation Bot")


async def _search_docs(query: str) -> List[Dict[str, Any]]:
    # search regulations and contracts collections; could choose based on context
    url = f"{CHROMA_URL}/collections/docs_regulations/query"
    payload = {"query": query, "n_results": 5}
    async with httpx.AsyncClient() as hc:
        resp = await hc.post(url, json=payload)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    return data.get("results", [])


async def _answer_from_chunks(chunks: List[Dict[str, Any]], user_query: str) -> str:
    texts = []
    for c in chunks:
        texts.append(c.get("document", ""))
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the following text chunks. "
        "If the answer is not contained in the chunks, say 'I don't know'.\n\n" +
        "\n---\n".join(texts) + "\n\nQuestion:\n" + user_query
    )
    response = client.responses.create(model="gemini-2.5-flash", input=prompt)
    answer = response.output_text if hasattr(response, "output_text") else ""
    if not answer and response.output:
        answer = "".join(str(item) for item in response.output)
    return answer.strip()


@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest):
    results = await _search_docs(req.query)
    if not results:
        return WorkerResponse(status="SUCCESS", data={"answer": ""}, citations=[], message="no relevant chunks")
    answer = await _answer_from_chunks(results, req.query)
    # build citations from metadata
    citations = []
    for r in results:
        md = r.get("metadata", {})
        file = md.get("filename")
        page = md.get("page")
        if file or page:
            citations.append(f"{file or ''}:{page or ''}")
    return WorkerResponse(status="SUCCESS", data={"answer": answer}, citations=citations, message="generated response")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
