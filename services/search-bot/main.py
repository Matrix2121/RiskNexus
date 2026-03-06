from __future__ import annotations

from typing import List
import asyncio

from fastapi import FastAPI, HTTPException

import genai
from duckduckgo_search import DDGS

from shared.models import WorkerRequest, WorkerResponse

app = FastAPI(title="Search Bot")
client = genai.Client()


async def _formulate_query(user_query: str) -> str:
    prompt = (
        "You are an expert at converting natural language questions into concise web search queries. "
        "Given the question below, output a single short search string (no explanation).\n\n" +
        user_query
    )
    resp = client.responses.create(model="gemini-2.5-flash", input=prompt)
    query = resp.output_text if hasattr(resp, "output_text") else ""
    if not query and resp.output:
        query = "".join(str(item) for item in resp.output)
    return query.strip().strip('"')


async def _search_web(query: str) -> List[dict]:
    # run DDGS search asynchronously
    results: List[dict] = []
    async for r in DDGS().text(query, max_results=3):
        results.append(r)
    return results


async def _summarize(snippets: List[str]) -> str:
    prompt = (
        "You are a summarization assistant. Read the following web snippets and produce a concise factual summary. "
        "Combine them into a few sentences.\n\n" + "\n---\n".join(snippets)
    )
    resp = client.responses.create(model="gemini-2.5-flash", input=prompt)
    text = resp.output_text if hasattr(resp, "output_text") else ""
    if not text and resp.output:
        text = "".join(str(item) for item in resp.output)
    return text.strip()


@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest):
    try:
        search_string = await _formulate_query(req.query)
        if not search_string:
            raise HTTPException(
                status_code=500, detail="Failed to construct search query")

        results = await _search_web(search_string)
        if not results:
            return WorkerResponse(status="SUCCESS", data={"summary": "no results"}, citations=[], message="no search results")

        snippets = [r.get("body", "") for r in results]
        summary = await _summarize(snippets)
        urls = [r.get("href") for r in results if r.get("href")]

        return WorkerResponse(status="SUCCESS", data={"summary": summary}, citations=urls, message="search summary")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
