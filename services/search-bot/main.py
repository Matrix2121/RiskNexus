from __future__ import annotations

from typing import List
import asyncio

from fastapi import FastAPI, HTTPException

import google.genai as genai
from ddgs import DDGS

from shared.models import WorkerRequest, WorkerResponse

app = FastAPI(title="Search Bot")
client = genai.Client()

# LangSmith tracing support (optional)
try:
    from langchain import LangSmithTracer
    tracer = LangSmithTracer()
    tracer.service = "search-bot"
except ImportError:
    tracer = None


async def _formulate_query(user_query: str) -> str:
    prompt = (
        "You are an expert at converting natural language questions into concise web search queries. "
        "Given the question below, output a single short search string (no explanation).\n\n" +
        user_query
    )
    if tracer is not None:
        with tracer.as_default():
            resp = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt)
    else:
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
    query = resp.text or ""
    return query.strip().strip('"')


async def _search_web(query: str) -> List[dict]:
    # the DDGS client is synchronous; run it in a thread to avoid blocking the event loop
    def sync_search(q: str) -> List[dict]:
        return DDGS().text(q, max_results=3, region='bg-bg')

    results = await asyncio.to_thread(sync_search, query)
    return results


async def _summarize(snippets: List[str]) -> str:
    prompt = (
        "You are a summarization assistant. Read the following web snippets and produce a concise factual summary. "
        "Combine them into a few sentences.\n\n" + "\n---\n".join(snippets)
    )
    if tracer is not None:
        with tracer.as_default():
            resp = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt)
    else:
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
    text = resp.text or ""
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
