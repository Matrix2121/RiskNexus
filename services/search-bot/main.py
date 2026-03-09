from __future__ import annotations

from typing import List, Dict, Any
import asyncio
import json

from fastapi import FastAPI, HTTPException, Request
from langsmith import tracing_context

import google.genai as genai
from ddgs import DDGS

from shared.models import WorkerRequest, WorkerResponse
from langsmith import wrappers, traceable

app = FastAPI(title="Search Bot")

# 1. Инициализиране на клиента и обвиването му за LangSmith
raw_client = genai.Client()
client = wrappers.wrap_gemini(raw_client)

# 2. Декориране на функциите, които използват LLM, за да се виждат като "llm" в LangSmith
@traceable(run_type="llm", name="Search-Query-Formulation")
async def _formulate_query(user_query: str) -> str:
    prompt = (
        "You are an expert at converting natural language questions into concise web search queries. "
        "Given the question below, output a single short search string (no explanation).\n\n" +
        user_query
    )
    # Обвитият клиент автоматично изпраща usage_metadata към LangSmith
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    
    query = resp.text or ""
    return query.strip().strip('"')

async def _search_web(query: str) -> List[dict]:
    # Използваме bg-bg за български резултати, както обсъдихме по-рано
    def sync_search(q: str) -> List[dict]:
        return DDGS().text(q, max_results=3, region='bg-bg')

    results = await asyncio.to_thread(sync_search, query)
    return results

@traceable(run_type="llm", name="Search-Result-Summarization")
async def _summarize(snippets: List[str]) -> str:
    # Тъй като искате финалният резултат да е на български, добавяме инструкция тук
    prompt = (
        "Вие сте асистент за обобщаване на информация. Прочетете следните откъси от мрежата "
        "и изгответе кратко фактическо обобщение на БЪЛГАРСКИ ЕЗИК. "
        "Комбинирайте ги в няколко изречения.\n\n" + "\n---\n".join(snippets)
    )
    
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt) ##TO-DO Change to pro model
    
    text = resp.text or ""
    return text.strip()

@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest, request: Request):
    with tracing_context(parent=request.headers):
        try:
            # Стъпка 1: Формулиране на заявка за търсене
            search_string = await _formulate_query(req.query)
            if not search_string:
                raise HTTPException(
                    status_code=500, detail="Failed to construct search query")

            # Стъпка 2: Търсене в мрежата
            results = await _search_web(search_string)
            if not results:
                return WorkerResponse(status="SUCCESS", data={"summary": "няма намерени резултати"}, citations=[], message="no search results")

            # Стъпка 3: Обобщаване на резултатите
            snippets = [r.get("body", "") for r in results]
            summary = await _summarize(snippets)
            urls = [r.get("href") for r in results if r.get("href")]

            return WorkerResponse(
                status="SUCCESS", 
                data={"summary": summary}, 
                citations=urls, 
                message="search summary"
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)