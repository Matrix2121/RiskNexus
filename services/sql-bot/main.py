from __future__ import annotations
import os
from typing import Any, Dict, List
import chromadb
from fastapi import FastAPI, HTTPException
import google.genai as genai
from shared.models import WorkerRequest, WorkerResponse

app = FastAPI(title="SQL Bot")
client = genai.Client()

# optional LangSmith tracer
try:
    from langchain import LangSmithTracer
    tracer = LangSmithTracer()
    tracer.service = "sql-bot"
except ImportError:
    tracer = None
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)


async def _search_schema(query: str) -> List[str]:
    collection = chroma_client.get_collection(name="schema_sql_core")
    results = collection.query(query_texts=[query], n_results=5)
    return results.get("documents", [[]])[0]


async def _generate_sql(ddls: List[str], user_query: str) -> str:
    prompt = f"Given these DDLs: {' '.join(ddls)}. Generate SQL for: {user_query}. Output ONLY SQL."
    if tracer is not None:
        with tracer.as_default():
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt)
    else:
        response = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt)
    return (response.text or "").strip()


@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest):
    ddls = await _search_schema(req.query)
    sql = await _generate_sql(ddls, req.query)
    return WorkerResponse(status="SUCCESS", data={"rows": []}, message=f"Generated SQL: {sql}")
