from __future__ import annotations

import os
import json
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException

import genai
from shared.models import WorkerRequest, WorkerResponse


CHROMA_URL = os.getenv("CHROMA_URL", "http://chromadb:8000")

# initialize Google GenAI client
client = genai.Client()

app = FastAPI(title="SQL Bot")


async def _search_schema(query: str) -> List[str]:
    # very simple search using chroma's search endpoint
    url = f"{CHROMA_URL}/collections/schema_sql_core/query"
    payload = {"query": query, "n_results": 5}
    async with httpx.AsyncClient() as hc:
        resp = await hc.post(url, json=payload)
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    data = resp.json()
    # assume the results have documents containing the DDL text
    return [d.get("document", "") for d in data.get("results", [])]


async def _generate_sql(ddls: List[str], user_query: str) -> str:
    prompt = (
        "You are an expert SQL generator. Given the following database schema DDLs delimited by ``` and a natural language question, "
        "produce a single SQL query that answers the question. Only output the SQL.\n\n"  
        "DDLs:\n```\n" + "\n```\n".join(ddls) + "\n```\n\nQuestion:\n" + user_query
    )
    response = client.responses.create(model="gemini-2.5-flash", input=prompt)
    # simplify assumption about output
    sql_text = response.output_text if hasattr(response, "output_text") else ""
    if not sql_text and response.output:
        sql_text = "".join(str(item) for item in response.output)
    return sql_text.strip()


@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest):
    # step 1: get schema DDLs relevant to the query
    ddls = await _search_schema(req.query)
    # step 2: ask LLM to generate sql
    sql = await _generate_sql(ddls, req.query)
    # step 3: execute sql against real database (mocked)
    dummy_result = [{"id": 1, "value": "mock"}]
    resp = WorkerResponse(
        status="SUCCESS",
        data={"rows": dummy_result},
        citations=["schema_sql_core"],
        message=f"Generated SQL and returned mock rows: {sql}" if sql else "No SQL generated",
    )
    return resp


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
