from __future__ import annotations
from psycopg2.extras import RealDictCursor
import psycopg2
import os
import asyncio
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
import google.genai as genai
import chromadb
from shared.models import WorkerRequest, WorkerResponse
from langsmith import wrappers, traceable, tracing_context

# initialize ChromaDB client for schema lookups
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)


app = FastAPI(title="SQL Bot")
raw_client = genai.Client()
client = wrappers.wrap_gemini(raw_client)

# optional LangSmith tracer
try:
    from langchain import LangSmithTracer
    tracer = LangSmithTracer()
    tracer.service = "sql-bot"
except ImportError:
    tracer = None


def _clean_code(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    return t.strip("`")


@traceable(run_type="embedding", name="Embed-Query")
async def _embed_query(text: str) -> list[float]:
    resp = client.models.embed_content(
        model="gemini-embedding-001", contents=text)
    return resp.embeddings[0].values


@traceable(run_type="tool", name="Retrieve-SQL-Schema")
async def _get_relevant_schema(query: str, collection_name: str = "sql_schema") -> str:
    emb = await _embed_query(query)
    coll = chroma_client.get_collection(name=collection_name)
    
    # Издърпваме до 10 резултата (ако имате повече таблици в бъдеще)
    results = coll.query(query_embeddings=[emb], n_results=10)
    
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0] # По-малко число = по-добро съвпадение
    
    relevant_docs = []
    
    if distances:
        # Взимаме резултата на най-доброто съвпадение
        best_score = distances[0]
        
        # Динамичен филтър: Взимаме най-добрата таблица + всички други, 
        # чието "разстояние" е много близко до най-доброто (напр. разлика до 0.15)
        for doc, dist in zip(docs, distances):
            if dist <= best_score + 0.15:
                relevant_docs.append(str(doc))
                
    return "\n\n".join(relevant_docs)


@traceable(run_type="llm", name="SQL-Generation")
async def _generate_sql(user_query: str, schema_context: str) -> str:
    prompt = (
        "You are a SQL generator for a PostgreSQL database. "
        "Based on the provided schema context and the user's question, "
        "return a valid SQL query. Output ONLY the SQL code with no explanations and strip any markdown backticks.\n\n"
        f"Schema context:\n{schema_context}\n\n"
        f"User question:\n{user_query}"
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    return _clean_code(resp.text or "")


@traceable(run_type="tool", name="Postgres-Execution")
def _run_postgresql(sql_query: str) -> List[Dict[str, Any]]:
    # synchronous helper executed in thread
    conn = psycopg2.connect(
        dbname=os.environ.get("POSTGRES_DB"),
        user=os.environ.get("POSTGRES_USER"),
        password=os.environ.get("POSTGRES_PASSWORD"),
        host=os.environ.get("POSTGRES_HOST"),
        port=os.environ.get("POSTGRES_PORT"),
    )
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_query)
            return cur.fetchall()
    finally:
        conn.close()


@traceable(run_type="llm", name="SQL-Synthesis")
async def _synthesize_answer(rows: List[Dict[str, Any]]) -> str:
    prompt = (
        "Предоставете кратко обобщение на български език въз основа на резултатите по-долу:\n"
        f"{rows}"
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    return resp.text or ""


@traceable(run_type="tool", name="SQL-Bot-Execution")
async def _process_sql_query(query: str) -> WorkerResponse:
    # retrieve only relevant schema chunks from ChromaDB
    schema_ctx = await _get_relevant_schema(query)

    # generate SQL
    sql = await _generate_sql(query, schema_ctx)
    if not sql:
        return WorkerResponse(status="ERROR", message="LLM failed to produce SQL.")

    # run SQL
    try:
        rows = await asyncio.to_thread(_run_postgresql, sql)
    except Exception as e:
        return WorkerResponse(
            status="ERROR",
            message=f"SQL execution failed: {sql}. Error: {e}",
        )

    # synthesize answer
    answer = await _synthesize_answer(rows)
    return WorkerResponse(status="SUCCESS", data={"rows": rows, "answer": answer}, message="Query executed.")


@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest, request: Request):
    with tracing_context(parent=request.headers):
        return await _process_sql_query(req.query)
