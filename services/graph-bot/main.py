from __future__ import annotations

import os
import asyncio
from typing import Any, Dict, List

from fastapi import FastAPI, Request
from langsmith import tracing_context, wrappers, traceable
import google.genai as genai
from shared.models import WorkerRequest, WorkerResponse
from shared.graph_schema import RiskGraphSchema
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv(override=True)

app = FastAPI(title="Graph Bot")

# LLM client wrapped for LangSmith
raw_client = genai.Client()
client = wrappers.wrap_gemini(raw_client)

# optional LangSmith tracer
try:
    from langchain import LangSmithTracer
    tracer = LangSmithTracer()
    tracer.service = "graph-bot"
except ImportError:
    tracer = None


def _clean_code(text: str) -> str:
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t[3:-3].strip()
    return t.strip("`")


@traceable(run_type="llm", name="Graph-Cypher-Generation")
async def _generate_cypher(user_query: str, schema_context: str) -> str:
    prompt = (
        "You are a Neo4j query generator. Given schema information and a user question, "
        "write a valid Cypher query. Return ONLY the Cypher query, no explanation, "
        "and strip any markdown backticks.\n\n"
        f"Schema:\n{schema_context}\n\n"
        f"User question:\n{user_query}"
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    cypher = resp.text or ""
    return _clean_code(cypher)


def _get_driver():
    uri = os.environ.get("NEO4J_URI")
    user = os.environ.get("NEO4J_USER")
    password = os.environ.get("NEO4J_PASSWORD")
    if not uri or not user or not password:
        raise RuntimeError(
            "NEO4J_URI, NEO4J_USER and NEO4J_PASSWORD must be set.")
    return GraphDatabase.driver(uri, auth=(user, password))


# Adding @traceable here so the database call itself is also tracked in the tree
@traceable(run_type="tool", name="Neo4j-Execution")
async def _run_neo4j(cypher: str) -> List[Dict[str, Any]]:
    def sync_query():
        driver = _get_driver()
        with driver.session() as session:
            result = session.run(cypher)
            return [record.data() for record in result]
    return await asyncio.to_thread(sync_query)


@traceable(run_type="llm", name="Graph-Answer-Synthesis")
async def _synthesize_answer(rows: List[Dict[str, Any]]) -> str:
    prompt = (
        "Предоставете кратък отговор на български език, базиран на резултатите по-долу:\n"
        f"{rows}"
    )
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    return resp.text or ""


# 1. Extract the core logic into its own traceable function
@traceable(run_type="tool", name="Graph-Bot-Execution")
async def _process_graph_query(query: str) -> WorkerResponse:
    schema = RiskGraphSchema()
    schema_ctx = schema.get_cypher_context()
    
    cypher = await _generate_cypher(query, schema_ctx)
    if not cypher:
        return WorkerResponse(
            status="ERROR",
            message="LLM failed to generate a Cypher query."
        )
        
    try:
        rows = await _run_neo4j(cypher)
    except Exception as e:
        return WorkerResponse(
            status="ERROR",
            message=f"Invalid Cypher: {cypher}. Error: {e}"
        )
        
    answer = await _synthesize_answer(rows)
    return WorkerResponse(
        status="SUCCESS",
        data={"answer": answer},
        message="Graph query executed."
    )


# 2. Remove @traceable from the endpoint. 
# Establish the context first, THEN call the traced logic.
@app.post("/query", response_model=WorkerResponse)
async def handle_query(req: WorkerRequest, request: Request):
    with tracing_context(parent=request.headers):
        return await _process_graph_query(req.query)