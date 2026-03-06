from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI

import genai
from shared.models import AgentState, WorkerRequest, WorkerResponse

# LangGraph imports (assumed available via pip install langgraph)
from langgraph import StateGraph, Node

app = FastAPI(title="RiskNexus Orchestrator")
client = genai.Client()

# --- Router node -----------------------------------------------------------


async def router(state: AgentState) -> AgentState:
    # ask the LLM which worker bots are required
    prompt = (
        "You are a router determining which microservice workers should handle a user query. "
        "Return a JSON array of service names from [\"sql-bot\", \"graph-bot\", \"search-bot\", "
        "\"doc-bot\", \"decision-bot\", \"calc-bot\", \"embedding-bot\"] that are needed.\n\n"
        f"Query:\n{state['query']}"
    )
    resp = client.responses.create(model="gemini-2.5-pro", input=prompt)
    text = resp.output_text if hasattr(resp, "output_text") else ""
    if not text and resp.output:
        text = "".join(str(item) for item in resp.output)
    try:
        names = json.loads(text)
    except Exception:
        names = []
    state["required_workers"] = names
    return state


# --- Parallel executor node ------------------------------------------------
# --- Parallel executor node ------------------------------------------------
async def execute_workers(state: AgentState) -> AgentState:
    raw_workers: List[str] = state.get("required_workers", []) or []
    query = state.get("query", "")
    ctx = state.get("user_context", {}) or {}

    # 1. Define the whitelist of valid service hostnames
    VALID_WORKERS = {
        "sql-bot", 
        "graph-bot", 
        "search-bot", 
        "doc-bot", 
        "decision-bot", 
        "calc-bot", 
        "embedding-bot"
    }

    # 2. Sanitize and filter the LLM's output
    sanitized_workers = set()
    for w in raw_workers:
        # Replace accidental underscores with dashes
        clean_name = str(w).replace("_", "-").strip().lower()
        if clean_name in VALID_WORKERS:
            sanitized_workers.add(clean_name)

    async with httpx.AsyncClient(timeout=3600.0) as hc:
        tasks = []
        # 3. Use the sanitized list to build the URLs
        for w in sanitized_workers:
            url = f"http://{w}:8000/query"
            req_obj = WorkerRequest(query=query, context=ctx)
            tasks.append(hc.post(url, json=req_obj.model_dump()))
        
        # Execute all valid tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

    results: List[Dict[str, Any]] = []
    for r in responses:
        if isinstance(r, Exception):
            results.append({"status": "ERROR", "message": str(r)})
        else:
            try:
                results.append(r.json())
            except Exception as e:
                results.append({"status": "ERROR", "message": str(e)})

    state["worker_responses"] = results
    return state

# --- Synthesis node --------------------------------------------------------
async def synthesize(state: AgentState) -> AgentState:
    responses = state.get("worker_responses", []) or []
    # simple aggregation for PoC: concatenate messages
    summary_parts: List[str] = []
    for r in responses:
        if isinstance(r, dict):
            summary_parts.append(r.get("message", ""))
    state["final_summary"] = "\n".join(summary_parts)
    return state


# --- Assemble LangGraph ----------------------------------------------------
graph = StateGraph(initial_state_type=AgentState)

graph.add_node(Node("router", router))
graph.add_node(Node("execute", execute_workers, parents=["router"]))
graph.add_node(Node("synthesize", synthesize, parents=["execute"]))


# --- FastAPI endpoint ------------------------------------------------------
@app.post("/chat")
async def chat(req: WorkerRequest) -> Dict[str, Any]:
    # seed the state
    state: AgentState = AgentState(
        query=req.query,
        user_context=req.context or {},
        required_workers=[],
    )

    finished = await graph.run(state)
    return {
        "final_summary": finished.get("final_summary"),
        "worker_responses": finished.get("worker_responses"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
