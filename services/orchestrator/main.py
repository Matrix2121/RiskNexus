from __future__ import annotations
from langgraph.graph import START

import asyncio
import json
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI

import google.genai as genai
from shared.models import AgentState, WorkerRequest, WorkerResponse

# LangGraph imports (assumed available via pip install langgraph)
from langgraph.graph import StateGraph

from langsmith import wrappers, traceable, get_current_run_tree

app = FastAPI(title="RiskNexus Orchestrator")
raw_client = genai.Client()

# wrap the underlying client so that usage_metadata (tokens, cost, etc.)
# emitted by Gemini responses is automatically collected by LangSmith
client = wrappers.wrap_gemini(raw_client)


# helper used by the router; decorated so that the run appears as an
# "llm" span in the LangSmith dashboard and token usage is measured.
@traceable("llm")
async def call_router_llm(prompt: str) -> str:
    # the wrapped client handles telemetry; just return the text
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt)
    return resp.text or ""

# --- LangSmith tracing ----------------------------------------------------
# if langchain/langsmith are installed and the environment variables are set,
# this tracer will capture the execution of the graph and any LangChain
# operations executed within the `with tracer.as_default():` context.  The
# `.env` file already contains LANGSMITH_* variables.
try:
    from langchain import LangSmithTracer
    tracer = LangSmithTracer()
    # label the service so traces show up nicely in the portal
    tracer.service = "orchestrator"
except ImportError:
    tracer = None
    # tracing is optional; absence will not crash the service

# --- Router node -----------------------------------------------------------


# the graph library now expects synchronous callables for
# compiled_graph.invoke; since we are using aasync invocation below we
# can keep the nodes async. Alternatively, these could be regular def
# functions if you prefer.
async def router(state: AgentState) -> AgentState:
    # 1. Extract the history from the state
    ctx = state.get("user_context", {}) or {}
    chat_history = ctx.get("chat_history", [])
    
    # 2. Format the history
    formatted_history = "No previous history."
    if chat_history:
        formatted_history = "\n".join(
            [f"{'User' if msg['role'] == 'user' else 'System'}: {msg['content']}" 
             for msg in chat_history[-4:]]
        )

    # 3. Ask the LLM which worker bots are required, including context
    prompt = (
        "You are a router determining which microservice workers should handle a user query. "
        "Return a JSON array of service names from [\"sql-bot\", \"graph-bot\", \"search-bot\", "
        "\"doc-bot\", \"embedding-bot\"] that are needed.\n\n" 
        f"CONVERSATION HISTORY (Context for the latest query):\n{formatted_history}\n\n"
        f"LATEST USER QUERY:\n{state['query']}"
    )
    
    # helper will perform the LLM call and is itself traceable
    try:
        if tracer is not None:
            with tracer.as_default():
                raw = await call_router_llm(prompt)
        else:
            raw = await call_router_llm(prompt)
        text = raw or ""

        # strip markdown code fences, backticks, and any leading "json" label
        def _clean(t: str) -> str:
            t = t.strip()
            if t.startswith("```") and t.endswith("```"):
                t = t[3:-3].strip()
            t = t.strip("`")
            if t.lower().startswith("json"):
                t = t[len("json"):].strip()
            return t

        clean_text = _clean(text)
        try:
            names = json.loads(clean_text)
        except Exception:
            print(f"DEBUG: Failed to parse JSON: {text}")
            names = []
        state["required_workers"] = names
    except Exception:
        # if the LLM call fails (rate limit, 503, etc.) just continue with no workers
        state["required_workers"] = []
    return state

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
        #"decision-bot",
        #"calc-bot",
        "embedding-bot"
    }

    # 2. Sanitize and filter the LLM's output
    sanitized_workers = set()
    for w in raw_workers:
        # Replace accidental underscores with dashes
        clean_name = str(w).replace("_", "-").strip().lower()
        if clean_name in VALID_WORKERS:
            sanitized_workers.add(clean_name)

        run_tree = get_current_run_tree()
        ls_headers = run_tree.to_headers() if run_tree else {}

    async with httpx.AsyncClient(timeout=3600.0) as hc:
        tasks = []
        # 3. Use the sanitized list to build the URLs
        for w in sanitized_workers:
            url = f"http://{w}:8000/query"
            req_obj = WorkerRequest(query=query, context=ctx)
            tasks.append(hc.post(
                url, 
                json=req_obj.model_dump(), 
                headers=ls_headers
            ))

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

    # Return ONLY the dictionary of what you want to append
    return {"worker_responses": results}

# --- Synthesis node --------------------------------------------------------


@traceable(run_type="llm", name="Final-Synthesizer")
async def synthesize(state: AgentState) -> AgentState:
    responses = state.get("worker_responses", []) or []
    if not responses:
        state["final_summary"] = "Няма намерена информация."
        return state

    # Предаваме директно данните към Gemini за кратък превод и синтез
    prompt = (
        "По-долу са резултати от търсене в банкова система (някои може да са на английски). "
        "Предоставете кратко обобщение на намерената информация на БЪЛГАРСКИ ЕЗИК. "
        "Ако има грешка в някой бот, просто я преведете накратко. "
        "Бъдете максимално кратки и не пишете уводи или доклади. "
        "Формат: Списък или кратки абзаци.\n\n"
        f"Данни: {json.dumps(responses, ensure_ascii=False)}"
    )
    
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt) ##TO-DO Change to pro model
        state["final_summary"] = resp.text.strip()
    except Exception as e:
        # Резервен вариант при проблем с LLM - връщаме суровите съобщения
        summary_parts = []
        for r in responses:
            msg = r.get("message") or r.get("detail") or str(r)
            summary_parts.append(str(msg))
        state["final_summary"] = "\n".join(summary_parts)
        
    return state

# --- Assemble LangGraph ----------------------------------------------------
# StateGraph constructor changed: it now requires a state_schema argument (the type
# of the state object). The previous `initial_state_type` parameter has been removed.
# We just pass our Pydantic model class directly.
graph = StateGraph(state_schema=AgentState)

# add nodes and then connect them with edges
graph.add_node("router", router)
graph.add_node("execute", execute_workers)
graph.add_node("synthesize", synthesize)

# connect the nodes sequentially
graph.add_edge(START, "router")
graph.add_edge("router", "execute")
graph.add_edge("execute", "synthesize")

# compile the graph once; compiled_graph.invoke(...) will execute the workflow
compiled_graph = graph.compile()


# --- FastAPI endpoint ------------------------------------------------------
@app.post("/chat")
async def chat(req: WorkerRequest) -> Dict[str, Any]:
    # seed the state
    state: AgentState = AgentState(
        query=req.query,
        user_context=req.context or {},
        required_workers=[],
    )

    # use the async API since our nodes (router/execute/synthesize) are async
    # compiled_graph.ainvoke returns a dict representing the final state.
    if tracer is not None:
        # the tracer will gather events during graph execution
        with tracer.as_default():
            finished = await compiled_graph.ainvoke(input=state)
    else:
        finished = await compiled_graph.ainvoke(input=state)
    # if you prefer a Pydantic object you could reconstruct AgentState(**finished)
    return {
        "final_summary": finished.get("final_summary"),
        "worker_responses": finished.get("worker_responses"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
