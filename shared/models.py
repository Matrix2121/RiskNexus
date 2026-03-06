from __future__ import annotations

from typing import Dict, List, TypedDict, Optional

from pydantic import BaseModel


# WorkerRequest model for internal communication between orchestrator and bots
class WorkerRequest(BaseModel):
    query: str
    context: Dict[str, Optional[str]] = {}


# WorkerResponse model returned by worker bots
class WorkerResponse(BaseModel):
    status: str  # expected values: "SUCCESS" or "ERROR"
    data: Dict[str, Optional[object]] = {}
    citations: List[str] = []
    message: str = ""


# LangGraph agent state definition used by orchestrator
class AgentState(TypedDict, total=False):
    query: str
    user_context: Dict[str, object]
    required_workers: List[str]  # e.g., ["sql_bot", "search_bot"]
    sql_data: Optional[Dict[str, object]]
    graph_data: Optional[Dict[str, object]]
    search_data: Optional[Dict[str, object]]
    doc_data: Optional[Dict[str, object]]
    calc_results: Optional[Dict[str, object]]
    decision_profile: Optional[str]
    final_summary: Optional[str]
    errors: List[str]
