from __future__ import annotations

import operator
from typing import Any, Dict, List, TypedDict, Optional, Annotated

from pydantic import BaseModel


# WorkerRequest model for internal communication between orchestrator and bots
class WorkerRequest(BaseModel):
    query: str
    context: Dict[str, Any] = {}


# WorkerResponse model returned by worker bots
class WorkerResponse(BaseModel):
    status: str  # expected values: "SUCCESS" or "ERROR"
    data: Dict[str, Optional[Any]] = {}
    citations: List[str] = []
    message: str = ""


# LangGraph agent state definition used by orchestrator
class AgentState(TypedDict, total=False):
    query: str
    user_context: Dict[str, Any]
    required_workers: List[str]  # e.g., ["sql-bot", "search-bot"]
    
    # --- THE FIX: Use Annotated and operator.add for lists ---
    # This tells LangGraph to APPEND new items to these lists, not overwrite them.
    worker_responses: Annotated[List[Dict[str, Any]], operator.add]
    errors: Annotated[List[str], operator.add]
    
    # Optional fields
    sql_data: Optional[Dict[str, Any]]
    graph_data: Optional[Dict[str, Any]]
    search_data: Optional[Dict[str, Any]]
    doc_data: Optional[Dict[str, Any]]
    #calc_results: Optional[Dict[str, Any]]
    #decision_profile: Optional[str]
    final_summary: Optional[str]