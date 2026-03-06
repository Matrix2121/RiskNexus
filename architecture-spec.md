Technical Specification: RiskNexus Agentic Architecture (Local PoC)

1. System Overview
   RiskNexus is a distributed, multi-agent AI system for the banking and risk sector. It is driven by a central LangGraph orchestrator that manages conversational state and execution logic. The orchestrator delegates specialized tasks via HTTP/REST to isolated FastAPI microservices (Worker Bots).

2. Technology Stack & LLM Mapping
   Orchestration & State: LangGraph

API Framework: FastAPI (Python 3.11+)

Data Validation: Pydantic V2

Vector Database: ChromaDB (Running locally via Docker)

LLM Provider: Google Gemini API

2.1 Model Assignments
gemini-2.5-pro (High Reasoning): Assigned to the Orchestrator Bot (for complex intent classification and final synthesis) and the Decision Making Bot (for complex financial risk profiling).

gemini-2.5-flash (Speed & Context): Assigned to the SQL Bot, Graph Bot, Search Bot, and Documentation Bot (for fast schema reading, Cypher/SQL generation, and RAG synthesis).

gemini-2.5-flash-lite (Efficiency): Assigned to the Embedding Bot (if using an LLM to generate document metadata before vectorization) and can be used for simple, high-volume classification tasks.

3. LangGraph State Definition
   The orchestrator maintains a strict state dictionary passed between nodes.

Python
class AgentState(TypedDict):
query: str
user_context: dict
required_workers: list[str] # e.g., ["sql_bot", "search_bot"]
sql_data: dict | None
graph_data: dict | None
search_data: dict | None
doc_data: dict | None
calc_results: dict | None
decision_profile: str | None
final_summary: str | None
errors: list[str] 4. API Contracts & Timeouts
Standardized payloads for orchestrator-to-worker communication.

4.1 Definitive Timeout Mechanism
Because LLMs can take time to generate complex SQL/Cypher or synthesize large documents, the Orchestrator's internal HTTP client (e.g., httpx.AsyncClient) communicating with the worker bots must be explicitly configured to prevent timeouts.

HTTP Client Config: timeout=3600.0 (1 hour) will be globally applied to all worker bot requests during this PoC to guarantee no connection drops.

4.2 WorkerRequest
JSON
{
"query": "string (The original user query or a sub-query)",
"context": "dict (Optional metadata, e.g., entity names, dates)"
}
4.3 WorkerResponse
JSON
{
"status": "enum [SUCCESS, ERROR]",
"data": "dict (The raw factual payload retrieved or calculated)",
"citations": "list of strings (Source tables, graph nodes, URLs, or doc pages)",
"message": "string (Human-readable summary of the worker's action)"
} 5. Microservice Definitions (The Bots)
Orchestrator Bot (gemini-2.5-pro): Analyzes the query, executes parallel REST requests (with timeout=3600.0) to required workers, waits for responses, and synthesizes the aggregated data into a final response.

SQL Bot (gemini-2.5-flash): Queries ChromaDB (sql_schemas) -> Passes DDLs + question to LLM -> Executes generated SQL -> Returns raw data rows.

Graph Bot (gemini-2.5-flash): Queries ChromaDB (graph_ontology) -> Passes schema + question to LLM -> Executes Cypher -> Returns graph data.

Search Bot (gemini-2.5-flash): Scrapes live web -> Returns summarized external context regarding entities.

Documentation Bot (gemini-2.5-flash): Queries ChromaDB (documents) -> Passes chunks + question to LLM -> Returns grounded answers with metadata citations.

Decision Making Bot (gemini-2.5-pro): Evaluates aggregated data against banking compliance rules to generate a formal risk profile.

Calculation Bot (Deterministic Python): Executes strict mathematical models. No LLM is used here.

Embedding Bot (Dual-Model Ingestion Pipeline): Populates ChromaDB. Not directly queried during the user chat flow. Operates using a strict two-step pipeline to ensure high-quality retrieval:

Step 1: Metadata Generation (gemini-2.5-flash-lite): Takes the raw chunk of text, SQL DDL, or Graph ontology and generates a JSON object containing a brief summary, key entities, and relevant tags.

Step 2: Vectorization (gemini-embedding-001): Takes the raw text (and optionally the generated summary) and converts it into a vector embedding.

Storage: Saves the vector and the generated metadata JSON into the respective ChromaDB collection.

Endpoints: Exposes /embed/documents, /embed/sql-schemas, and /embed/graph-schemas.

6. Persistence & Infrastructure
   To ensure the vector database does not wipe its data when containers restart, ChromaDB requires a definitive local volume mount.

Security: For this PoC, all internal FastAPI endpoints will remain unauthenticated. Internal port exposure will be restricted to the Docker network.

ChromaDB Docker Configuration:

YAML
chromadb:
image: chromadb/chroma:latest
volumes: - ./chroma_data:/chroma/chroma
ports: - "8000:8000"
ChromaDB Collections: docs_regulations, docs_contracts, schema_sql_core, schema_graph_entities.
