"""FastAPI REST API server for nanocortex.

Provides HTTP endpoints for document ingestion, querying, approval workflows,
feedback, and audit trail access.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nanocortex.api.orchestrator import NanoCortex
from nanocortex.models.domain import PolicyRule, PolicyVerdict

logger = logging.getLogger(__name__)

# ── Pydantic Request/Response Models ──────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    top_k: int = Field(5, description="Number of evidence chunks to retrieve")
    strategy: str = Field("hybrid", description="Retrieval strategy: bm25, vector, or hybrid")
    context: dict[str, str] | None = Field(None, description="Optional context for policy evaluation")


class ApprovalRequest(BaseModel):
    reason: str = Field("", description="Reason for approval/rejection")


class FeedbackRequest(BaseModel):
    rating: str = Field(..., description="Rating: correct, partially_correct, incorrect, hallucination")
    corrected_answer: str = Field("", description="Optional corrected answer")
    explanation: str = Field("", description="Optional explanation")


class PolicyRuleRequest(BaseModel):
    name: str = Field(..., description="Rule name")
    description: str = Field("", description="Rule description")
    condition: str = Field(..., description="Condition: no_evidence, contains:<pattern>, min_score:<threshold>")
    verdict: str = Field("needs_approval", description="Verdict: allow, deny, needs_approval")


class HealthResponse(BaseModel):
    status: str
    version: str
    chunks_indexed: int
    policy_rules: int


# ── FastAPI App ───────────────────────────────────────────────────

app = FastAPI(
    title="nanocortex",
    description="Unified AI System REST API for auditable multimodal AI decision-making",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global NanoCortex instance
_cortex: NanoCortex | None = None


def get_cortex() -> NanoCortex:
    global _cortex
    if _cortex is None:
        _cortex = NanoCortex()
    return _cortex


# ── Health Check ──────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    cortex = get_cortex()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        chunks_indexed=cortex.knowledge.chunk_count,
        policy_rules=len(cortex.policy_engine.rules),
    )


# ── Document Ingestion ────────────────────────────────────────────

@app.post("/v1/ingest", tags=["Ingestion"])
async def ingest_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Ingest a document (PDF or image) and index it for retrieval."""
    cortex = get_cortex()
    
    # Save uploaded file temporarily
    temp_path = Path(f"./data/uploads/{file.filename}")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        content = await file.read()
        temp_path.write_bytes(content)
        
        result = await cortex.ingest(temp_path)
        return result
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/v1/ingest/path", tags=["Ingestion"])
async def ingest_document_from_path(file_path: str) -> dict[str, Any]:
    """Ingest a document from a local file path."""
    cortex = get_cortex()
    path = Path(file_path)
    
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    
    return await cortex.ingest(path)


# ── Query & Retrieval ─────────────────────────────────────────────

@app.post("/v1/query", tags=["Query"])
async def query(request: QueryRequest) -> dict[str, Any]:
    """Run a query through retrieval → reasoning → decision pipeline."""
    cortex = get_cortex()
    
    result = await cortex.query(
        question=request.question,
        top_k=request.top_k,
        strategy=request.strategy,
        context=request.context,
    )
    return result


# ── Human-in-the-Loop Approval ────────────────────────────────────

@app.post("/v1/decisions/{decision_id}/approve", tags=["Approval"])
async def approve_decision(decision_id: str) -> dict[str, Any]:
    """Approve a pending decision."""
    cortex = get_cortex()
    result = cortex.approve_decision(decision_id)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


@app.post("/v1/decisions/{decision_id}/reject", tags=["Approval"])
async def reject_decision(decision_id: str, request: ApprovalRequest) -> dict[str, Any]:
    """Reject a pending decision with optional reason."""
    cortex = get_cortex()
    result = cortex.reject_decision(decision_id, request.reason)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


# ── Learning Feedback ─────────────────────────────────────────────

@app.post("/v1/feedback/{decision_id}", tags=["Learning"])
async def submit_feedback(decision_id: str, request: FeedbackRequest) -> dict[str, Any]:
    """Submit feedback on a decision for the learning loop."""
    cortex = get_cortex()
    
    valid_ratings = {"correct", "partially_correct", "incorrect", "hallucination"}
    if request.rating not in valid_ratings:
        raise HTTPException(status_code=400, detail=f"Invalid rating. Must be one of: {valid_ratings}")
    
    return cortex.submit_feedback(
        decision_id=decision_id,
        rating=request.rating,
        corrected_answer=request.corrected_answer,
        explanation=request.explanation,
    )


@app.get("/v1/learning/stats", tags=["Learning"])
async def get_learning_stats() -> dict[str, Any]:
    """Get current learning loop statistics."""
    cortex = get_cortex()
    return cortex.get_learning_stats()


# ── Policy Management ─────────────────────────────────────────────

@app.get("/v1/policies", tags=["Policy"])
async def list_policies() -> list[dict[str, Any]]:
    """List all active policy rules."""
    cortex = get_cortex()
    return [
        {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "description": rule.description,
            "condition": rule.condition,
            "verdict": rule.verdict.value,
        }
        for rule in cortex.policy_engine.rules
    ]


@app.post("/v1/policies", tags=["Policy"])
async def add_policy(request: PolicyRuleRequest) -> dict[str, Any]:
    """Add a new policy rule."""
    cortex = get_cortex()
    
    verdict_map = {
        "allow": PolicyVerdict.ALLOW,
        "deny": PolicyVerdict.DENY,
        "needs_approval": PolicyVerdict.NEEDS_APPROVAL,
    }
    
    if request.verdict not in verdict_map:
        raise HTTPException(status_code=400, detail=f"Invalid verdict. Must be one of: {list(verdict_map.keys())}")
    
    rule = PolicyRule(
        name=request.name,
        description=request.description,
        condition=request.condition,
        verdict=verdict_map[request.verdict],
    )
    cortex.policy_engine.add_rule(rule)
    
    return {
        "rule_id": rule.rule_id,
        "name": rule.name,
        "verdict": rule.verdict.value,
        "message": "Policy rule added successfully",
    }


# ── Audit Trail ───────────────────────────────────────────────────

@app.get("/v1/audit", tags=["Audit"])
async def get_audit_trail(decision_id: str | None = None) -> list[dict[str, Any]]:
    """Get audit events, optionally filtered by decision ID."""
    cortex = get_cortex()
    return cortex.get_audit_trail(decision_id)


@app.get("/v1/audit/{decision_id}", tags=["Audit"])
async def get_decision_trace(decision_id: str) -> list[dict[str, Any]]:
    """Get the complete audit trace for a specific decision."""
    cortex = get_cortex()
    return cortex.get_audit_trail(decision_id)


# ── Server Entry Point ────────────────────────────────────────────

def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
