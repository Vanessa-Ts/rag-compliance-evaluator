"""RAG endpoints: /ingest and /query."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.rag.ingest import ingest
from app.rag.llm import LLMUnavailable
from app.rag.pipeline import answer_query
from app.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse

router = APIRouter(tags=["rag"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_corpus(request: IngestRequest) -> IngestResponse:
    return await ingest(force=request.force)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse | JSONResponse:
    try:
        return await answer_query(request)
    except LLMUnavailable as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})
