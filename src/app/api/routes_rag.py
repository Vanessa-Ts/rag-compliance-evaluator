"""RAG endpoints: /ingest and /query."""

from __future__ import annotations

import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.rag.ingest import ingest
from app.rag.llm import LLMUnavailable
from app.rag.pipeline import answer_query, stream_query
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


@router.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    async def event_generator():
        try:
            async for item in stream_query(request):
                yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"
        except LLMUnavailable as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
