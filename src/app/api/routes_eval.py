"""Evaluation endpoints: /evaluate and /evaluate/last."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.eval.runner import get_last, run_eval
from app.rag.llm import LLMUnavailable
from app.schemas import EvalRequest, EvalResponse

router = APIRouter(tags=["eval"])


@router.post("/evaluate", response_model=EvalResponse)
async def evaluate(request: EvalRequest) -> EvalResponse | JSONResponse:
    try:
        return await run_eval(request)
    except LLMUnavailable as exc:
        return JSONResponse(status_code=503, content={"detail": str(exc)})


@router.get("/evaluate/last", response_model=EvalResponse)
async def evaluate_last() -> EvalResponse | JSONResponse:
    result = get_last()
    if result is None:
        return JSONResponse(status_code=404, content={"detail": "No evaluation has been run yet."})
    return result
