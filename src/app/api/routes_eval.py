"""Evaluation endpoints: POST /evaluate, GET /evaluate/stream/{job_id}, GET /evaluate/last."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from app.eval.runner import JobState, get_job, get_last, is_running, register_job, run_eval_bg
from app.rag.llm import LLMUnavailable
from app.schemas import EvalRequest, EvalResponse, EvalStartResponse

router = APIRouter(tags=["eval"])


@router.post("/evaluate", response_model=EvalStartResponse, status_code=202)
async def evaluate(request: EvalRequest) -> EvalStartResponse | JSONResponse:
    if is_running():
        return JSONResponse(status_code=409, content={"detail": "An evaluation is already running."})
    job_id = register_job()
    asyncio.create_task(run_eval_bg(job_id, request))
    return EvalStartResponse(job_id=job_id)


async def _sse_generator(job: JobState) -> AsyncGenerator[str, None]:
    sent = 0
    while True:
        while sent < len(job.results):
            r = job.results[sent]
            data = json.dumps({
                "event": "progress",
                "n_done": sent + 1,
                "n_total": job.n_total,
                "result": r.model_dump(),
            })
            yield f"data: {data}\n\n"
            sent += 1

        if job.done:
            if job.response is not None:
                data = json.dumps({
                    "event": "done",
                    "summary": job.response.summary.model_dump(),
                    "config": job.response.config,
                    "timestamp": job.response.timestamp,
                })
                yield f"data: {data}\n\n"
            break

        # Clear before re-checking to avoid missing a set() that races with us
        job.event.clear()
        if sent < len(job.results) or job.done:
            continue
        await job.event.wait()


@router.get("/evaluate/stream/{job_id}", response_model=None)
async def evaluate_stream(job_id: str) -> StreamingResponse | JSONResponse:
    job = get_job(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "Job not found."})
    return StreamingResponse(_sse_generator(job), media_type="text/event-stream")


@router.get("/evaluate/last", response_model=EvalResponse)
async def evaluate_last() -> EvalResponse | JSONResponse:
    result = get_last()
    if result is None:
        return JSONResponse(status_code=404, content={"detail": "No evaluation has been run yet."})
    return result
