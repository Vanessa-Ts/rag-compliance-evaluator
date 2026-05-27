import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes_eval import router as eval_router
from app.api.routes_rag import router as rag_router
from app.core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info(
        "Starting %s v%s [%s]",
        settings.app_name,
        settings.version,
        settings.environment,
    )
    # Warm embeddings model and auto-ingest corpus if the collection is empty.
    try:
        from app.rag.embeddings import get_embeddings
        from app.rag.ingest import ingest
        from app.rag.store import collection_count

        get_embeddings()  # warm the model so first query is fast
        if collection_count() == 0:
            logger.info("Collection empty — running initial corpus ingest...")
            result = await ingest()
            logger.info(
                "Ingest complete: %d docs, %d chunks (%.0f ms)",
                result.documents,
                result.chunks,
                result.duration_ms,
            )
    except Exception as exc:
        logger.warning("Startup ingest failed (non-fatal): %s", exc)

    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
)

app.include_router(rag_router)
app.include_router(eval_router)

app.mount("/static", StaticFiles(directory="src/app/ui"), name="static")
templates = Jinja2Templates(directory="src/app/ui")


# --Health--
@app.get("/health/live")
async def liveness() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/health/ready")
async def readiness() -> JSONResponse:
    """Ready only when the vector collection is non-empty (corpus ingested)."""
    try:
        from app.rag.store import collection_count

        count = collection_count()
        if count == 0:
            return JSONResponse(status_code=503, content={"status": "not ready", "detail": "corpus not ingested"})
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "not ready", "detail": str(exc)})
    return JSONResponse({"status": "ok", "vectors": count})


# --Routes--
@app.get("/")
async def index(request: Request) -> Response:
    return templates.TemplateResponse("index.html", {
        "request": request,
        "app_name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=settings.environment == "development")  # noqa: S104
