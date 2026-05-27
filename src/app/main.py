import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.config import settings

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle hook."""
    logger.info(
        "Starting %s v%s [%s]",
        settings.app_name,
        settings.version,
        settings.environment,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)




app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url=None,
)
 
app.mount("/static", StaticFiles(directory="src/app/ui"), name="static")
templates = Jinja2Templates(directory="src/app/ui")


# --Health--
@app.get("/health/live")
async def liveness() -> JSONResponse:
    """Liveness probe — process is running."""
    return JSONResponse({"status": "ok"})
 
 
@app.get("/health/ready")
async def readiness() -> JSONResponse:
    """Readiness probe — add dependency checks here."""
    return JSONResponse({"status": "ok"})


# --Routes--
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "app_name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=settings.environment == "development",) # noqa: S104
