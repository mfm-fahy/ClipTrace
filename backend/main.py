from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import concurrent.futures
import traceback

from app.db.database import init_db
from app.api import videos, match, verify, monetization

# Thread pool for CPU-heavy video processing (segmentation + ML inference)
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    loop = asyncio.get_event_loop()
    loop.set_default_executor(_executor)
    yield
    _executor.shutdown(wait=False)


app = FastAPI(
    title="ClipTrace API",
    description="Intelligent Video Identity and Tracking System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos.router,       prefix="/api/videos",       tags=["Videos"])
app.include_router(match.router,        prefix="/api/match",         tags=["Matching"])
app.include_router(verify.router,       prefix="/api/verify",        tags=["Verification"])
app.include_router(monetization.router, prefix="/api/monetization",  tags=["Monetization"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    return JSONResponse(status_code=500, content={"detail": str(exc), "traceback": tb})


@app.get("/")
async def root():
    return {"service": "ClipTrace", "status": "running", "version": "1.0.0"}
