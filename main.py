from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

from Ai.model_registry import model_registry
from Hr.routes import hr_router
from Job.routes import job_router
from User.routes import user_router
from db.database import Base, engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text(
                "ALTER TABLE video_processing "
                "ADD COLUMN IF NOT EXISTS queued_at TIMESTAMPTZ"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE video_processing "
                "ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE video_processing "
                "ADD COLUMN IF NOT EXISTS completed_at TIMESTAMPTZ"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE video_processing "
                "ADD COLUMN IF NOT EXISTS result_quality VARCHAR DEFAULT 'complete'"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE video_processing "
                "ADD COLUMN IF NOT EXISTS quality_warnings TEXT"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE video_processing "
                "ADD COLUMN IF NOT EXISTS question_quality TEXT"
            )
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_tables()
    yield
    model_registry.unload_all()


app = FastAPI(
    title="Interview Engine API",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "message": "Interview Engine API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.include_router(hr_router)
app.include_router(user_router, prefix="/user", tags=["User"])
app.include_router(job_router, prefix="/job", tags=["Job"])
