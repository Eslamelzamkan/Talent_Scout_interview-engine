from fastapi import APIRouter, BackgroundTasks, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any

from db.database import get_db
from Hr.schemas import HrCreate, GetHr, GetUserScores, ComputeUserScores
from Hr.service import HrService

hr_router = APIRouter(
    prefix="/hr",
    tags=["HR"],
)

hr_service = HrService()


@hr_router.post("/create", response_model=Dict[str, Any])
async def create_hr(request: HrCreate, db: AsyncSession = Depends(get_db)):
    """Create a new HR user."""
    return await hr_service.create_hr(request, db)


@hr_router.post("/login", response_model=Dict[str, Any])
async def login_hr(request: GetHr, db: AsyncSession = Depends(get_db)):
    """Login for HR users."""
    return await hr_service.hr_login(request, db)


@hr_router.post("/get_user_scores", response_model=Dict[str, Any])
async def get_scores(request: GetUserScores, db: AsyncSession = Depends(get_db)):
    """Get scores for a user."""
    return await hr_service.get_user_scores(request, db)


@hr_router.post(
    "/compute_scores",
    response_model=Dict[str, Any],
    status_code=status.HTTP_202_ACCEPTED,
)
async def compute_scores(
    request: ComputeUserScores,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Kick off AI score computation for a candidate.

    Returns **202 Accepted** immediately — the pipeline runs in the background.
    Poll `GET /hr/task_status/{user_id}/{job_id}` to track progress.
    """
    return await hr_service.compute_scores(request, db, background_tasks)


@hr_router.get(
    "/task_status/{user_id}/{job_id}",
    response_model=Dict[str, Any],
)
async def task_status(
    user_id: int,
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Poll the processing status of a compute_scores job.

    Possible status values:
    - **queued** — job accepted, waiting for the GPU lock
    - **processing** — inference is actively running
    - **done** — all scores have been saved to the database
    - **failed** — pipeline encountered an unrecoverable error
    - **not_found** — no record for this user/job combination
    """
    return await hr_service.get_task_status(user_id, job_id, db)