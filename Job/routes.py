from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from pydantic import BaseModel

from db.database import get_db
from Job.schemas import JobCreate, JobDetailResponse, JobListingResponse
from Job.service import (
    create_job,
    get_all_jobs,
    get_all_jobs_by_hr,
    get_job_details,
    insert_questions,
    fetch_questions,
)

job_router = APIRouter()


class JobListResponse(BaseModel):
    jobs: List[JobListingResponse]


@job_router.post("/create_job", response_model=JobDetailResponse)
async def create_new_job(job_data: JobCreate, db: AsyncSession = Depends(get_db)):
    """Create a job and its associated questions in one call."""
    job = await create_job(db, job_data)
    await insert_questions(job.id, job_data.questions, db)
    questions = await fetch_questions(job.id, db)

    return JobDetailResponse(
        id=job.id,
        title=job.title,
        hrId=job.hrId,
        description=job.description,
        salary=job.salary,
        company=job.company,
        job_type=job.job_type,
        skills=job.skills,
        requirements=job.requirements,
        questions=questions,
    )


@job_router.get("/get_jobs", response_model=JobListResponse)
async def get_jobs(db: AsyncSession = Depends(get_db)):
    """List all available jobs."""
    jobs = await get_all_jobs(db)
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found")
    return {"jobs": jobs}


@job_router.get("/get_jobs_HRId", response_model=JobListResponse)
async def get_jobs_by_hr(HRId: int, db: AsyncSession = Depends(get_db)):
    """List all jobs posted by a specific HR user."""
    jobs = await get_all_jobs_by_hr(HRId, db)
    if not jobs:
        raise HTTPException(status_code=404, detail="No jobs found")
    return {"jobs": jobs}


@job_router.get("/get_job_info", response_model=JobDetailResponse)
async def get_job_info(job_id: int, db: AsyncSession = Depends(get_db)):
    """Get full details for a single job, including its questions."""
    return await get_job_details(db, job_id)