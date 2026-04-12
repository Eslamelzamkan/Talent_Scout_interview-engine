"""
Job/service.py
Renamed from: services.py  (to match the Hr and User module convention)

All job-related business logic: create, list, and retrieve jobs and questions.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from fastapi import HTTPException
from typing import List

from db.Models.JobModels import Job, JobQuestion
from Hr.defaults import resolve_hr_id
from Job.schemas import (
    JobCreate,
    JobDetailResponse,
    JobListingResponse,
    JobQuestionResponse,
    JobQuestionCreate,
)


async def create_job(db: AsyncSession, job_data: JobCreate) -> JobDetailResponse:
    """Insert a new Job row and return its details."""
    resolved_hr_id = await resolve_hr_id(db, explicit_hr_id=job_data.HRId)

    new_job = Job(
        title=job_data.title,
        description=job_data.description,
        HRId=resolved_hr_id,
        salary=job_data.salary,
        company=job_data.company,
        job_type=job_data.job_type,
        skills=job_data.skills,
        requirements=job_data.requirements,
    )
    db.add(new_job)
    await db.commit()
    await db.refresh(new_job)

    return JobDetailResponse(
        id=new_job.id,
        title=new_job.title,
        description=new_job.description,
        salary=new_job.salary,
        hrId=resolved_hr_id,
        company=new_job.company,
        job_type=new_job.job_type,
        skills=new_job.skills,
        requirements=new_job.requirements,
        questions=[],
    )


async def insert_questions(
    job_id: int, questions: List[JobQuestionCreate], db: AsyncSession
) -> None:
    """Bulk-insert job questions."""
    if questions:
        db.add_all([JobQuestion(job_id=job_id, question=q.question) for q in questions])
        await db.commit()


async def fetch_questions(job_id: int, db: AsyncSession) -> List[JobQuestionResponse]:
    """Fetch all questions for a given job."""
    result = await db.execute(
        select(JobQuestion).where(JobQuestion.job_id == job_id)
    )
    return [
        JobQuestionResponse(id=q.id, job_id=q.job_id, question=q.question)
        for q in result.scalars().all()
    ]


async def get_all_jobs(db: AsyncSession) -> List[JobListingResponse]:
    """Return all jobs (summary fields only)."""
    result = await db.execute(select(Job))
    return [
        JobListingResponse(
            id=job.id,
            title=job.title,
            company=job.company,
            salary=job.salary,
            job_type=job.job_type,
            description=job.description,
        )
        for job in result.scalars().all()
    ]


async def get_all_jobs_by_hr(hr_id: int, db: AsyncSession) -> List[JobListingResponse]:
    """Return all jobs posted by a specific HR user."""
    result = await db.execute(select(Job).where(Job.HRId == hr_id))
    return [
        JobListingResponse(
            id=job.id,
            title=job.title,
            company=job.company,
            salary=job.salary,
            job_type=job.job_type,
            description=job.description,
        )
        for job in result.scalars().all()
    ]


async def get_job_details(db: AsyncSession, job_id: int) -> JobDetailResponse:
    """Return full job details including all questions."""
    result = await db.execute(
        select(Job).options(selectinload(Job.questions)).filter(Job.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobDetailResponse(
        id=job.id,
        title=job.title,
        company=job.company,
        salary=job.salary,
        hrId=job.HRId,
        job_type=job.job_type,
        description=job.description,
        skills=job.skills,
        requirements=job.requirements,
        questions=[
            JobQuestionResponse(id=q.id, job_id=q.job_id, question=q.question)
            for q in job.questions
        ],
    )
