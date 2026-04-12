from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.Models.HrModel import HR
from db.Models.JobModels import Job

SYSTEM_HR_NAME = "System HR"
SYSTEM_HR_EMAIL = "system@interview-engine.local"
SYSTEM_HR_PASSWORD = "system"


async def ensure_default_hr_id(db: AsyncSession) -> int:
    result = await db.execute(select(HR).where(HR.email == SYSTEM_HR_EMAIL))
    hr = result.scalar_one_or_none()

    if hr is None:
        hr = HR(
            name=SYSTEM_HR_NAME,
            email=SYSTEM_HR_EMAIL,
            password=SYSTEM_HR_PASSWORD,
        )
        db.add(hr)
        await db.commit()
        await db.refresh(hr)

    return hr.id


async def resolve_hr_id(
    db: AsyncSession,
    explicit_hr_id: int | None = None,
    job_id: int | None = None,
) -> int:
    if explicit_hr_id is not None:
        result = await db.execute(select(HR).where(HR.id == explicit_hr_id))
        hr = result.scalar_one_or_none()
        if hr is not None:
            return hr.id

    if job_id is not None:
        result = await db.execute(select(Job.HRId).where(Job.id == job_id))
        job_hr_id = result.scalar_one_or_none()
        if job_hr_id is not None:
            return job_hr_id

    return await ensure_default_hr_id(db)
