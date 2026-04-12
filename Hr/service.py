"""
Hr/service.py
─────────────────────────────────────────────────────────────────────────────
Key design decisions:

  1. LLM errors → handled in Ai/Text_Model/Gemini.py with provider-specific
     fallbacks for summary + relevance.

  2. Singleton models → NOT instantiated here.  They are loaded once at startup
     via Ai.model_registry and injected through the module-level singleton.

  3. Background processing → compute_scores() returns 202 Accepted immediately
     and offloads all heavy work to _compute_scores_task() (a BackgroundTask).
     Poll GET /hr/task_status/{user_id}/{job_id} to check progress.

  4. GPU concurrency guard → threading.Lock() (_gpu_lock) ensures only ONE
     compute job can touch the GPU at a time, preventing CUDA OOM crashes when
     multiple HR users click "Compute" simultaneously.

Architecture:
  ┌─────────────────────────────────────────────────────────┐
  │  HTTP request (async)                                   │
  │    compute_scores() ──► creates "queued" DB row         │
  │                    ──► adds _compute_scores_task        │
  │                    ──► returns 202 immediately           │
  └─────────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────────┐
  │  BackgroundTask (async, after response is sent)         │
  │    _compute_scores_task()                               │
  │      • async ops: cheating detection, transcription     │
  │      • await run_in_executor(_sync_inference)           │
  │          └─ _sync_inference() holds _gpu_lock          │
  │               • video traits / emotion                  │
  │               • text personality                        │
  │               • audio English score                     │
  │               • LLM summarize + relevance               │
  │      • writes results + "done" status to DB             │
  └─────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from db.database import SessionLocal
from db.Models import HrModel, JobModels, UserModel
from Hr.defaults import resolve_hr_id
from Hr.schemas import HrCreate, GetHr, GetUserScores, ComputeUserScores
from Ai.model_registry import model_registry
from Ai.media_utils import MediaUtils as HelperText
from Ai.Text_Model.Gemini import Gemini, SUMMARY_FALLBACK_TEXT

logger = logging.getLogger(__name__)

# ── GPU concurrency guard ─────────────────────────────────────────────────────
# Only one compute job may touch the GPU at a time.
# threading.Lock is used (not asyncio.Lock) because _sync_inference() runs
# inside a ThreadPoolExecutor, not directly in the event loop.
_gpu_lock = threading.Lock()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _loads_json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _unique_messages(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _build_model_quality_warnings() -> list[str]:
    warnings: list[str] = []

    audio_model = model_registry.audio_model
    if audio_model is None:
        warnings.append("English scoring is unavailable and may be using a neutral fallback.")
    elif getattr(audio_model, "is_fallback", False):
        mode = getattr(audio_model, "mode", "heuristic")
        if mode == "hf_pronunciation":
            warnings.append(
                "English scoring uses the Hugging Face pronunciation fallback because the original audio checkpoint is missing."
            )
        else:
            warnings.append(
                "English scoring uses a heuristic fallback because the original audio checkpoint is missing."
            )

    video_model = model_registry.video_traits_model
    if video_model is None or getattr(video_model, "is_fallback", False):
        warnings.append(
            "Video personality traits are running in fallback mode because the video checkpoint is unavailable."
        )

    if model_registry.text_traits_model is None:
        warnings.append(
            "Text personality scoring is unavailable, so trait labels may be degraded."
        )

    return _unique_messages(warnings)


def _finalize_quality(question_quality: list[dict], base_warnings: list[str]) -> tuple[str, list[str]]:
    warnings = list(base_warnings)
    if any(item.get("degraded") for item in question_quality):
        warnings.append("One or more interview answers used fallback LLM output.")
    warnings = _unique_messages(warnings)
    return ("partial" if warnings else "complete", warnings)


def _build_persisted_quality_payload(processing) -> tuple[str, list[str], list[dict]]:
    question_quality = _loads_json_list(getattr(processing, "question_quality", None))
    warnings = _loads_json_list(getattr(processing, "quality_warnings", None))
    result_quality = getattr(processing, "result_quality", None) or ("partial" if warnings else "complete")

    if not question_quality:
        summaries = [
            processing.summarized_text1,
            processing.summarized_text2,
            processing.summarized_text3,
        ]
        question_quality = []
        for index, summary in enumerate(summaries, start=1):
            degraded = bool(summary and summary.strip() == SUMMARY_FALLBACK_TEXT)
            question_quality.append(
                {
                    "question_index": index,
                    "degraded": degraded,
                    "warnings": (
                        ["Summary uses fallback text because the LLM request failed."]
                        if degraded
                        else []
                    ),
                }
            )
        if any(item["degraded"] for item in question_quality):
            warnings = _unique_messages(
                [*warnings, "One or more interview answers used fallback LLM output."]
            )
            result_quality = "partial"

    result_quality, warnings = _finalize_quality(
        question_quality,
        [*warnings, *_build_model_quality_warnings()],
    )

    return result_quality, warnings, question_quality


def _trait_vector(
    values,
    trait_order: list[str],
    default_score: float = 0.5,
) -> list[float]:
    if isinstance(values, dict):
        return [float(values.get(trait, default_score)) for trait in trait_order]

    if isinstance(values, np.ndarray):
        flattened = values.astype(float).flatten().tolist()
        vector = flattened[: len(trait_order)]
        if len(vector) < len(trait_order):
            vector.extend([default_score] * (len(trait_order) - len(vector)))
        return vector

    if isinstance(values, (list, tuple)):
        vector = [float(item) for item in list(values)[: len(trait_order)]]
        if len(vector) < len(trait_order):
            vector.extend([default_score] * (len(trait_order) - len(vector)))
        return vector

    return [default_score] * len(trait_order)


# ─────────────────────────────────────────────────────────────────────────────
# DB helper
# ─────────────────────────────────────────────────────────────────────────────

async def _get_or_create_processing_record(
    db: AsyncSession,
    hr_id: int,
    job_id: int,
    user_id: int,
    status_val: str,
) -> HrModel.VideoProcessing:
    """Fetch an existing VideoProcessing row or create a fresh one, then
    set processing_status to *status_val* and commit."""
    result = await db.execute(
        select(HrModel.VideoProcessing).where(
            HrModel.VideoProcessing.hr_id   == hr_id,
            HrModel.VideoProcessing.job_id  == job_id,
            HrModel.VideoProcessing.user_id == user_id,
        )
    )
    vp = result.scalar_one_or_none()

    if vp is None:
        vp = HrModel.VideoProcessing(
            hr_id=hr_id,
            job_id=job_id,
            user_id=user_id,
            processing_status=status_val,
        )
        db.add(vp)
    else:
        vp.processing_status = status_val

    now = _utc_now()

    if status_val == "queued":
        vp.queued_at = now
        vp.started_at = None
        vp.completed_at = None
        vp.result_quality = None
        vp.quality_warnings = None
        vp.question_quality = None
    elif status_val == "processing":
        vp.queued_at = vp.queued_at or now
        vp.started_at = now
        vp.completed_at = None
        vp.result_quality = None
    elif status_val in {"done", "failed"}:
        vp.queued_at = vp.queued_at or now
        vp.started_at = vp.started_at or vp.queued_at or now
        vp.completed_at = now
        if status_val == "failed":
            vp.result_quality = "failed"
            vp.quality_warnings = json.dumps(
                ["Assessment pipeline failed before a complete report was generated."]
            )

    await db.commit()
    await db.refresh(vp)
    return vp


# ─────────────────────────────────────────────────────────────────────────────
# Background task — runs AFTER the HTTP response has been sent
# ─────────────────────────────────────────────────────────────────────────────

async def _compute_scores_task(user_id: int, job_id: int, hr_id: int):
    """
    Full AI pipeline for one candidate.  Creates its own DB session so it is
    completely independent of the request's session (which is already closed).
    """
    async with SessionLocal() as db:

        # ── Mark as processing ────────────────────────────────────────────
        try:
            vp = await _get_or_create_processing_record(
                db, hr_id, job_id, user_id, "processing"
            )
        except Exception as e:
            logger.error(
                f"[compute_task] Failed to set processing status "
                f"(user={user_id}, job={job_id}): {e}"
            )
            return

        try:
            if not model_registry._loaded:
                logger.info(
                    "[compute_task] Lazy-loading AI models for user=%s job=%s",
                    user_id,
                    job_id,
                )
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, model_registry.load_all)

            # ── Fetch videos & questions ──────────────────────────────────
            videos_q = await db.execute(
                select(UserModel.UserVideo).where(
                    UserModel.UserVideo.userId == user_id
                )
            )
            videos      = videos_q.scalars().all()
            video_paths = [v.videoPath for v in videos]

            questions_q = await db.execute(
                select(JobModels.JobQuestion).where(
                    JobModels.JobQuestion.job_id == job_id
                )
            )
            questions_text = [q.question for q in questions_q.scalars().all()]

            # ── Async operations (cheating detection + transcription) ─────
            # These are awaitable so we run them directly in the event loop.
            audio_paths: list = []
            cheating:    list = []
            texts:       list = []

            for video_path in video_paths:
                # Extract audio (sync file/ffmpeg call — fast)
                try:
                    name = video_path.split("\\")[-1].split(".")[0]
                    audio_paths.append(
                        HelperText.extract_audio(user_id, job_id, name)
                    )
                except Exception as e:
                    logger.error(f"[compute_task] Audio extraction failed for {video_path}: {e}")
                    audio_paths.append(None)

                # Cheating detection — async
                try:
                    score = await model_registry.cheating_model.detect_gaze_cheating_async(
                        video_path
                    )
                    cheating.append(score)
                except Exception as e:
                    logger.error(f"[compute_task] Cheating detection failed for {video_path}: {e}")
                    cheating.append(1.0)  # neutral fallback

            # Transcription — async
            for audio_path in audio_paths:
                try:
                    if audio_path:
                        texts.append(await HelperText.transcribe_audio(audio_path))
                    else:
                        texts.append("")
                except Exception as e:
                    logger.error(f"[compute_task] Transcription failed: {e}")
                    texts.append("")

            # ── GPU inference — sync, in executor, behind the lock ────────
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                _sync_inference,
                video_paths,
                audio_paths,
                texts,
                questions_text,
                cheating,
                user_id,
                job_id,
            )

            # ── Persist results ───────────────────────────────────────────
            # Guard: pad to at least 3 elements so index access never crashes
            def _safe(lst, i, default=None):
                return lst[i] if i < len(lst) else default

            vp.total_score          = round(results["total_score"])
            vp.summarized_text1     = _safe(results["summarizations"], 0, "")
            vp.summarized_text2     = _safe(results["summarizations"], 1, "")
            vp.summarized_text3     = _safe(results["summarizations"], 2, "")
            vp.relevance1           = _safe(results["relevance"], 0, 0)
            vp.relevance2           = _safe(results["relevance"], 1, 0)
            vp.relevance3           = _safe(results["relevance"], 2, 0)
            vp.total_english_score  = round(results["total_english_score"])
            vp.emotion1             = _safe(results["emotions"], 0, "Neutral")
            vp.emotion2             = _safe(results["emotions"], 1, "Neutral")
            vp.emotion3             = _safe(results["emotions"], 2, "Neutral")
            labels = results["trait_labels"]
            vp.trait1 = _safe(labels, 0, "")
            vp.trait2 = _safe(labels, 1, "")
            vp.trait3 = _safe(labels, 2, "")
            vp.trait4 = _safe(labels, 3, "")
            vp.trait5 = _safe(labels, 4, "")
            vp.result_quality      = results["result_quality"]
            vp.quality_warnings    = json.dumps(results["warnings"])
            vp.question_quality    = json.dumps(results["question_quality"])
            vp.processing_status = "done"
            vp.completed_at = _utc_now()
            vp.started_at = vp.started_at or vp.completed_at
            vp.queued_at = vp.queued_at or vp.started_at

            await db.commit()
            logger.info(
                f"[compute_task] Completed → user={user_id}, job={job_id}"
            )

        except Exception as e:
            logger.error(
                f"[compute_task] Pipeline failed (user={user_id}, job={job_id}): {e}",
                exc_info=True,
            )
            try:
                vp.processing_status = "failed"
                vp.result_quality = "failed"
                vp.quality_warnings = json.dumps(
                    ["Assessment pipeline failed before a complete report was generated."]
                )
                vp.question_quality = None
                vp.completed_at = _utc_now()
                vp.started_at = vp.started_at or vp.completed_at
                vp.queued_at = vp.queued_at or vp.started_at
                await db.commit()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Sync inference — runs in ThreadPoolExecutor, holds GPU lock
# ─────────────────────────────────────────────────────────────────────────────

def _sync_inference(
    video_paths:    list,
    audio_paths:    list,
    texts:          list,
    questions:      list,
    cheating:       list,
    user_id:        int,
    job_id:         int,
) -> dict:
    """
    All CPU/GPU model inference happens here.
    The _gpu_lock ensures only ONE thread is in this function at a time,
    preventing CUDA OOM when concurrent requests arrive.

    Returns a dict with all computed scores and labels.
    """
    with _gpu_lock:
        logger.info(f"[inference] GPU lock acquired (user={user_id}, job={job_id})")

        video_traits:   list = []
        emotions:       list = []
        text_traits:    list = []
        english_scores: list = []
        summarizations: list = []
        relevance:      list = []
        question_quality: list = []

        trait_order = ['AGR', 'CONN', 'EXT', 'NEU', 'OPN']
        _default_traits = {k: 0.5 for k in trait_order}

        # ── Video traits + emotion ────────────────────────────────────────
        for video_path in video_paths:
            if model_registry.video_traits_model is None:
                video_traits.append(_default_traits.copy())
            else:
                try:
                    video_traits.append(
                        model_registry.video_traits_model.process_new_video(video_path)
                    )
                except Exception as e:
                    logger.error(f"[inference] video_traits failed for {video_path}: {e}")
                    video_traits.append(_default_traits.copy())

            if model_registry.video_emotion_model is None:
                emotions.append({"Assessment": "Neutral"})
            else:
                try:
                    emotions.append(
                        model_registry.video_emotion_model.analyze_video(video_path)
                    )
                except Exception as e:
                    logger.error(f"[inference] video_emotion failed for {video_path}: {e}")
                    emotions.append({"Assessment": "Neutral"})

        # ── LLM: summarize + relevance ────────────────────────────────────
        # The wrapper selects Groq or Gemini from env vars; no GPU involved.
        llm = Gemini()
        for index, (text, question) in enumerate(zip(texts, questions), start=1):
            summary_result = llm.summarize_result(text)
            relevance_result = llm.relevance_check_result(text, question)

            summarizations.append(summary_result["value"])
            relevance.append(relevance_result["value"])
            question_quality.append(
                {
                    "question_index": index,
                    "degraded": bool(summary_result["degraded"] or relevance_result["degraded"]),
                    "warnings": _unique_messages(
                        [
                            str(summary_result["warning"] or "").strip(),
                            str(relevance_result["warning"] or "").strip(),
                        ]
                    ),
                }
            )

        # ── Text personality traits ───────────────────────────────────────
        for text in texts:
            if model_registry.text_traits_model is None:
                text_traits.append(_default_traits.copy())
            else:
                try:
                    text_traits.append(model_registry.text_traits_model.predict(text))
                except Exception as e:
                    logger.error(f"[inference] text_traits failed: {e}")
                    text_traits.append(_default_traits.copy())

        # ── Audio English score ───────────────────────────────────────────
        for audio_path, text in zip(audio_paths, texts):
            if model_registry.audio_model is None:
                english_scores.append(5.0)
            else:
                try:
                    if audio_path:
                        english_scores.append(
                            model_registry.audio_model.run(text, audio_path)
                        )
                    else:
                        english_scores.append(5.0)
                except Exception as e:
                    logger.error(f"[inference] audio_model failed: {e}")
                    english_scores.append(5.0)

        total_english_score = (
            sum(english_scores) / len(english_scores) if english_scores else 0.0
        )

        # ── Combine scores ────────────────────────────────────────────────
        combined_traits_all: list = []
        video_total_scores:  list = []

        use_video_traits = (
            model_registry.video_traits_model is not None
            and not getattr(model_registry.video_traits_model, "is_fallback", False)
        )
        use_text_traits = model_registry.text_traits_model is not None

        for i in range(len(video_traits)):
            v_traits        = _trait_vector(video_traits[i], trait_order)
            t_traits        = _trait_vector(
                text_traits[i] if i < len(text_traits) else _default_traits,
                trait_order,
            )
            rel_score       = relevance[i]       if i < len(relevance)      else 5
            eng_score       = english_scores[i]  if i < len(english_scores) else 5.0
            cheating_score  = cheating[i]         if i < len(cheating)       else 1.0

            if use_video_traits and use_text_traits:
                combined = [0.7 * v + 0.3 * t for v, t in zip(v_traits, t_traits)]
            elif use_video_traits:
                combined = v_traits
            elif use_text_traits:
                combined = t_traits
            else:
                combined = [0.5] * len(trait_order)
            combined_traits_all.append(combined)

            avg_traits  = sum(combined) / len(combined)
            video_score = ((0.8 * rel_score) + (0.7 * avg_traits) + (0.6 * eng_score)) * cheating_score
            video_total_scores.append(video_score)

        emotions_out    = [e.get("Assessment", "Neutral") for e in emotions]
        combined_traits = np.mean(combined_traits_all, axis=0).tolist() if combined_traits_all else [0.5] * 5
        total_score     = sum(video_total_scores) / len(video_total_scores) if video_total_scores else 0.0

        trait_labels = [
            "Authentic"    if combined_traits[0] > 0.5 else "Self-Interested",
            "Organized"    if combined_traits[1] > 0.5 else "Sloppy",
            "Friendly"     if combined_traits[2] > 0.5 else "Reserved",
            "Comfortable"  if combined_traits[3] > 0.5 else "Uneasy",
            "Imaginative"  if combined_traits[4] > 0.5 else "Practical",
        ]
        result_quality, warnings = _finalize_quality(
            question_quality,
            _build_model_quality_warnings(),
        )

        logger.info(f"[inference] GPU lock released (user={user_id}, job={job_id})")

        return {
            "total_score":         total_score,
            "summarizations":      summarizations,
            "relevance":           relevance,
            "total_english_score": total_english_score,
            "emotions":            emotions_out,
            "trait_labels":        trait_labels,
            "result_quality":      result_quality,
            "warnings":            warnings,
            "question_quality":    question_quality,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Service class
# ─────────────────────────────────────────────────────────────────────────────

class HrService:

    # ── Create HR account ─────────────────────────────────────────────────
    async def create_hr(self, request: HrCreate, db: AsyncSession):
        existing = await db.execute(
            select(HrModel.HR).where(HrModel.HR.email == request.email)
        )
        if existing.scalar():
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"response": False},
            )

        new_hr = HrModel.HR(
            name=request.name,
            email=request.email,
            password=request.password,
        )
        db.add(new_hr)
        await db.commit()
        await db.refresh(new_hr)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"response": True},
        )

    # ── Login ─────────────────────────────────────────────────────────────
    async def hr_login(self, request: GetHr, db: AsyncSession):
        result = await db.execute(
            select(HrModel.HR).where(
                (HrModel.HR.email    == request.email) &
                (HrModel.HR.password == request.password)
            )
        )
        hr_record = result.scalar()
        hr_id     = hr_record.id if hr_record else None

        if hr_id:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"id": hr_id},
            )
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"id": 0},
        )

    # ── Get computed scores ───────────────────────────────────────────────
    async def get_user_scores(self, request: GetUserScores, db: AsyncSession):
        processing_result = await db.execute(
            select(HrModel.VideoProcessing).where(
                (HrModel.VideoProcessing.user_id == request.user_id) &
                (HrModel.VideoProcessing.job_id == request.job_id)
            )
        )
        processing = processing_result.scalar_one_or_none()

        if processing is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"response": None},
            )

        user_result = await db.execute(
            select(UserModel.User).where(UserModel.User.id == request.user_id)
        )
        user = user_result.scalar_one_or_none()

        if user is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"response": None},
            )

        questions_result = await db.execute(
            select(JobModels.JobQuestion)
            .where(JobModels.JobQuestion.job_id == request.job_id)
            .order_by(JobModels.JobQuestion.id)
        )
        job_questions = questions_result.scalars().all()

        videos_result = await db.execute(
            select(UserModel.UserVideo)
            .where(UserModel.UserVideo.userId == request.user_id)
            .order_by(UserModel.UserVideo.videoPath)
        )
        user_videos = videos_result.scalars().all()

        summaries = [
            processing.summarized_text1,
            processing.summarized_text2,
            processing.summarized_text3,
        ]
        relevances = [
            processing.relevance1,
            processing.relevance2,
            processing.relevance3,
        ]
        emotions = [
            processing.emotion1,
            processing.emotion2,
            processing.emotion3,
        ]
        result_quality, quality_warnings, question_quality = _build_persisted_quality_payload(processing)

        video_by_question_id = {}
        fallback_videos = []
        for video in user_videos:
            try:
                question_id = int(Path(video.videoPath).stem)
            except ValueError:
                question_id = None

            if question_id is None:
                fallback_videos.append(video.videoPath)
            else:
                video_by_question_id[question_id] = video.videoPath

        questions = []
        for index, question in enumerate(job_questions):
            video_path = video_by_question_id.get(question.id)
            if video_path is None and index < len(fallback_videos):
                video_path = fallback_videos[index]
            question_meta = question_quality[index] if index < len(question_quality) else {
                "question_index": index + 1,
                "degraded": False,
                "warnings": [],
            }

            questions.append(
                {
                    "question": question.question,
                    "video": video_path,
                    "summary": summaries[index] if index < len(summaries) else None,
                    "relevance": relevances[index] if index < len(relevances) else None,
                    "emotion": emotions[index] if index < len(emotions) else None,
                    "degraded": bool(question_meta.get("degraded")),
                    "warnings": question_meta.get("warnings", []),
                }
            )

        response = {
            "user_id":    user.id,
            "first_name": user.first_name,
            "last_name":  user.last_name,
            "email":      user.email,
            "phone":      user.phone,
            "cv":         user.CV_FilePath,
        }

        response["questions"]           = questions
        response["total_score"]         = processing.total_score
        response["total_english_score"] = processing.total_english_score
        response["trait1"]              = processing.trait1
        response["trait2"]              = processing.trait2
        response["trait3"]              = processing.trait3
        response["trait4"]              = processing.trait4
        response["trait5"]              = processing.trait5
        response["result_quality"]      = result_quality
        response["degraded"]            = result_quality != "complete"
        response["warnings"]            = quality_warnings

        return JSONResponse(status_code=status.HTTP_200_OK, content=response)

    # ── Poll task status ──────────────────────────────────────────────────
    async def get_task_status(self, user_id: int, job_id: int, db: AsyncSession):
        """
        Returns the current processing_status for a compute_scores job.
        Possible values: queued | processing | done | failed | not_found
        """
        result = await db.execute(
            select(HrModel.VideoProcessing).where(
                HrModel.VideoProcessing.user_id == user_id,
                HrModel.VideoProcessing.job_id  == job_id,
            )
        )
        vp = result.scalar_one_or_none()

        if vp is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "not_found",
                    "queued_at": None,
                    "started_at": None,
                    "completed_at": None,
                    "result_quality": None,
                },
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": vp.processing_status,
                "queued_at": _serialize_timestamp(vp.queued_at),
                "started_at": _serialize_timestamp(vp.started_at),
                "completed_at": _serialize_timestamp(vp.completed_at),
                "result_quality": vp.result_quality,
            },
        )

    # ── Trigger score computation (non-blocking) ──────────────────────────
    async def compute_scores(
        self,
        request: ComputeUserScores,
        db: AsyncSession,
        background_tasks: BackgroundTasks,
    ):
        """
        Returns 202 Accepted immediately.
        The full AI pipeline runs in _compute_scores_task() as a BackgroundTask.
        Poll GET /hr/task_status/{user_id}/{job_id} to track progress.
        """
        resolved_hr_id = await resolve_hr_id(
            db,
            explicit_hr_id=request.hr_id,
            job_id=request.job_id,
        )

        # Create (or reset) the DB row with status = "queued" before returning
        vp = await _get_or_create_processing_record(
            db, resolved_hr_id, request.job_id, request.user_id, "queued"
        )

        background_tasks.add_task(
            _compute_scores_task,
            request.user_id,
            request.job_id,
            resolved_hr_id,
        )

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "status":  "queued",
                "queued_at": _serialize_timestamp(vp.queued_at),
                "started_at": _serialize_timestamp(vp.started_at),
                "completed_at": _serialize_timestamp(vp.completed_at),
                "message": (
                    f"Score computation started. "
                    f"Poll /hr/task_status/{request.user_id}/{request.job_id} for updates."
                ),
            },
        )
