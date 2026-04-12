"""
Ai/media_utils.py

Audio extraction and transcription utilities used by the HR scoring pipeline.
"""

import logging
import subprocess
from pathlib import Path

import numpy as np
import torch
import whisper
from fastapi import HTTPException

from Ai.runtime_env import prepare_runtime_environment

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class MediaUtils:
    """Static utilities for extracting audio from video and transcribing it."""

    _whisper_model = None
    _whisper_model_name = None

    @staticmethod
    def extract_audio(user_id: int, job_id: int, question_id) -> str:
        """
        Extract the audio track from an uploaded video and save it as WAV.

        Returns:
            Absolute path to the extracted WAV file.

        Raises:
            HTTPException 404 if the source video is not found.
            HTTPException 500 on extraction failure.
        """
        try:
            ffmpeg_exe = prepare_runtime_environment()
            video_path = UPLOAD_DIR / str(job_id) / str(user_id) / "videos" / f"{question_id}.mp4"

            if not video_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Video file not found: {video_path}",
                )

            audio_dir = UPLOAD_DIR / str(job_id) / str(user_id) / "audios"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / f"{question_id}.wav"

            command = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                str(audio_path),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)

            logger.info("Extracted audio: %s", audio_path)
            return str(audio_path)

        except HTTPException:
            raise
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            logger.error("Audio extraction failed: %s", stderr or exc)
            raise HTTPException(status_code=500, detail=f"Error extracting audio: {stderr or exc}")
        except Exception as exc:
            logger.error("Audio extraction failed: %s", exc)
            raise HTTPException(status_code=500, detail=f"Error extracting audio: {exc}")

    @classmethod
    def _get_whisper_model(cls, device: str):
        model_name = "base.en"
        download_root = (Path(".cache") / "whisper").resolve()
        download_root.mkdir(parents=True, exist_ok=True)
        if cls._whisper_model is None or cls._whisper_model_name != model_name:
            cls._whisper_model = whisper.load_model(
                model_name,
                device=device,
                download_root=str(download_root),
            )
            cls._whisper_model_name = model_name
        return cls._whisper_model

    @staticmethod
    def _load_audio_with_ffmpeg(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
        ffmpeg_exe = prepare_runtime_environment()
        command = [
            ffmpeg_exe,
            "-nostdin",
            "-threads",
            "0",
            "-i",
            audio_path,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-",
        ]
        process = subprocess.run(command, check=True, capture_output=True)
        return np.frombuffer(process.stdout, np.int16).astype(np.float32) / 32768.0

    @staticmethod
    async def transcribe_audio(audio_path: str) -> str:
        """
        Transcribe an audio file to text using Whisper.

        Returns:
            The transcribed text string, or "" on failure.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            audio = MediaUtils._load_audio_with_ffmpeg(audio_path)
            audio = whisper.pad_or_trim(audio)
            model = MediaUtils._get_whisper_model(device)
            result = model.transcribe(audio, fp16=(device == "cuda"))
            transcript = result["text"]
            logger.info("Transcribed %s: %s chars", audio_path, len(transcript))
            return transcript
        except Exception as exc:
            logger.error("Transcription failed for %s: %s", audio_path, exc)
            return ""


# Backward-compat alias so existing code using HelperText still works
HelperText = MediaUtils
