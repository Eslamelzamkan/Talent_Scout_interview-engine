"""
Ai/Video_Model/emotion_analyzer.py
Renamed from: facial_expressions.py

Contains VideoEmotionAnalyzer — analyzes per-frame facial emotions in a video
using DeepFace and aggregates them into a final assessment report.
"""

import cv2
import logging
import numpy as np
from deepface import DeepFace
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


class VideoEmotionAnalyzer:
    """Analyzes facial emotions across video frames and returns an assessment report."""

    POSITIVE_EMOTIONS = {"happy", "neutral", "surprise"}
    NEGATIVE_EMOTIONS = {"sad", "angry", "fear", "disgust"}

    def __init__(self):
        self.face_model = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    @staticmethod
    def _to_python_floats(data):
        """Recursively convert NumPy float32/64 to Python float for JSON serialization."""
        if isinstance(data, dict):
            return {k: VideoEmotionAnalyzer._to_python_floats(v) for k, v in data.items()}
        if isinstance(data, list):
            return [VideoEmotionAnalyzer._to_python_floats(v) for v in data]
        if isinstance(data, (np.float32, np.float64)):
            return float(data)
        return data

    def analyze_video(self, video_path: str) -> dict:
        """
        Sample 5 frames per second, run DeepFace emotion analysis on each,
        and return an aggregated report with dominant emotion and an assessment.

        Returns {} if no faces were detected.
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        logger.info(f"Analyzing emotions: FPS={fps}, duration={duration:.1f}s, frames={total_frames}")

        total_emotions: dict = defaultdict(float)
        frames_processed = 0

        for second in range(int(duration)):
            for i in range(5):  # 5 frames per second
                frame_idx = (second * fps) + (i * (fps // 5))
                if frame_idx >= total_frames:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                try:
                    result = DeepFace.analyze(
                        frame,
                        actions=["emotion"],
                        enforce_detection=False,
                        detector_backend="mediapipe",
                    )
                    if isinstance(result, list):
                        result = result[0]

                    for emotion, score in result.get("emotion", {}).items():
                        total_emotions[emotion] += score

                    frames_processed += 1

                except Exception as e:
                    logger.debug(f"Frame {frame_idx} skipped: {e}")

        cap.release()

        if frames_processed == 0:
            logger.warning(f"No faces detected in video: {video_path}")
            return {"Assessment": "Neutral"}

        avg_emotions = {k: v / frames_processed for k, v in total_emotions.items()}
        dominant_emotion = max(avg_emotions, key=avg_emotions.get)

        positive_score = sum(avg_emotions.get(e, 0) for e in self.POSITIVE_EMOTIONS)
        negative_score = sum(avg_emotions.get(e, 0) for e in self.NEGATIVE_EMOTIONS)

        if positive_score > negative_score:
            assessment = "The candidate appeared confident and engaged."
        elif negative_score > positive_score:
            assessment = "The candidate might have felt nervous, frustrated, or disengaged."
        else:
            assessment = "The candidate had mixed reactions, indicating varying confidence levels."

        report = {
            "Dominant Emotion":    dominant_emotion,
            "Emotion Distribution": avg_emotions,
            "Assessment":          assessment,
        }
        return self._to_python_floats(report)
