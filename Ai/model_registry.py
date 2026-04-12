"""
AI Model Registry
=================
Loads heavy AI models once at application startup and keeps them as
module-level singletons.

Startup is best-effort. If a model cannot be loaded because a dependency,
checkpoint file, or remote asset is unavailable, the registry keeps running
in degraded mode and leaves that model attribute as ``None``.
"""

import logging
from importlib import import_module

from Ai.runtime_env import prepare_runtime_environment

logger = logging.getLogger(__name__)


class AIModelRegistry:
    """Holds references to all loaded AI models."""

    def __init__(self):
        self.video_traits_model = None
        self.video_emotion_model = None
        self.cheating_model = None
        self.text_traits_model = None
        self.audio_model = None
        self._loaded = False

    def load_all(self):
        """Load every model from disk exactly once."""
        if self._loaded:
            logger.warning("AIModelRegistry.load_all() called more than once; skipping.")
            return

        prepare_runtime_environment()
        logger.info("Loading AI models into memory...")
        components = [
            (
                "Video_PersonalityTraits",
                "video_traits_model",
                "Ai.Video_Model.personality_traits",
                "Video_PersonalityTraits",
            ),
            (
                "VideoEmotionAnalyzer",
                "video_emotion_model",
                "Ai.Video_Model.emotion_analyzer",
                "VideoEmotionAnalyzer",
            ),
            (
                "CheatingDetection",
                "cheating_model",
                "Ai.Video_Model.cheating_detection",
                "CheatingDetection",
            ),
            (
                "PredictPersonality",
                "text_traits_model",
                "Ai.Text_Model.PredictPersonality",
                "PredictPersonality",
            ),
            (
                "AudioModel",
                "audio_model",
                "Ai.Audio_Model.English_Evaluation",
                "AudioModel",
            ),
        ]
        available = []
        unavailable = []

        for index, (label, attribute, module_name, class_name) in enumerate(
            components, start=1
        ):
            logger.info("  [%s/%s] Loading %s...", index, len(components), label)
            try:
                module = import_module(module_name)
                setattr(self, attribute, getattr(module, class_name)())
                available.append(label)
            except Exception as exc:
                setattr(self, attribute, None)
                unavailable.append(f"{label}: {exc}")
                logger.exception(
                    "Failed to load %s; continuing in degraded mode.", label
                )

        self._loaded = True
        if unavailable:
            logger.warning(
                "AI model registry ready in degraded mode. Available=%s Unavailable=%s",
                available or ["none"],
                unavailable,
            )
        else:
            logger.info("All AI models loaded successfully.")

    def unload_all(self):
        """Release model references on shutdown so the GC can reclaim memory."""
        import gc
        import torch

        self.video_traits_model = None
        self.video_emotion_model = None
        self.cheating_model = None
        self.text_traits_model = None
        self.audio_model = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("AI models unloaded.")


model_registry = AIModelRegistry()
