"""
Ai/Audio_Model/English_Evaluation.py

Evaluates a candidate's spoken English across four dimensions:
  Accuracy, Fluency, Completeness, Prosodic quality

Uses a multimodal BERT + Wav2Vec2 model (EnglishModel) trained on audio + transcript pairs.
"""

import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import logging
import math
import librosa
import os
import re
from typing import List

from Ai.runtime_env import prepare_runtime_environment
from Ai.Text_Model.Gemini import Gemini
from .EnglishModel import EnglishModel
from .hf_pronunciation import HFPronunciationScorer
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class AudioModel:
    """
    Scores spoken English from an (audio_path, transcript) pair.

    Pipeline:
      load audio → resample → chunk → EnglishModel → weighted score per chunk → mean
    """

    def __init__(
        self,
        llm_name: str = "answerdotai/ModernBERT-base",
        speech_encoder_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-english",
        apply_preprocessing: bool = False,
    ):
        prepare_runtime_environment()
        self.device              = "cuda" if torch.cuda.is_available() else "cpu"
        self.apply_preprocessing = apply_preprocessing
        self.target_sample_rate  = 16_000
        self.mean_waveform_length = 80_000   # 5 seconds at 16 kHz
        self.is_fallback = False
        weights_path = os.path.join(
            os.path.dirname(__file__), "EnglishModel_weights_best_epoch.pth"
        )
        self.model = None
        self.tokenizer = None
        self.llm = None
        self.pronunciation_model = None
        self.pronunciation_init_failed = False
        self.mode = "heuristic"

        if not os.path.exists(weights_path):
            self.is_fallback = True
            self.mode = "hf_pronunciation"
            logger.warning(
                "EnglishModel weights not found at %s. Using Hugging Face pronunciation fallback.",
                weights_path,
            )
            return

        try:
            self.model = EnglishModel(
                llm_name=llm_name,
                speech_encoder_name=speech_encoder_name,
            )
            self.model.load_state_dict(
                torch.load(weights_path, weights_only=True, map_location=self.device)
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
            self.mode = "full"
            logger.info(f"AudioModel loaded on {self.device}")
        except Exception as exc:
            self.model = None
            self.tokenizer = None
            self.is_fallback = True
            self.mode = "hf_pronunciation"
            logger.warning(
                "AudioModel initialization failed; falling back to Hugging Face pronunciation scoring: %s",
                exc,
            )

    # ── Audio preprocessing ───────────────────────────────────────────────────

    def _resample(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        if original_sr != self.target_sample_rate:
            waveform = T.Resample(original_sr, self.target_sample_rate)(waveform)
        return waveform

    def _pad_chunk(self, waveform: torch.Tensor, length: int = 80_000) -> torch.Tensor:
        if waveform.size(1) < length:
            waveform = F.pad(waveform, (0, length - waveform.size(1)))
        return waveform

    def _split_waveform(
        self, waveform: torch.Tensor, segment_length: int = 80_000
    ) -> List[torch.Tensor]:
        total   = waveform.size(1)
        n_segs  = math.ceil(total / segment_length)
        logger.debug(f"Splitting audio into {n_segs} chunk(s)")
        segments = []
        for i in range(n_segs):
            seg = waveform[:, i * segment_length: (i + 1) * segment_length]
            if seg.size(1) < segment_length:
                seg = self._pad_chunk(seg, segment_length)
            segments.append(seg)
        return segments

    def _prepare_inputs(
        self, text: str, audio_path: str, text_max_length: int = 12
    ):
        """Load audio from disk and tokenize the transcript."""
        waveform, sample_rate = librosa.load(audio_path, sr=self.target_sample_rate)
        waveform = torch.tensor(waveform).unsqueeze(0)
        waveform = self._resample(waveform, sample_rate)

        if self.apply_preprocessing:
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=text_max_length,
            ).to(self.device)
            chunks = (
                self._split_waveform(waveform.to(self.device))
                if waveform.size(1) > self.mean_waveform_length
                else [self._pad_chunk(waveform).to(self.device)]
            )
        else:
            tokens = self.tokenizer(text, return_tensors="pt").to(self.device)
            chunks = [waveform.to(self.device)]

        return chunks, tokens.input_ids, tokens.attention_mask

    # ── Inference ─────────────────────────────────────────────────────────────

    def _score_chunk(
        self,
        waveform: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> float:
        """Run one audio chunk through the model and return the weighted score."""
        torch.cuda.empty_cache()
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                waveforms=waveform,
                attention_masks=attention_mask,
            )

        accuracy, fluency, completeness, prosodic, total = (
            torch.round(output).long().squeeze(0).tolist()
        )
        logger.debug(
            f"Chunk scores — accuracy={accuracy}, fluency={fluency}, "
            f"prosodic={prosodic}, completeness={completeness}, total={total}"
        )

        return 0.2 * accuracy + 0.2 * fluency + 0.25 * prosodic + 0.25 * completeness + 0.1 * total

    def run(self, audio_transcript: str, audio_path: str) -> float:
        """Score a single candidate answer. Returns the mean score across all chunks."""
        if self.mode != "full" or self.model is None or self.tokenizer is None:
            return self._heuristic_score(audio_transcript, audio_path)

        chunks, input_ids, attn_mask = self._prepare_inputs(audio_transcript, audio_path)
        scores = []
        for i, chunk in enumerate(chunks, start=1):
            logger.debug(f"Scoring chunk {i}/{len(chunks)}")
            scores.append(self._score_chunk(chunk, input_ids, attn_mask))
        return sum(scores) / len(scores)

    def get_total_applicant_score(self, scores: List[float]) -> float:
        """Average per-answer scores into a single applicant-level English score."""
        return sum(scores) / len(scores)

    def _get_pronunciation_model(self):
        if self.pronunciation_model is not None:
            return self.pronunciation_model
        if self.pronunciation_init_failed:
            return None

        try:
            self.pronunciation_model = HFPronunciationScorer()
        except Exception as exc:
            self.pronunciation_init_failed = True
            self.mode = "heuristic"
            logger.warning(
                "Hugging Face pronunciation scorer initialization failed; using heuristic fallback: %s",
                exc,
            )
            return None

        self.mode = "hf_pronunciation"
        return self.pronunciation_model

    def _base_heuristic_score(self, audio_transcript: str, audio_path: str) -> float:
        duration = 0.0
        try:
            duration = float(librosa.get_duration(path=audio_path))
        except Exception as exc:
            logger.debug("Could not read audio duration for %s: %s", audio_path, exc)

        words = re.findall(r"[A-Za-z']+", audio_transcript.lower())
        word_count = len(words)
        unique_ratio = len(set(words)) / word_count if word_count else 0.0
        words_per_minute = (word_count / duration) * 60 if duration > 0 else 0.0

        if 90 <= words_per_minute <= 170:
            pace_score = 1.0
        elif words_per_minute == 0:
            pace_score = 0.2
        else:
            pace_score = max(0.2, 1 - (abs(words_per_minute - 130) / 130))

        content_score = min(1.0, word_count / 45) if word_count else 0.1
        diversity_score = min(1.0, max(0.15, unique_ratio))
        duration_score = min(1.0, duration / 20) if duration > 0 else 0.1

        score = 10 * (
            0.35 * content_score
            + 0.30 * pace_score
            + 0.20 * diversity_score
            + 0.15 * duration_score
        )
        return round(max(0.0, min(10.0, score)), 2)

    def _heuristic_score(self, audio_transcript: str, audio_path: str) -> float:
        """Fallback score when the trained checkpoint is unavailable."""
        heuristic_score = self._base_heuristic_score(audio_transcript, audio_path)

        pronunciation_score = None
        if audio_transcript.strip() and audio_path:
            pronunciation_model = self._get_pronunciation_model()
            if pronunciation_model is not None:
                try:
                    pronunciation_score = pronunciation_model.score(audio_path, audio_transcript)
                    self.mode = "hf_pronunciation"
                except Exception as exc:
                    logger.warning(
                        "Hugging Face pronunciation scoring failed for %s; falling back to heuristic scoring: %s",
                        audio_path,
                        exc,
                    )
                    self.mode = "heuristic"
                    pronunciation_score = None

        if pronunciation_score is not None:
            blended = (0.80 * pronunciation_score) + (0.20 * heuristic_score)
            return round(max(0.0, min(10.0, blended)), 2)

        llm_score = None
        if audio_transcript.strip():
            try:
                if self.llm is None:
                    self.llm = Gemini()
                llm_score = self.llm.english_score(audio_transcript)
            except Exception as exc:
                logger.debug("LLM English fallback failed: %s", exc)

        if llm_score is None:
            return heuristic_score

        blended = (0.65 * llm_score) + (0.35 * heuristic_score)
        return round(max(0.0, min(10.0, blended)), 2)
