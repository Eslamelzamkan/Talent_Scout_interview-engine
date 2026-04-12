"""
Hugging Face-backed pronunciation scoring.

This wraps the `Jianshu001/wavlm-phoneme-scorer` checkpoint into a local,
lazy-loadable scorer that returns a 0-10 English pronunciation score.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F_audio
from g2p_en import G2p
from Ai.runtime_env import prepare_runtime_environment

prepare_runtime_environment()

from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, WavLMModel

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
HF_REPO_ID = "Jianshu001/wavlm-phoneme-scorer"
HF_CHECKPOINT_FILENAME = "wavlm_finetuned.pt"
ALIGNMENT_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
BACKBONE_MODEL = "microsoft/wavlm-large"
DEFAULT_PHERR_THRESHOLD = 0.70

ARPABET_TO_IPA = {
    "aa": ["ɑː", "ɑ", "ɒ", "a"],
    "ae": ["æ"],
    "ah": ["ʌ", "ə", "ɐ"],
    "ao": ["ɔː", "ɔ", "ɒ"],
    "aw": ["aʊ"],
    "ax": ["ə", "ɐ", "ʌ"],
    "ay": ["aɪ"],
    "b": ["b"],
    "ch": ["tʃ"],
    "d": ["d"],
    "dh": ["ð"],
    "eh": ["ɛ", "e"],
    "er": ["ɜː", "ɝ", "ɚ", "ɜ"],
    "ey": ["eɪ"],
    "f": ["f"],
    "g": ["ɡ", "g"],
    "hh": ["h"],
    "ih": ["ɪ", "ᵻ"],
    "iy": ["iː", "i"],
    "ir": ["ɪɹ"],
    "jh": ["dʒ"],
    "k": ["k"],
    "l": ["l"],
    "m": ["m"],
    "n": ["n"],
    "ng": ["ŋ"],
    "ow": ["oʊ", "o", "əʊ"],
    "oy": ["ɔɪ"],
    "p": ["p"],
    "r": ["ɹ", "r"],
    "s": ["s"],
    "sh": ["ʃ"],
    "t": ["t"],
    "th": ["θ"],
    "uh": ["ʊ"],
    "uw": ["uː", "u"],
    "ur": ["ʊɹ"],
    "v": ["v"],
    "w": ["w"],
    "y": ["j"],
    "z": ["z"],
    "zh": ["ʒ"],
    "ar": ["ɑːɹ"],
    "oo": ["ʊ", "uː"],
    "dr": ["dɹ"],
    "tr": ["tɹ"],
    "ts": ["ts"],
    "dz": ["dz"],
}

ALL_PHONES = sorted(ARPABET_TO_IPA.keys())
PHONE_TO_ID = {phone: index for index, phone in enumerate(ALL_PHONES)}
N_PHONE_TYPES = len(ALL_PHONES)


class PhoneScorerHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 1024,
        n_phone_types: int = N_PHONE_TYPES,
        phone_emb_dim: int = 32,
        mlp_dim: int = 512,
    ):
        super().__init__()
        self.phone_emb = nn.Embedding(n_phone_types, phone_emb_dim)
        input_dim = hidden_dim + phone_emb_dim + 2
        self.shared = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.score_head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.pherr_head = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        phone_id: torch.Tensor,
        gop: torch.Tensor,
        n_frames: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.phone_emb(phone_id)
        features = torch.cat(
            [hidden_states, embedded, gop.unsqueeze(-1), n_frames.unsqueeze(-1)],
            dim=-1,
        )
        shared = self.shared(features)
        return (
            self.score_head(shared).squeeze(-1),
            self.pherr_head(shared).squeeze(-1),
        )


class HFPronunciationScorer:
    """Pronunciation scorer adapted from the Hugging Face reference pipeline."""

    def __init__(
        self,
        device: str | None = None,
        pherr_threshold: float = DEFAULT_PHERR_THRESHOLD,
    ):
        prepare_runtime_environment()
        self.device = torch.device(device or "cpu")
        self.pherr_threshold = pherr_threshold
        self.cache_dir = str(Path(os.environ.get("HF_HOME", ".cache/huggingface")).resolve())
        self._backbone: WavLMModel | None = None
        self._scorer: PhoneScorerHead | None = None
        self._feature_extractor_backbone: Wav2Vec2FeatureExtractor | None = None
        self._ctc_model: Wav2Vec2ForCTC | None = None
        self._feature_extractor_ctc: Wav2Vec2FeatureExtractor | None = None
        self._vocab: dict[str, int] | None = None
        self._blank_idx: int = 0
        self._g2p: G2p | None = None

    def _ensure_nltk_resources(self) -> None:
        import nltk

        nltk_root = Path(os.environ.get("NLTK_DATA", ".cache/nltk")).resolve()
        nltk_root.mkdir(parents=True, exist_ok=True)
        if str(nltk_root) not in nltk.data.path:
            nltk.data.path.insert(0, str(nltk_root))

        resources = (
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
            ("corpora/cmudict", "cmudict"),
        )
        for lookup_path, resource_name in resources:
            try:
                nltk.data.find(lookup_path)
            except LookupError:
                logger.info("Downloading NLTK resource %s for pronunciation scoring.", resource_name)
                nltk.download(resource_name, download_dir=str(nltk_root), quiet=True)
                nltk.data.find(lookup_path)

    def _load_models(self) -> None:
        if self._backbone is not None:
            return

        self._ensure_nltk_resources()

        checkpoint_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_CHECKPOINT_FILENAME,
            cache_dir=self.cache_dir,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state"]

        backbone_state: dict[str, torch.Tensor] = {}
        scorer_state: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                backbone_state[key[len("backbone."):]] = value
            else:
                scorer_state[key] = value

        self._backbone = WavLMModel.from_pretrained(
            BACKBONE_MODEL,
            cache_dir=self.cache_dir,
            output_hidden_states=False,
            mask_time_prob=0.0,
        )
        self._backbone.load_state_dict(backbone_state, strict=False)
        self._backbone.to(self.device)
        self._backbone.eval()
        self._feature_extractor_backbone = Wav2Vec2FeatureExtractor.from_pretrained(
            BACKBONE_MODEL,
            cache_dir=self.cache_dir,
        )

        self._scorer = PhoneScorerHead()
        self._scorer.load_state_dict(scorer_state)
        self._scorer.to(self.device)
        self._scorer.eval()

        self._ctc_model = Wav2Vec2ForCTC.from_pretrained(
            ALIGNMENT_MODEL,
            cache_dir=self.cache_dir,
        ).to(self.device)
        self._ctc_model.eval()
        self._feature_extractor_ctc = Wav2Vec2FeatureExtractor.from_pretrained(
            ALIGNMENT_MODEL,
            cache_dir=self.cache_dir,
        )

        vocab_path = hf_hub_download(
            ALIGNMENT_MODEL,
            "vocab.json",
            cache_dir=self.cache_dir,
        )
        with open(vocab_path, encoding="utf-8") as handle:
            self._vocab = json.load(handle)
        self._blank_idx = self._vocab.get("<pad>", 0)
        self._g2p = G2p()
        logger.info("Loaded Hugging Face pronunciation scorer on %s.", self.device)

    @staticmethod
    def _load_audio(audio_path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != SAMPLE_RATE:
            waveform = F_audio.resample(waveform, sample_rate, SAMPLE_RATE)
        return waveform

    def _text_to_phonemes(self, text: str) -> list[dict[str, object]]:
        if self._g2p is None:
            raise RuntimeError("Pronunciation scorer models are not loaded.")

        words = re.sub(r"[^\w' ]", " ", text).split()
        phonemes: list[dict[str, object]] = []
        for word_index, word in enumerate(words):
            for phone in self._g2p(word):
                if phone == " ":
                    continue
                clean = re.sub(r"\d", "", phone).lower()
                if clean:
                    phonemes.append(
                        {
                            "phone": clean,
                            "word": word,
                            "word_idx": word_index,
                        }
                    )
        return phonemes

    def _arpabet_to_model_idx(self, phone: str) -> int:
        if self._vocab is None:
            raise RuntimeError("Pronunciation scorer vocabulary is not loaded.")

        for ipa_token in ARPABET_TO_IPA.get(phone, []):
            if ipa_token in self._vocab:
                return self._vocab[ipa_token]
        return self._vocab.get(phone, -1)

    @staticmethod
    def _viterbi_align(
        emissions: torch.Tensor,
        phone_indices: list[int],
        blank_idx: int,
    ) -> list[tuple[int, int]]:
        frame_count, _ = emissions.shape
        state_count = len(phone_indices)
        if state_count == 0 or frame_count < state_count:
            return []

        extended = [blank_idx]
        for phone_index in phone_indices:
            extended.append(phone_index)
            extended.append(blank_idx)

        extended_count = len(extended)
        negative_inf = float("-inf")
        dp = np.full((frame_count, extended_count), negative_inf, dtype=np.float64)
        backpointers = np.zeros((frame_count, extended_count), dtype=np.int32)

        dp[0][0] = emissions[0, extended[0]].item()
        if extended_count > 1:
            dp[0][1] = emissions[0, extended[1]].item()

        for frame in range(1, frame_count):
            for state in range(extended_count):
                emission_score = emissions[frame, extended[state]].item()
                best_score = dp[frame - 1][state]
                best_state = state
                if state > 0 and dp[frame - 1][state - 1] > best_score:
                    best_score = dp[frame - 1][state - 1]
                    best_state = state - 1
                if (
                    state > 1
                    and extended[state] != blank_idx
                    and extended[state] != extended[state - 2]
                    and dp[frame - 1][state - 2] > best_score
                ):
                    best_score = dp[frame - 1][state - 2]
                    best_state = state - 2
                dp[frame][state] = best_score + emission_score
                backpointers[frame][state] = best_state

        if extended_count >= 2 and dp[frame_count - 1][extended_count - 1] >= dp[frame_count - 1][extended_count - 2]:
            state = extended_count - 1
        else:
            state = max(extended_count - 2, 0)

        path: list[tuple[int, int]] = []
        for frame in range(frame_count - 1, -1, -1):
            path.append((frame, extended[state]))
            state = backpointers[frame][state]
        path.reverse()
        return path

    @torch.no_grad()
    def _score_phonemes(
        self,
        waveform: torch.Tensor,
        phone_indices: list[int],
        phone_names: list[str],
    ) -> list[dict[str, float]]:
        if (
            self._feature_extractor_ctc is None
            or self._ctc_model is None
            or self._feature_extractor_backbone is None
            or self._backbone is None
            or self._scorer is None
        ):
            raise RuntimeError("Pronunciation scorer models are not loaded.")

        waveform_np = waveform.squeeze(0).numpy()

        ctc_inputs = self._feature_extractor_ctc(
            waveform_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        ctc_logits = self._ctc_model(ctc_inputs.input_values.to(self.device)).logits
        emissions = torch.log_softmax(ctc_logits, dim=-1).squeeze(0).cpu()

        path = self._viterbi_align(emissions, phone_indices, self._blank_idx)
        if not path:
            raise ValueError("Pronunciation alignment failed.")

        segments: list[tuple[int, list[int]]] = []
        current_token = None
        current_frames: list[int] = []
        for frame, token in path:
            if token == self._blank_idx:
                if current_token is not None:
                    segments.append((current_token, current_frames))
                    current_token = None
                    current_frames = []
                continue
            if token != current_token:
                if current_token is not None:
                    segments.append((current_token, current_frames))
                current_token = token
                current_frames = [frame]
            else:
                current_frames.append(frame)
        if current_token is not None:
            segments.append((current_token, current_frames))

        backbone_inputs = self._feature_extractor_backbone(
            waveform_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        hidden_states = self._backbone(backbone_inputs.input_values.to(self.device)).last_hidden_state.squeeze(0)
        hidden_frame_count = hidden_states.shape[0]
        ctc_frame_count = emissions.shape[0]
        scale = hidden_frame_count / ctc_frame_count

        pooled_states = []
        gop_values = []
        frame_counts = []
        valid_result_indices = []
        results: list[dict[str, float]] = []

        for index, expected_idx in enumerate(phone_indices):
            if index >= len(segments):
                results.append({"gop": -20.0, "score": 0.0, "pherr_prob": 1.0})
                continue

            _, frames = segments[index]
            hidden_start = max(0, int(min(frames) * scale))
            hidden_end = min(hidden_frame_count, int((max(frames) + 1) * scale))
            if hidden_end <= hidden_start:
                hidden_end = hidden_start + 1
            pooled_state = hidden_states[hidden_start:hidden_end].mean(dim=0)

            segment_emissions = emissions[frames]
            target_logprob = segment_emissions[:, expected_idx].mean().item()
            phone_mask = torch.ones(emissions.shape[1], dtype=torch.bool)
            phone_mask[self._blank_idx] = False
            phone_mask[expected_idx] = False
            best_other = segment_emissions[:, phone_mask].max(dim=-1).values.mean().item()
            gop = target_logprob - best_other

            pooled_states.append(pooled_state)
            gop_values.append(gop)
            frame_counts.append(float(len(frames)))
            valid_result_indices.append(index)
            results.append({"gop": gop})

        if not pooled_states:
            return results

        phone_ids = torch.tensor(
            [PHONE_TO_ID.get(phone_names[index], 0) for index in valid_result_indices],
            dtype=torch.long,
            device=self.device,
        )
        hidden_batch = torch.stack(pooled_states).to(self.device)
        gop_batch = torch.tensor(gop_values, dtype=torch.float32, device=self.device)
        frame_batch = torch.tensor(frame_counts, dtype=torch.float32, device=self.device)

        predicted_scores, predicted_error_logits = self._scorer(
            hidden_batch,
            phone_ids,
            gop_batch,
            frame_batch,
        )
        predicted_scores = predicted_scores.detach().cpu().numpy()
        predicted_errors = torch.sigmoid(predicted_error_logits).detach().cpu().numpy()

        for batch_index, result_index in enumerate(valid_result_indices):
            results[result_index]["score"] = float(np.clip(predicted_scores[batch_index], 0, 100))
            results[result_index]["pherr_prob"] = float(predicted_errors[batch_index])

        return results

    def assess(self, audio_path: str, text: str) -> dict[str, object]:
        self._load_models()

        phone_info = self._text_to_phonemes(text)
        if not phone_info:
            raise ValueError("No phonemes extracted from transcript.")

        phone_indices: list[int] = []
        valid_phone_info: list[dict[str, object]] = []
        for phone in phone_info:
            model_index = self._arpabet_to_model_idx(str(phone["phone"]))
            if model_index >= 0:
                phone_indices.append(model_index)
                valid_phone_info.append(phone)

        if not phone_indices:
            raise ValueError("Transcript phonemes could not be mapped to the scorer vocabulary.")

        waveform = self._load_audio(audio_path)
        raw_scores = self._score_phonemes(
            waveform,
            phone_indices,
            [str(phone["phone"]) for phone in valid_phone_info],
        )

        phoneme_results = []
        for phone, raw in zip(valid_phone_info, raw_scores):
            score = float(raw.get("score", 0.0))
            error_probability = float(raw.get("pherr_prob", 1.0))
            phoneme_results.append(
                {
                    "phone": str(phone["phone"]),
                    "word": str(phone["word"]),
                    "word_idx": int(phone["word_idx"]),
                    "score": round(score, 1),
                    "gop": round(float(raw.get("gop", -20.0)), 3),
                    "pherr_prob": round(error_probability, 3),
                    "error": error_probability >= self.pherr_threshold,
                }
            )

        overall_score = float(np.mean([item["score"] for item in phoneme_results])) if phoneme_results else 0.0
        error_count = sum(1 for item in phoneme_results if item["error"])

        return {
            "text": text,
            "overall_score": round(overall_score, 1),
            "n_phonemes": len(phoneme_results),
            "n_errors": error_count,
            "error_rate": round((error_count / len(phoneme_results)) * 100, 1) if phoneme_results else 100.0,
        }

    def score(self, audio_path: str, text: str) -> float:
        result = self.assess(audio_path, text)
        return round(max(0.0, min(10.0, float(result["overall_score"]) / 10.0)), 2)
