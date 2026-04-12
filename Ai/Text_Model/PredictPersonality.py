"""
Ai/Text_Model/PredictPersonality.py

Predicts Big-Five personality traits from interview text using BERT embeddings
and per-trait classification models (Keras .h5 or scikit-learn .pkl).

The BERT tokenizer/model wrapper (previously in Helper.py) is merged here
since this is its only consumer.
"""

import torch
import numpy as np
import logging
import re
import tensorflow as tf
import joblib
import os
from pathlib import Path
from transformers import BertTokenizer, BertModel
import preprocessor as p

from Ai.runtime_env import prepare_runtime_environment

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text Preprocessing + BERT Features (merged from Helper class)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_text(sentence: str) -> str:
    """Clean tweet-style noise, URLs, and excess whitespace."""
    sentence = p.clean(sentence)
    sentence = re.sub(r"http\S+",  " ", sentence)
    sentence = re.sub(r"\s+",      " ", sentence).strip()
    sentence = re.sub(r"\|\|\|",   " ", sentence)
    return sentence


# ─────────────────────────────────────────────────────────────────────────────
# Personality Prediction
# ─────────────────────────────────────────────────────────────────────────────

class PredictPersonality:
    """
    Loads per-trait models from Ai/Text_Model/Models/ and predicts Big-Five
    personality scores from raw interview text.

    Supported model formats: .h5 (Keras), .pkl (scikit-learn / joblib)
    """

    def __init__(self):
        prepare_runtime_environment()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.trait_models = self._load_trait_models(
            os.path.join(os.path.dirname(__file__), "Models")
        )

        if torch.cuda.is_available():
            logger.info(
                f"GPU found: {torch.cuda.get_device_name(torch.cuda.current_device())} "
                f"({torch.cuda.device_count()} device(s))"
            )
        else:
            logger.info("PredictPersonality running on CPU")

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_bert_features(
        self, text: str, token_length: int = 512, overlap: int = 256
    ) -> np.ndarray:
        """
        Split long text into overlapping segments, extract BERT [CLS] embeddings,
        and return their mean as a single (1, 768) array.
        """
        tokens = self.tokenizer.tokenize(text)
        segments = []
        start = 0
        while start < len(tokens):
            end = min(start + token_length, len(tokens))
            segments.append(tokens[start:end])
            if end == len(tokens):
                break
            start = end - overlap

        embeddings_list = []
        with torch.no_grad():
            for segment in segments:
                inputs = self.tokenizer(
                    " ".join(segment),
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.device)
                outputs = self.bert_model(**inputs)
                embeddings_list.append(outputs.last_hidden_state[:, 0, :].cpu().numpy())

        if len(embeddings_list) > 1:
            stacked = np.concatenate(embeddings_list, axis=0)
            return np.mean(stacked, axis=0, keepdims=True)
        return embeddings_list[0]

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_trait_models(self, model_dir: str) -> dict:
        """Load all .h5 and .pkl model files from model_dir into a trait→model dict."""
        model_path = Path(model_dir)
        if not model_path.is_dir():
            raise FileNotFoundError(f"Trait model directory not found: {model_dir}")

        models = {}
        for path in model_path.glob("*.*"):
            try:
                if path.suffix == ".h5":
                    models[path.stem] = tf.keras.models.load_model(
                        str(path),
                        compile=False,
                    )
                elif path.suffix == ".pkl":
                    models[path.stem] = joblib.load(str(path))
                else:
                    logger.debug(f"Skipping unsupported model file: {path.name}")
            except Exception as e:
                logger.error(f"Failed to load trait model {path.name}: {e}")

        logger.info(f"Loaded {len(models)} trait model(s): {list(models.keys())}")
        return models

    # ── Inference ─────────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    def predict(self, text: str) -> dict:
        """
        Run the full inference pipeline on raw text.

        Returns a dict mapping trait name → probability score (float).
        """
        clean_text = _preprocess_text(text)
        if not clean_text:
            return {}
        embeddings = self._extract_bert_features(clean_text)

        predictions = {}
        for trait, model in self.trait_models.items():
            try:
                if isinstance(model, tf.keras.Model):
                    raw = model.predict(embeddings, verbose=0)
                else:
                    raw = model.predict(embeddings)
                score = float(self._softmax(raw)[0][1])
                predictions[trait] = score
            except Exception as e:
                logger.error(f"Prediction failed for trait '{trait}': {e}")

        return predictions
