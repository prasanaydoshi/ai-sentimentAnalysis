# models/sentiment_analyzer.py
"""
Advanced sentiment analysis using Hugging Face transformers with robust loading.

Defaults:
  - Primary: cardiffnlp/twitter-roberta-base-sentiment-latest (Twitter-optimized)
  - Fallback: distilbert-base-uncased-finetuned-sst-2-english (small & reliable)

Features:
  - Retries with exponential backoff on hub timeouts (e.g., 504)
  - Customizable HF cache dir via TRANSFORMERS_CACHE
  - Optional: pass model_name in constructor to override default
"""

from __future__ import annotations

import os
import time
import logging
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

logger = logging.getLogger("sentiment_analyzer")


def _env_default(name: str, default: Optional[str]) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default


class SentimentAnalyzer:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Args:
            model_name: HF model id. If None, uses primary then fallback automatically.
            device: 'cpu' or 'cuda'. If None, auto-detect.
        """
        self.primary_model = model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.fallback_model = "distilbert-base-uncased-finetuned-sst-2-english"

        # Hub settings (increase timeouts; allow hf-transfer if installed)
        os.environ.setdefault("HF_HUB_TIMEOUT", "180")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # ignored if package not present

        # Respect user cache if provided
        # e.g., export TRANSFORMERS_CACHE=/Users/you/.cache/hf
        self.cache_dir = os.getenv("TRANSFORMERS_CACHE", None)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.pipeline: TextClassificationPipeline = self._load_pipeline_with_fallback()

    # -------------------------
    # Loading helpers
    # -------------------------
    def _load_pipeline_with_fallback(self) -> TextClassificationPipeline:
        # Try primary with retries
        try:
            return self._build_pipeline(self.primary_model)
        except Exception as e1:
            logger.warning(f"Primary model load failed: {e1}")
            logger.warning(f"Falling back to {self.fallback_model} ...")
            try:
                return self._build_pipeline(self.fallback_model)
            except Exception as e2:
                logger.error(f"Fallback model load also failed: {e2}")
                raise

    def _build_pipeline(self, model_id: str) -> TextClassificationPipeline:
        max_retries = 4
        delay = 2.0
        last_err: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Loading model: {model_id} (attempt {attempt}/{max_retries})")
                tok = AutoTokenizer.from_pretrained(model_id, cache_dir=self.cache_dir)
                mdl = AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=self.cache_dir)
                pipe = TextClassificationPipeline(
                    model=mdl,
                    tokenizer=tok,
                    device=0 if self.device == "cuda" else -1,
                    top_k=None,
                    function_to_apply="softmax",
                    truncation=True
                )
                logger.info(f"Model loaded: {model_id}")
                return pipe
            except Exception as e:
                last_err = e
                logger.warning(f"Model load failed ({model_id}): {e}")
                if attempt < max_retries:
                    sleep_for = delay
                    logger.warning(f"Retrying in {sleep_for:.1f}s ...")
                    time.sleep(sleep_for)
                    delay *= 1.8

        # Exhausted retries
        assert last_err is not None
        raise last_err

    # -------------------------
    # Public API
    # -------------------------
    def get_model_info(self) -> str:
        try:
            name = getattr(self.pipeline.model.config, "_name_or_path", "unknown")
            return str(name)
        except Exception:
            return "unknown"

    def _postprocess_label(self, label: str) -> str:
        """Normalize different model labels to ['positive','negative','neutral']."""
        l = label.lower()
        if "pos" in l:
            return "positive"
        if "neg" in l:
            return "negative"
        if "neutral" in l or "neu" in l:
            return "neutral"
        # DistilBERT SST-2 is binary; map anything else to positive/negative
        return "positive" if "1" in l else "negative"

    def analyze_texts(self, texts: List[str]) -> List[Dict]:
        if not texts:
            return []

        outputs = self.pipeline(texts)
        # The pipeline may return a dict or list depending on top_k; normalize
        results: List[Dict] = []
        for text, out in zip(texts, outputs):
            if isinstance(out, list):
                # list of dicts with labels & scores – take the max
                best = max(out, key=lambda d: d.get("score", 0.0))
            else:
                best = out

            label = self._postprocess_label(best.get("label", "neutral"))
            score = float(best.get("score", 0.0))

            # Fake class probabilities when the model is binary
            probs = {"positive_prob": 0.0, "negative_prob": 0.0, "neutral_prob": 0.0}
            if label == "positive":
                probs["positive_prob"] = score
            elif label == "negative":
                probs["negative_prob"] = score
            else:
                probs["neutral_prob"] = score

            results.append({
                "text": text,
                "sentiment_label": label,
                "sentiment_score": (1.0 if label == "positive" else (-1.0 if label == "negative" else 0.0)) * score,
                "confidence": score,
                **probs
            })
        return results

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = "cleaned_text") -> pd.DataFrame:
        if df is None or df.empty:
            return df

        texts = df[text_column].fillna("").astype(str).tolist()
        results = self.analyze_texts(texts)
        if not results:
            return df

        res_df = pd.DataFrame(results)
        # Keep original df + new columns (avoid duplicating text column)
        keep_cols = [c for c in df.columns if c != text_column]
        merged = pd.concat([df[keep_cols].reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)
        return merged


class FinancialSentimentEnhancer:
    """
    Lightweight post-processor for financial context. Keeps it simple to avoid extra downloads.
    """
    def enhance_sentiment(self, text: str, base: Dict) -> Dict:
        label = base.get("sentiment_label", "neutral")
        score = float(base.get("sentiment_score", 0.0))

        t = (text or "").lower()
        # Simple heuristics
        bull_words = ("beat earnings", "revenue beat", "record high", "upgrade", "bullish", "buyback", "raises guidance")
        bear_words = ("miss earnings", "downgrade", "guidance cut", "sec probe", "lawsuit", "bankruptcy", "bearish")

        if any(w in t for w in bull_words) and label != "positive":
            label = "positive"
            score = abs(score)
        if any(w in t for w in bear_words) and label != "negative":
            label = "negative"
            score = -abs(score)

        return {
            **base,
            "sentiment_label": label,
            "sentiment_score": score,
        }
