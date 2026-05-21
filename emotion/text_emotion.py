import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Fine-tuned RoBERTa emotion classifier source id.
MODEL_NAME = os.getenv("TEXT_EMOTION_MODEL", "bhadresh-savani/roberta-base-emotion")
LOCAL_MODEL_DIR = Path(os.getenv("TEXT_EMOTION_LOCAL_DIR", "./roberta_emotion_local"))
LOCAL_ONLY = os.getenv("TEXT_EMOTION_LOCAL_ONLY", "0") == "1"
MAX_TOKENS = 256


def _has_local_model() -> bool:
    return (LOCAL_MODEL_DIR / "config.json").exists() and (LOCAL_MODEL_DIR / "model.safetensors").exists()


def _ensure_local_model() -> None:
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if _has_local_model():
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    model.save_pretrained(LOCAL_MODEL_DIR)


@lru_cache(maxsize=1)
def _load_classifier():
    if LOCAL_ONLY:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR, local_files_only=True)
    else:
        _ensure_local_model()
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(LOCAL_MODEL_DIR)
    model.eval()
    return tokenizer, model


def detect_text_emotion(text: str) -> Dict[str, Any]:
    """Return top emotion and distribution for the given text."""
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0, "scores": []}

    try:
        tokenizer, model = _load_classifier()
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS,
        )
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)

        id2label = getattr(model.config, "id2label", {}) or {}
        scores: List[Dict[str, Any]] = []
        for i in range(probs.shape[0]):
            label = id2label.get(i, str(i)).lower()
            score = float(probs[i].item())
            scores.append({"label": label, "score": score})
        scores.sort(key=lambda x: x["score"], reverse=True)

        best = scores[0]
        return {"label": best["label"], "score": best["score"], "scores": scores}
    except Exception as exc:
        return {"label": "unknown", "score": 0.0, "scores": [], "error": str(exc)}


def format_emotion_context(result: Dict[str, Any]) -> str:
    """Format emotion model output for injecting into an LLM system prompt."""
    label = result.get("label", "unknown")
    score = float(result.get("score", 0.0))
    top_scores = result.get("scores", [])[:3]
    top_text = ", ".join(f"{x['label']}={x['score']:.2f}" for x in top_scores) or "none"
    return f"Detected text emotion: {label} ({score:.2f}). Top scores: {top_text}."
