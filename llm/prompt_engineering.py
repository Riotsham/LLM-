"""Decision engine utilities for adaptive mental-health response prompting."""

from __future__ import annotations

from typing import Dict


_CRISIS_PHRASES = (
    "kill myself",
    "want to die",
    "want to commit suicide"
)

_MODE_PROMPTS = {
    "crisis": (
        "You are a safety-focused assistant. Prioritize immediate safety, "
        "ask if the person is safe right now, encourage contacting local "
        "emergency services or a trusted person, and provide crisis support "
        "direction with hotline guidance when location is unknown. "
        "Do not provide therapeutic techniques or clinical advice. "
        "Keep language calm, direct, and supportive."
    ),
    "support": (
        "You are an empathetic support assistant. Validate feelings, use warm "
        "non-judgmental language, and ask open-ended questions to help the user "
        "express themselves. Avoid medical or clinical advice."
    ),
    "normal": (
        "You are a friendly conversational assistant. Respond casually and "
        "helpfully with a light, respectful tone."
    ),
    "uncertain": (
        "You are a careful assistant. Emotion signal is uncertain, so respond "
        "gently, ask clarifying questions, and avoid strong assumptions."
    ),
}


def generate_llm_prompt(mode: str, user_text: str) -> str:
    """Build a mode-specific LLM prompt for the given user text."""
    base = _MODE_PROMPTS.get(mode, _MODE_PROMPTS["uncertain"])
    return f"{base}\n\nUser message: {user_text}"


def decide_action(emotion_label: str, text: str, confidence: float) -> Dict[str, str]:
    """Decide response mode and generate a tailored LLM prompt.

    Rules:
    1. crisis: explicit self-harm risk phrases in text
    2. support: emotion in {sadness, anger, fear} and confidence >= 0.6
    3. normal: emotion in {joy, calm} and confidence >= 0.6
    4. uncertain: confidence < 0.6 or missing label
    """
    label = (emotion_label or "").strip().lower()
    user_text = (text or "").strip()
    lowered_text = user_text.lower()

    if any(phrase in lowered_text for phrase in _CRISIS_PHRASES):
        mode = "crisis"
    elif not label or confidence < 0.6:
        mode = "uncertain"
    elif label in {"sadness", "anger", "fear"}:
        mode = "support"
    elif label in {"joy", "calm"}:
        mode = "normal"
    else:
        mode = "uncertain"

    return {"mode": mode, "prompt": generate_llm_prompt(mode, user_text)}
