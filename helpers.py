from __future__ import annotations

import re


def is_greeting(text: str) -> bool:
    """Return True for short, neutral greeting-like openers."""
    if not text:
        return False

    cleaned = re.sub(r"[^\w\s']", " ", text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return False

    direct = {
        "hi",
        "hello",
        "hey",
        "how are you",
        "how are you doing",
        "whats up",
        "what's up",
        "sup",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
    }
    if cleaned in direct:
        return True

    tokens = cleaned.split()
    if len(tokens) <= 3 and any(g in tokens for g in {"hi", "hello", "hey"}):
        return True
    if cleaned.startswith("how are you") and len(tokens) <= 5:
        return True
    if ("what's up" in text.lower() or "whats up" in cleaned) and len(tokens) <= 5:
        return True
    return False


def is_affirmative(text: str) -> bool:
    """Detect simple affirmative replies."""
    if not text:
        return False
    cleaned = text.strip().lower()
    return cleaned in {"yes", "yep", "yeah", "sure", "ok", "okay", "please", "y"} or bool(
        re.match(r"^(yes|yeah|yep|sure|ok|okay)\b", cleaned)
    )


def is_negative(text: str) -> bool:
    """Detect simple negative replies."""
    if not text:
        return False
    cleaned = text.strip().lower()
    return cleaned in {"no", "nope", "nah", "not now", "later"} or bool(
        re.match(r"^(no|nah|nope)\b", cleaned)
    )


def detect_breathing_offer(text: str) -> bool:
    """Heuristic: did the assistant offer a breathing exercise?"""
    if not text:
        return False
    lowered = text.lower()
    return "breath" in lowered and ("with me" in lowered or "together" in lowered)


def run_breathing_sequence() -> str:
    """Deterministic guided breathing sequence."""
    return (
        "Okay, let's do a short breathing exercise together.\n\n"
        "1) Inhale through your nose for 4 seconds.\n"
        "2) Hold for 2 seconds.\n"
        "3) Exhale slowly for 6 seconds.\n"
        "4) Repeat this cycle 5 times, counting each breath.\n\n"
        "I'll stay with you through it. Let me know when you're done."
    )


def decline_exercise_response() -> str:
    """Response when the user declines an exercise."""
    return (
        "No problem. We can skip it.\n\n"
        "If you want, tell me what's feeling most difficult right now."
    )


def is_out_of_scope(text: str) -> bool:
    """Returns True if input appears unrelated to mental health."""
    if not text:
        return False
    t = text.strip().lower()

    programming = {
        "javascript", "python", "java", "c++", "c#", "html", "css", "sql", "coding",
        "programming", "debug", "bug", "compiler", "api", "framework", "react", "node",
    }
    math_topics = {
        "algebra", "calculus", "geometry", "trigonometry", "equation", "derivative",
        "integral", "matrix", "probability", "statistics",
    }
    sports = {
        "football", "soccer", "basketball", "baseball", "hockey", "tennis", "golf",
        "nba", "nfl", "mlb", "nhl",
    }
    trivia = {
        "capital", "population", "weather", "stock price", "exchange rate", "history",
        "president", "prime minister", "timeline", "wikipedia",
    }

    keywords = programming | math_topics | sports | trivia
    return any(k in t for k in keywords)


def redirect_message() -> str:
    return (
        "I'm here primarily to support your emotional well-being. "
        "If something is stressing or affecting you, I'm happy to help with that. "
        "How are you feeling right now?"
    )
