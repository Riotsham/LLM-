import re


# Weighted phrase patterns for rule-based risk detection.
# Higher weights signal stronger indicators of self-harm or suicide intent.
_CRISIS_PATTERNS: list[tuple[str, float]] = [
    (r"\bkill myself\b", 0.9),
    (r"\bend my life\b", 0.9),
    (r"\btake my life\b", 0.9),
    (r"\bwant to die\b", 0.85),
    (r"\bi want to die\b", 0.85),
    (r"\bno reason to live\b", 0.8),
    (r"\bcan't go on\b", 0.75),
    (r"\bsuicid(al|e)\b", 0.9),
    (r"\bself[- ]?harm\b", 0.8),
    (r"\bcut myself\b", 0.8),
    (r"\bhurt myself\b", 0.75),
    (r"\boverdose\b", 0.8),
    (r"\bjump off\b", 0.8),
]

_DISTRESS_PATTERNS: list[tuple[str, float]] = [
    (r"\bdepressed\b", 0.3),
    (r"\bhopeless\b", 0.35),
    (r"\bworthless\b", 0.35),
    (r"\boverwhelmed\b", 0.25),
    (r"\bpanicking\b", 0.25),
    (r"\bcan't cope\b", 0.3),
    (r"\bso alone\b", 0.25),
    (r"\bno one cares\b", 0.3),
    (r"\bnumb\b", 0.2),
    (r"\bempty\b", 0.2),
]


def detect_risk(text: str) -> float:
    """Return a risk score between 0.0 and 1.0 using rule-based pattern matching."""
    if not text:
        return 0.0

    cleaned = " ".join(text.lower().split())
    score = 0.0

    for pattern, weight in _CRISIS_PATTERNS:
        if re.search(pattern, cleaned):
            score += weight

    for pattern, weight in _DISTRESS_PATTERNS:
        if re.search(pattern, cleaned):
            score += weight

    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
