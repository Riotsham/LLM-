def apply_rules(risk_score: float) -> str:
    """Map a numeric risk score to a discrete risk state."""
    if risk_score >= 0.75:
        return "CRISIS"
    if risk_score >= 0.35:
        return "HIGH_DISTRESS"
    return "NORMAL"
