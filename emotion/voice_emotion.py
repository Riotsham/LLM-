from typing import Any, Dict, List

import numpy as np

try:
    import librosa
except ModuleNotFoundError:
    librosa = None


def _to_unit(signal: np.ndarray) -> np.ndarray:
    y = np.asarray(signal, dtype=np.float32).flatten()
    if y.size == 0:
        return y
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return y


def _frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))
    n_frames = 1 + max(0, (len(y) - frame_length) // hop_length)
    if n_frames == 0:
        return np.zeros((0, frame_length), dtype=np.float32)
    starts = np.arange(n_frames) * hop_length
    return np.stack([y[s : s + frame_length] for s in starts]).astype(np.float32)


def _extract_rms_zcr(y: np.ndarray, sample_rate: int) -> tuple[float, float]:
    frame_length = min(2048, max(256, y.size))
    hop_length = max(128, frame_length // 4)

    if librosa is not None:
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(
            y=y,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]
    else:
        frames = _frame_signal(y, frame_length, hop_length)
        if frames.size == 0:
            return 0.0, 0.0
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        crossings = np.sum(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)
        zcr = crossings / (frame_length - 1)

    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0
    return rms_mean, zcr_mean


def _extract_pitch(y: np.ndarray, sample_rate: int) -> float:
    """Estimate average pitch using normalized autocorrelation.

    Using a NumPy-only approach avoids platform-specific crashes seen with
    librosa.yin/numba on some Windows environments.
    """
    if y.size == 0 or sample_rate <= 0:
        return 0.0

    frame_length = min(max(1024, sample_rate // 20), y.size)
    if frame_length < 256:
        return 0.0
    hop = max(128, frame_length // 2)

    fmin = 50.0
    fmax = 400.0
    min_lag = max(1, int(sample_rate / fmax))
    max_lag = max(min_lag + 1, int(sample_rate / fmin))
    max_lag = min(max_lag, frame_length - 1)
    if max_lag <= min_lag:
        return 0.0

    pitches: list[float] = []
    for start in range(0, max(1, y.size - frame_length + 1), hop):
        frame = y[start : start + frame_length]
        if frame.size < frame_length:
            break
        frame = frame - float(np.mean(frame))
        energy = float(np.dot(frame, frame))
        if energy <= 1e-8:
            continue

        corr = np.correlate(frame, frame, mode="full")[frame_length - 1 :]
        if corr.size <= max_lag:
            continue
        segment = corr[min_lag : max_lag + 1]
        if segment.size == 0:
            continue
        lag = int(np.argmax(segment)) + min_lag
        peak = float(corr[lag] / (corr[0] + 1e-8))
        if peak < 0.25:
            continue
        pitches.append(float(sample_rate) / float(lag))

    return float(np.mean(pitches)) if pitches else 0.0


def _normalize(scores: Dict[str, float]) -> List[Dict[str, Any]]:
    clipped = {k: max(0.01, float(v)) for k, v in scores.items()}
    total = sum(clipped.values())
    normalized = [{"label": k, "score": clipped[k] / total} for k in clipped]
    normalized.sort(key=lambda x: x["score"], reverse=True)
    return normalized


def detect_voice_emotion(
    audio_signal: np.ndarray,
    sample_rate: int,
    pitch_baseline: float = 0.0,
) -> Dict[str, Any]:
    """Infer coarse emotional tone from voice features (heuristic, non-diagnostic)."""
    if audio_signal is None or sample_rate <= 0:
        return {"label": "unknown", "score": 0.0, "scores": []}

    y = _to_unit(audio_signal)
    if y.size == 0:
        return {"label": "unknown", "score": 0.0, "scores": []}

    rms_mean, zcr_mean = _extract_rms_zcr(y, sample_rate)
    pitch_hz = _extract_pitch(y, sample_rate)

    pitch_ratio = (pitch_hz / pitch_baseline) if pitch_baseline and pitch_hz > 0 else 1.0

    # Heuristic score construction:
    # - Anxiety/anger rises with high energy + high crossing + raised pitch.
    # - Sadness rises with lower energy/crossing + lowered pitch.
    # - Calm rises when cues are closer to middle.
    anxiety = 0.15
    anger = 0.12
    sadness = 0.12
    calm = 0.25

    if rms_mean > 0.24:
        anxiety += 0.18
        anger += 0.20
        calm -= 0.10
    elif rms_mean < 0.10:
        sadness += 0.22
        calm += 0.05
    else:
        calm += 0.10

    if zcr_mean > 0.18:
        anxiety += 0.22
        anger += 0.12
        calm -= 0.08
    elif zcr_mean < 0.08:
        sadness += 0.12
        calm += 0.08
    else:
        calm += 0.07

    if pitch_ratio >= 1.20:
        anxiety += 0.20
        anger += 0.10
    elif pitch_ratio <= 0.85:
        sadness += 0.20
    else:
        calm += 0.06

    scores = _normalize(
        {
            "anxiety": anxiety,
            "anger": anger,
            "sadness": sadness,
            "calm": calm,
        }
    )
    best = scores[0]
    return {
        "label": best["label"],
        "score": best["score"],
        "scores": scores,
        "features": {
            "rms_mean": rms_mean,
            "zcr_mean": zcr_mean,
            "pitch_hz": pitch_hz,
            "pitch_ratio": pitch_ratio,
        },
    }


def format_voice_emotion_context(result: Dict[str, Any]) -> str:
    label = result.get("label", "unknown")
    score = float(result.get("score", 0.0))
    top_scores = result.get("scores", [])[:3]
    top_text = ", ".join(f"{x['label']}={x['score']:.2f}" for x in top_scores) or "none"

    features = result.get("features", {}) or {}
    rms_mean = float(features.get("rms_mean", 0.0))
    zcr_mean = float(features.get("zcr_mean", 0.0))
    pitch_hz = float(features.get("pitch_hz", 0.0))

    return (
        f"Detected voice emotion: {label} ({score:.2f}). Top scores: {top_text}. "
        f"Features: rms={rms_mean:.3f}, zcr={zcr_mean:.3f}, pitch_hz={pitch_hz:.1f}."
    )
