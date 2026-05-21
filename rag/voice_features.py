import numpy as np

try:
    import librosa
except ModuleNotFoundError:
    librosa = None


def _to_level(value: float, low_th: float, high_th: float) -> str:
    if value < low_th:
        return "low"
    if value > high_th:
        return "high"
    return "medium"


def _frame_signal(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if y.size < frame_length:
        y = np.pad(y, (0, frame_length - y.size))
    n_frames = 1 + max(0, (len(y) - frame_length) // hop_length)
    if n_frames == 0:
        return np.zeros((0, frame_length), dtype=np.float32)
    starts = np.arange(n_frames) * hop_length
    return np.stack([y[s : s + frame_length] for s in starts]).astype(np.float32)


def extract_voice_indicators(audio_signal: np.ndarray, sr: int) -> dict[str, str]:
    default = {"energy_level": "medium", "zcr_level": "medium"}
    if audio_signal is None or sr <= 0:
        return default

    y = np.asarray(audio_signal, dtype=np.float32).flatten()
    if y.size == 0:
        return default

    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

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
            return default
        rms = np.sqrt(np.mean(np.square(frames), axis=1))
        crossings = np.sum(np.signbit(frames[:, 1:]) != np.signbit(frames[:, :-1]), axis=1)
        zcr = crossings / (frame_length - 1)

    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    zcr_mean = float(np.mean(zcr)) if zcr.size else 0.0

    energy_level = _to_level(rms_mean, low_th=0.10, high_th=0.25)
    zcr_level = _to_level(zcr_mean, low_th=0.08, high_th=0.18)
    return {"energy_level": energy_level, "zcr_level": zcr_level}
