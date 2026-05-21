import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import librosa
except ModuleNotFoundError:
    librosa = None

# Ensure project root is on sys.path when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emotion.voice_emotion import detect_voice_emotion


def _read_wav_pcm(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = int(wf.getframerate())
        n_channels = int(wf.getnchannels())
        width = int(wf.getsampwidth())
        n_frames = int(wf.getnframes())
        raw = wf.readframes(n_frames)

    if width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width} bytes")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data.astype(np.float32), sr


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    if librosa is not None:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return np.asarray(y, dtype=np.float32), int(sr)
    return _read_wav_pcm(path)


def iter_wavs(paths: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted(p.rglob("*.wav")))
        elif p.is_file() and p.suffix.lower() == ".wav":
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run heuristic voice-emotion detection on WAV files.")
    parser.add_argument("paths", nargs="+", help="WAV file(s) or directory(ies)")
    parser.add_argument("--baseline", type=float, default=0.0, help="Optional pitch baseline in Hz")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")
    args = parser.parse_args()

    wavs = iter_wavs(args.paths)
    if not wavs:
        raise SystemExit("No WAV files found.")

    for wav_path in wavs:
        audio, sr = load_audio(wav_path)
        result = detect_voice_emotion(audio, sr, pitch_baseline=float(args.baseline or 0.0))
        if args.json:
            print(json.dumps({"file": str(wav_path), "result": result}, indent=2))
            continue

        label = result.get("label", "unknown")
        score = float(result.get("score", 0.0))
        features = result.get("features", {}) or {}
        rms_mean = float(features.get("rms_mean", 0.0))
        zcr_mean = float(features.get("zcr_mean", 0.0))
        pitch_hz = float(features.get("pitch_hz", 0.0))
        print(
            f"{wav_path} | label={label} score={score:.2f} "
            f"| rms={rms_mean:.3f} zcr={zcr_mean:.3f} pitch={pitch_hz:.1f}Hz"
        )


if __name__ == "__main__":
    main()
