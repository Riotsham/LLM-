import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from typing import Optional
import os

SAMPLE_RATE = 16000
BLOCK_SECONDS = 10  # chunk length


print("Loading model...")
# keep the model loaded at module import to avoid reloading between calls
model = WhisperModel(
    "models/faster-whisper-base.en",
    device="cuda",
    compute_type="float16",
)

_INPUT_DEVICE = os.getenv("ASSISTANT_INPUT_DEVICE")
_INPUT_HOSTAPI = (os.getenv("ASSISTANT_INPUT_HOSTAPI") or "WASAPI").strip().lower()
_PRINTED_DEVICE_INFO = False


def _resolve_input_device():
    """Resolve configured input device to a concrete sounddevice selector (index or name)."""
    configured = (_INPUT_DEVICE or "").strip()
    if not configured:
        return None

    # Allow explicit numeric device index from env.
    if configured.isdigit():
        return int(configured)

    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception:
        return configured

    matches = []
    needle = configured.lower()
    for idx, dev in enumerate(devices):
        try:
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            name = str(dev.get("name", ""))
            if needle not in name.lower():
                continue
            hostapi_idx = int(dev.get("hostapi", -1))
            hostapi_name = ""
            if 0 <= hostapi_idx < len(hostapis):
                hostapi_name = str(hostapis[hostapi_idx].get("name", ""))
            matches.append((idx, name, hostapi_name))
        except Exception:
            continue

    if not matches:
        return configured
    if len(matches) == 1:
        return matches[0][0]

    for idx, _, hostapi_name in matches:
        if _INPUT_HOSTAPI and _INPUT_HOSTAPI in hostapi_name.lower():
            return idx

    # deterministic fallback if no hostapi preference match
    return matches[0][0]


def _resolve_capture_sample_rate(device) -> int:
    """Pick a supported capture sample rate for the selected device."""
    target = int(SAMPLE_RATE)
    try:
        sd.check_input_settings(device=device, samplerate=target, channels=1, dtype="float32")
        return target
    except Exception:
        pass

    try:
        info = sd.query_devices(device, "input")
        fallback = int(float(info.get("default_samplerate", target)))
        if fallback > 0:
            return fallback
    except Exception:
        pass
    return target


def _iter_input_device_candidates(preferred_device):
    """Yield candidate input device selectors in priority order."""
    yielded = set()
    if preferred_device is not None:
        yielded.add(str(preferred_device))
        yield preferred_device

    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = None
    if default_in is not None and int(default_in) >= 0:
        key = str(default_in)
        if key not in yielded:
            yielded.add(key)
            yield int(default_in)

    try:
        devices = sd.query_devices()
    except Exception:
        devices = []
    for idx, dev in enumerate(devices):
        try:
            if int(dev.get("max_input_channels", 0)) <= 0:
                continue
            key = str(idx)
            if key in yielded:
                continue
            yielded.add(key)
            yield idx
        except Exception:
            continue


def _resolve_capture_config(preferred_device):
    """Return (device, capture_rate, channels) validated with PortAudio checks."""
    for candidate in _iter_input_device_candidates(preferred_device):
        try:
            info = sd.query_devices(candidate, "input")
        except Exception:
            continue

        max_in = int(info.get("max_input_channels", 0) or 0)
        if max_in <= 0:
            continue

        channels = 1 if max_in >= 1 else max_in
        if channels <= 0:
            continue

        rates_to_try = []
        preferred_rate = _resolve_capture_sample_rate(candidate)
        if preferred_rate > 0:
            rates_to_try.append(preferred_rate)
        default_rate = int(float(info.get("default_samplerate", SAMPLE_RATE) or SAMPLE_RATE))
        if default_rate > 0 and default_rate not in rates_to_try:
            rates_to_try.append(default_rate)
        if SAMPLE_RATE not in rates_to_try:
            rates_to_try.append(SAMPLE_RATE)

        for rate in rates_to_try:
            try:
                sd.check_input_settings(device=candidate, samplerate=rate, channels=channels, dtype="float32")
                return candidate, int(rate), int(channels)
            except Exception:
                continue

    raise RuntimeError(
        "No usable input audio device was found. "
        "Set ASSISTANT_INPUT_DEVICE to a valid microphone index/name."
    )


def _resample_to_model_rate(audio: np.ndarray, src_rate: int, dst_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Resample mono float audio using linear interpolation (dependency-free)."""
    if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return np.asarray(audio, dtype=np.float32)
    y = np.asarray(audio, dtype=np.float32).flatten()
    if y.size == 0:
        return y
    duration = y.size / float(src_rate)
    dst_size = max(1, int(round(duration * float(dst_rate))))
    src_x = np.linspace(0.0, 1.0, num=y.size, endpoint=False, dtype=np.float64)
    dst_x = np.linspace(0.0, 1.0, num=dst_size, endpoint=False, dtype=np.float64)
    out = np.interp(dst_x, src_x, y).astype(np.float32)
    return out


def record_chunk(seconds: int = BLOCK_SECONDS) -> np.ndarray:
    """Record `seconds` seconds from the default input device and return a 1-D numpy array."""
    global _PRINTED_DEVICE_INFO
    print("🎤 Speak...")
    kwargs = {"dtype": "float32"}
    preferred_device = _resolve_input_device()
    device, capture_rate, channels = _resolve_capture_config(preferred_device)
    kwargs["device"] = device
    kwargs["samplerate"] = capture_rate
    kwargs["channels"] = channels
    if not _PRINTED_DEVICE_INFO:
        try:
            info = sd.query_devices(device, "input")
            print(f"🎙️ Input device: {info['name']}")
            print(f"🎚️ Capture rate: {capture_rate} Hz")
            print(f"🎛️ Channels: {channels}")
        except Exception:
            pass
        _PRINTED_DEVICE_INFO = True
    audio = sd.rec(int(seconds * capture_rate), **kwargs)
    sd.wait()
    audio_mono = audio.flatten()
    return _resample_to_model_rate(audio_mono, src_rate=capture_rate, dst_rate=SAMPLE_RATE)


def transcribe_once(seconds: int = BLOCK_SECONDS, vad_filter: bool = True) -> Optional[str]:
    """Record audio for `seconds`, transcribe with faster-whisper, and return the recognized text or None."""
    audio_data = record_chunk(seconds)
    print("🧠 Processing...")
    segments, _ = model.transcribe(audio_data, vad_filter=vad_filter, beam_size=1)
    text = " ".join(seg.text for seg in segments).strip()
    if not text and vad_filter:
        # Fallback pass for quiet mics where VAD can drop valid speech.
        segments, _ = model.transcribe(audio_data, vad_filter=False, beam_size=1)
        text = " ".join(seg.text for seg in segments).strip()
    if text:
        print("You said:", text)
        return text
    return None


if __name__ == "__main__":
    # convenience: continuously transcribe when run directly
    while True:
        txt = transcribe_once()
        if txt:
            pass
