import os
import shutil
from pathlib import Path

import whisper


def _ensure_ffmpeg_on_path() -> None:
    if shutil.which("ffmpeg"):
        return

    candidates = []
    localapp = os.environ.get("LOCALAPPDATA")
    if localapp:
        winget_root = (
            Path(localapp)
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
        )
        if winget_root.exists():
            for bin_dir in winget_root.glob("ffmpeg-*-full_build/bin"):
                candidates.append(bin_dir)

    candidates.extend(
        [
            Path(r"C:\ffmpeg\bin"),
            Path(r"C:\Program Files\ffmpeg\bin"),
            Path(r"C:\Program Files\Gyan\ffmpeg\bin"),
        ]
    )

    for bin_dir in candidates:
        if (bin_dir / "ffmpeg.exe").exists():
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            return

    raise FileNotFoundError(
        "ffmpeg.exe not found on PATH. Install FFmpeg and add its bin directory to PATH."
    )

_ensure_ffmpeg_on_path()

model = whisper.load_model("turbo")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(r"D:\LLM\Awesome God - Holydrill_ Telman_ The Excentric (Lyric video)(MP3_160K).mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
