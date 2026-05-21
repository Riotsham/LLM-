import asyncio
import subprocess
import tempfile
import ctypes
import os
import shlex
import socket
import time
from datetime import datetime
from pathlib import Path
from utils.rpi_ssh import resolve_rpi_host, resolve_ssh_key_path

try:
    from edge_tts import Communicate
except Exception:
    Communicate = None


_ASSISTANT_STATE_FILE = Path(os.getenv("ASSISTANT_STATE_FILE", "/tmp/llm_tts/assistant_state.txt"))
_VALID_ASSISTANT_STATES = {"idle", "listening", "thinking", "speaking"}
_RPI_UNREACHABLE_UNTIL = 0.0


def _can_attempt_rpi_connection(host: str, port: str) -> bool:
    """Fast TCP pre-check + cooldown to avoid repeated SSH timeout spam."""
    global _RPI_UNREACHABLE_UNTIL

    now = time.monotonic()
    if now < _RPI_UNREACHABLE_UNTIL:
        return False

    try:
        port_int = int(port or "22")
    except Exception:
        port_int = 22

    probe_timeout = float(os.getenv("RPI_CONNECT_PROBE_TIMEOUT", "0.7").strip() or "0.7")
    probe_attempts = int(os.getenv("RPI_CONNECT_PROBE_ATTEMPTS", "3").strip() or "3")
    probe_retry_delay = float(os.getenv("RPI_CONNECT_PROBE_RETRY_DELAY", "0.2").strip() or "0.2")
    cooldown = float(os.getenv("RPI_CONNECT_RETRY_COOLDOWN", "30").strip() or "30")
    if probe_attempts < 1:
        probe_attempts = 1

    for attempt in range(probe_attempts):
        try:
            with socket.create_connection((host, port_int), timeout=probe_timeout):
                return True
        except Exception:
            if attempt < probe_attempts - 1 and probe_retry_delay > 0:
                time.sleep(probe_retry_delay)

    _RPI_UNREACHABLE_UNTIL = now + max(1.0, cooldown)
    return False


async def speak_async(text: str, out_path: str = "response.mp3") -> None:
    """Synthesize `text` with Edge TTS and save to mp3."""
    if not text:
        return
    if Communicate is None:
        print("(edge-tts unavailable) Reply:", text)
        return

    try:
        communicate = Communicate(text, voice="en-US-JennyNeural")
        await communicate.save(out_path)
    except Exception as e:
        print(f"(TTS failed: {e}) Reply:", text)


def _is_rpi_output_enabled() -> bool:
    output_mode = os.getenv("TTS_OUTPUT_DEVICE", "").strip().lower()
    if output_mode in {"rpi", "raspi", "raspberrypi", "remote"}:
        return True
    return os.getenv("RPI_TTS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _write_local_activity_state(state: str) -> None:
    try:
        _ASSISTANT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _ASSISTANT_STATE_FILE.write_text(state, encoding="utf-8")
    except Exception:
        pass


def _push_activity_state_to_rpi(state: str) -> None:
    port = os.getenv("RPI_PORT", "22").strip() or "22"
    host = resolve_rpi_host(port=port)
    if not host:
        return
    if not _can_attempt_rpi_connection(host, port):
        return

    user = os.getenv("RPI_USER", "pi").strip() or "pi"
    remote_dir = os.getenv("RPI_REMOTE_DIR", "/tmp/llm_tts").strip() or "/tmp/llm_tts"
    ssh_key_path = resolve_ssh_key_path(os.getenv("RPI_SSH_KEY", ""), cwd=Path(__file__).resolve().parents[1])
    target = f"{user}@{host}"
    connect_timeout = os.getenv("RPI_SSH_CONNECT_TIMEOUT", "4").strip() or "4"
    ssh_cmd = [
        "ssh",
        "-p",
        port,
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        "ConnectionAttempts=1",
    ]
    if ssh_key_path:
        ssh_cmd.extend(["-i", ssh_key_path, "-o", "IdentitiesOnly=yes"])

    remote_file = f"{remote_dir}/assistant_state.txt"
    remote_cmd = (
        f"mkdir -p {shlex.quote(remote_dir)}; "
        f"printf %s {shlex.quote(state)} > {shlex.quote(remote_file)}"
    )
    try:
        subprocess.run([*ssh_cmd, target, remote_cmd], check=True, timeout=8)
    except Exception:
        pass


def set_assistant_activity(state: str) -> None:
    normalized = (state or "").strip().lower()
    if normalized not in _VALID_ASSISTANT_STATES:
        normalized = "idle"
    _write_local_activity_state(normalized)
    if _is_rpi_output_enabled():
        _push_activity_state_to_rpi(normalized)


def _play_on_raspberry_pi(path: str, text: str, timeout_seconds: int = 30) -> bool:
    """Send mp3 to Raspberry Pi via SCP and play it there via SSH."""
    port = os.getenv("RPI_PORT", "22").strip() or "22"
    host = resolve_rpi_host(port=port)
    if not host:
        print("(RPI playback skipped) Set RPI_HOST to enable Raspberry Pi audio output.")
        return False
    if not _can_attempt_rpi_connection(host, port):
        print("(RPI playback skipped) Raspberry Pi is currently unreachable.")
        return False

    user = os.getenv("RPI_USER", "pi").strip() or "pi"
    remote_dir = os.getenv("RPI_REMOTE_DIR", "/tmp/llm_tts").strip() or "/tmp/llm_tts"
    ssh_key_path = resolve_ssh_key_path(os.getenv("RPI_SSH_KEY", ""), cwd=Path(__file__).resolve().parents[1])
    mood = os.getenv("RPI_MOOD", "sad").strip().lower()
    if mood not in {"happy", "sad", "anger"}:
        mood = "sad"
    remote_file = f"{remote_dir}/response.mp3"
    remote_text_file = f"{remote_dir}/last_text.txt"
    remote_mood_file = f"{remote_dir}/mood.txt"
    remote_tts_flag_file = f"{remote_dir}/tts_playing.flag"
    target = f"{user}@{host}"
    connect_timeout = os.getenv("RPI_SSH_CONNECT_TIMEOUT", "4").strip() or "4"
    ssh_cmd = [
        "ssh",
        "-p",
        port,
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        "ConnectionAttempts=1",
    ]
    scp_cmd = [
        "scp",
        "-P",
        port,
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
    ]
    if ssh_key_path:
        ssh_cmd.extend(["-i", ssh_key_path, "-o", "IdentitiesOnly=yes"])
        scp_cmd.extend(["-i", ssh_key_path, "-o", "IdentitiesOnly=yes"])

    try:
        subprocess.run(
            [*ssh_cmd, target, f"mkdir -p {shlex.quote(remote_dir)}"],
            check=True,
            timeout=10,
        )
        subprocess.run(
            [*scp_cmd, str(Path(path).resolve()), f"{target}:{remote_file}"],
            check=True,
            timeout=20,
        )

        remote_cmd = (
            f"printf %s {shlex.quote(text)} > {shlex.quote(remote_text_file)}; "
            f"printf %s {shlex.quote(mood)} > {shlex.quote(remote_mood_file)}; "
            f"touch {shlex.quote(remote_tts_flag_file)}; "
            f"trap 'rm -f {shlex.quote(remote_tts_flag_file)}' EXIT; "
            "if command -v ffplay >/dev/null 2>&1 && "
            f"ffplay -nodisp -autoexit -loglevel quiet {shlex.quote(remote_file)}; then "
            "exit 0; "
            "fi; "
            "if command -v cvlc >/dev/null 2>&1 && "
            f"cvlc --play-and-exit --quiet {shlex.quote(remote_file)}; then "
            "exit 0; "
            "fi; "
            "if command -v mpg123 >/dev/null 2>&1 && "
            f"mpg123 -q -o alsa {shlex.quote(remote_file)}; then "
            "exit 0; "
            "fi; "
            "echo 'No usable audio player on Raspberry Pi (tried mpg123/ffplay/cvlc).'; "
            "exit 1"
        )
        subprocess.run(
            [*ssh_cmd, target, remote_cmd],
            check=True,
            timeout=timeout_seconds,
        )
        return True
    except Exception as e:
        print(f"(RPI playback failed: {e})")
        return False


def _play_mp3_blocking(path: str, timeout_seconds: int = 20) -> bool:
    """Play mp3 on Windows and block until playback completes."""
    resolved = str(Path(path).resolve()).replace("'", "''")

    # Preferred path: MCI playback via winmm (fully blocking, no external player process).
    alias = "codex_tts"
    open_cmd = f'open "{resolved}" type mpegvideo alias {alias}'
    play_cmd = f"play {alias} wait"
    close_cmd = f"close {alias}"

    try:
        if ctypes.windll.winmm.mciSendStringW(open_cmd, None, 0, None) == 0:
            try:
                if ctypes.windll.winmm.mciSendStringW(play_cmd, None, 0, None) == 0:
                    ctypes.windll.winmm.mciSendStringW(close_cmd, None, 0, None)
                    return True
            finally:
                ctypes.windll.winmm.mciSendStringW(close_cmd, None, 0, None)
    except Exception:
        pass

    # Fallback path: Windows Media Player COM.
    ps_script = (
        "$ErrorActionPreference = 'Stop'; "
        "$p = New-Object -ComObject WMPlayer.OCX; "
        f"$p.URL = '{resolved}'; "
        "$p.controls.play(); "
        "$sw = [Diagnostics.Stopwatch]::StartNew(); "
        f"while (($p.playState -ne 1) -and ($sw.Elapsed.TotalSeconds -lt {int(timeout_seconds)})) "
        "{ Start-Sleep -Milliseconds 150 }; "
        "$p.controls.stop(); "
        "$p.close(); "
        f"if ($sw.Elapsed.TotalSeconds -ge {int(timeout_seconds)}) {{ exit 2 }}"
    )
    result = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=False)
    if result.returncode == 0:
        return True

    # Fallback: use wmplayer CLI if COM playback fails on this machine.
    cli = f'start "" /wait wmplayer /play /close "{resolved}"'
    try:
        subprocess.run(["cmd", "/c", cli], check=False, timeout=timeout_seconds + 2)
        return True
    except Exception:
        return False


def _run_async(coro) -> None:
    try:
        asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(coro)
        finally:
            loop.close()


def speak_blocking(text: str) -> None:
    """Synthesize and play speech synchronously."""
    if not text:
        return
    if Communicate is None:
        print("(edge-tts unavailable) Reply:", text)
        return

    temp_name = f"response_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.mp3"
    temp_path = str(Path(tempfile.gettempdir()) / temp_name)

    try:
        set_assistant_activity("speaking")
        _run_async(speak_async(text, temp_path))
        if _is_rpi_output_enabled():
            played = _play_on_raspberry_pi(temp_path, text)
            if not played:
                print("(RPI playback unavailable) Falling back to local speaker playback.")
                played = _play_mp3_blocking(temp_path)
        else:
            played = _play_mp3_blocking(temp_path)
        if not played:
            print("(TTS playback failed) File saved at:", temp_path)
    except Exception as e:
        print(f"(TTS failed: {e}) Reply:", text)
        print("(TTS failed) Reply:", text)
    finally:
        set_assistant_activity("idle")
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


def speak(text: str) -> None:
    """Backwards-compatible wrapper."""
    speak_blocking(text)
