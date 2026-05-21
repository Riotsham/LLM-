from __future__ import annotations

import json
import os
import random
import re
import shlex
import socket
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

import numpy as np

try:
    import librosa
except ModuleNotFoundError:
    librosa = None
try:
    import qrcode
except ModuleNotFoundError:
    qrcode = None
try:
    import pywhatkit
except ModuleNotFoundError:
    pywhatkit = None

from audio.record import SAMPLE_RATE, model as whisper_model, record_chunk
from audio.tts import set_assistant_activity, speak_blocking
from database.mongo_db import create_user, get_user_by_name, log_pitch, save_full_session, update_user_by_id
from helpers import is_affirmative, is_negative
from llm.model import generate_llama_response, generate_response
from risk.risk_model import detect_risk
from risk.rules import apply_rules
from api.server import ensure_server_started
from utils.rpi_ssh import resolve_rpi_host, resolve_ssh_key_path

_WHATSAPP_LOG_PATH = Path(__file__).resolve().parent / "database" / "whatsapp_outbox.jsonl"
_ONBOARDING_STATE_FILE = Path(os.getenv("ASSISTANT_ONBOARDING_STATE_FILE", "/tmp/llm_tts/onboarding_state.json"))
_ONBOARDING_QR_FILE = Path(os.getenv("ASSISTANT_ONBOARDING_QR_FILE", "/tmp/llm_tts/onboarding_qr.png"))
_SESSION_CLOSE_MESSAGES = (
    "Thank you for sharing today. I will close the session now. Take care.",
    "Thanks for talking with me today. I am ending the session now. Please take good care of yourself.",
    "I appreciate you opening up today. I will wrap up this session now. Wishing you calm and strength.",
)
_SESSION_WRAPUP_PROMPTS = (
    "Can we wind up the session now? Please say yes or no.",
    "Would you like to end the session now? Please say yes or no.",
    "Shall we close this session here? Please say yes or no.",
)
_SESSION_WRAPUP_REPROMPTS = (
    "Please say yes if you want to end this session, or no if you want to continue.",
    "Please answer with yes to close now, or no to keep talking.",
    "Say yes to finish the session now, or no if you want to continue.",
)


def _pick_session_message(options: tuple[str, ...]) -> str:
    return random.choice(options)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_onboarding_state(status: str, url: str = "", message: str = "") -> None:
    payload = {"status": status, "url": url, "message": message, "updated_at": _now_iso()}
    try:
        _ONBOARDING_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _ONBOARDING_STATE_FILE.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    except Exception:
        pass


def _read_onboarding_public_base() -> str:
    configured = (os.getenv("ONBOARDING_PUBLIC_BASE_URL") or "").strip().rstrip("/")

    port = int(os.getenv("ONBOARDING_SERVER_PORT") or 8765)
    rpi_host = (os.getenv("RPI_HOST") or "").strip()
    rpi_prefix = ".".join(rpi_host.split(".")[:3]) if rpi_host and "." in rpi_host else ""

    candidates: list[str] = []
    preferred_lan_ip = ""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            preferred_lan_ip = s.getsockname()[0]
        if preferred_lan_ip and not preferred_lan_ip.startswith("127."):
            candidates.append(preferred_lan_ip)
    except Exception:
        pass

    host = socket.gethostname()
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
            if family != socket.AF_INET:
                continue
            ip = str(sockaddr[0]).strip()
            if not ip or ip.startswith("127."):
                continue
            candidates.append(ip)
    except Exception:
        pass

    # Deduplicate while preserving order.
    unique_candidates: list[str] = []
    seen: set[str] = set()
    for ip in candidates:
        if ip not in seen:
            seen.add(ip)
            unique_candidates.append(ip)

    if configured:
        parsed = urlparse.urlparse(configured if "://" in configured else f"http://{configured}")
        configured_host = (parsed.hostname or "").strip().lower()
        local_ips = {ip.lower() for ip in unique_candidates}
        rpi_host_lc = rpi_host.lower()

        def _is_ipv4_literal(host_value: str) -> bool:
            try:
                socket.inet_aton(host_value)
                return host_value.count(".") == 3
            except OSError:
                return False

        # Avoid generating phone QR URLs that point to loopback or a non-local Pi host.
        if configured_host not in {"localhost", "127.0.0.1"}:
            if configured_host in local_ips:
                return configured
            # Reject stale literal IPs that are no longer assigned to this laptop.
            if _is_ipv4_literal(configured_host):
                configured_host = ""
            else:
                try:
                    resolved_ips = {
                        str(sockaddr[0]).strip().lower()
                        for family, _, _, _, sockaddr in socket.getaddrinfo(configured_host, None)
                        if family == socket.AF_INET
                    }
                    if resolved_ips & local_ips:
                        return configured
                except Exception:
                    # Keep user-supplied DNS names when we cannot resolve here.
                    return configured
            if not (rpi_host_lc and configured_host == rpi_host_lc and rpi_host_lc not in local_ips):
                if configured_host:
                    return configured

    # Prefer IP in same /24 subnet as Raspberry Pi if available.
    if rpi_prefix:
        for ip in unique_candidates:
            if ip.startswith(f"{rpi_prefix}."):
                return f"http://{ip}:{port}"

    if unique_candidates:
        return f"http://{unique_candidates[0]}:{port}"

    return "http://127.0.0.1:8765"


def _json_request(url: str, method: str = "GET", payload: dict | None = None, timeout: float = 5.0) -> dict | None:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urlrequest.Request(url=url, data=data, method=method.upper(), headers=headers)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        return json.loads(raw) if raw else {}
    except (urlerror.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None


def _is_rpi_onboarding_enabled() -> bool:
    return os.getenv("TTS_OUTPUT_DEVICE", "").strip().lower() in {"rpi", "raspi", "raspberrypi", "remote"} or (
        os.getenv("RPI_TTS_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}
    )


def _push_onboarding_assets_to_rpi(local_qr_path: Path | None) -> None:
    if not _is_rpi_onboarding_enabled():
        return

    port = os.getenv("RPI_PORT", "22").strip() or "22"
    host = resolve_rpi_host(port=port)
    if not host:
        return

    user = os.getenv("RPI_USER", "pi").strip() or "pi"
    remote_dir = os.getenv("RPI_REMOTE_DIR", "/tmp/llm_tts").strip() or "/tmp/llm_tts"
    ssh_key_path = resolve_ssh_key_path(os.getenv("RPI_SSH_KEY", ""), cwd=Path(__file__).resolve().parent)
    target = f"{user}@{host}"
    ssh_cmd = ["ssh", "-p", port, "-o", "BatchMode=yes"]
    scp_cmd = ["scp", "-P", port, "-o", "BatchMode=yes"]
    if ssh_key_path and Path(ssh_key_path).exists():
        ssh_cmd.extend(["-i", ssh_key_path, "-o", "IdentitiesOnly=yes"])
        scp_cmd.extend(["-i", ssh_key_path, "-o", "IdentitiesOnly=yes"])

    try:
        subprocess.run([*ssh_cmd, target, f"mkdir -p {shlex.quote(remote_dir)}"], check=True, timeout=8)
        subprocess.run([*scp_cmd, str(_ONBOARDING_STATE_FILE.resolve()), f"{target}:{remote_dir}/onboarding_state.json"], check=True, timeout=12)
        if local_qr_path and local_qr_path.exists():
            subprocess.run([*scp_cmd, str(local_qr_path.resolve()), f"{target}:{remote_dir}/onboarding_qr.png"], check=True, timeout=12)
    except Exception:
        pass


def _clear_onboarding_files() -> None:
    _write_onboarding_state(status="idle", url="", message="")
    try:
        _ONBOARDING_QR_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    _push_onboarding_assets_to_rpi(local_qr_path=None)


def _download_qr_image(url: str) -> Path | None:
    if not url:
        return None
    if qrcode is not None:
        try:
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=12,
                border=4,
            )
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            _ONBOARDING_QR_FILE.parent.mkdir(parents=True, exist_ok=True)
            img.save(_ONBOARDING_QR_FILE)
            return _ONBOARDING_QR_FILE
        except Exception:
            pass

    encoded = urlparse.quote(url, safe="")
    qr_url = (
        "https://api.qrserver.com/v1/create-qr-code/"
        f"?size=420x420&ecc=H&qzone=4&format=png&color=000-000-000&bgcolor=255-255-255&data={encoded}"
    )
    try:
        with urlrequest.urlopen(qr_url, timeout=8) as resp:
            data = resp.read()
        _ONBOARDING_QR_FILE.parent.mkdir(parents=True, exist_ok=True)
        _ONBOARDING_QR_FILE.write_bytes(data)
        return _ONBOARDING_QR_FILE
    except Exception:
        return None


def _collect_profile_from_qr(timeout_seconds: int = 600) -> dict | None:
    try:
        _, local_port = ensure_server_started()
    except Exception:
        return None

    local_base_url = f"http://127.0.0.1:{local_port}"
    public_base_url = _read_onboarding_public_base()
    ttl_minutes = max(10, int(timeout_seconds / 60) + 5)
    session_doc = _json_request(
        f"{local_base_url}/api/onboarding/session",
        method="POST",
        payload={"ttl_minutes": ttl_minutes},
        timeout=5.0,
    )
    if not session_doc or not session_doc.get("token"):
        return None

    token = str(session_doc.get("token"))
    started_at_iso = _now_iso()
    onboarding_url = str(session_doc.get("onboarding_url", "")).strip()
    if onboarding_url.startswith("/"):
        onboarding_url = f"{public_base_url}{onboarding_url}"
    if not onboarding_url:
        onboarding_url = f"{public_base_url}/onboarding/{token}"
    # Use stable endpoint in QR so stale scans can still reach current active session.
    onboarding_url = f"{public_base_url}/onboarding/latest"

    qr_path = _download_qr_image(onboarding_url)
    _write_onboarding_state(status="waiting", url=onboarding_url, message="Scan QR and submit profile")
    _push_onboarding_assets_to_rpi(local_qr_path=qr_path)

    end_time = time.time() + max(30, timeout_seconds)
    status_url = f"{local_base_url}/api/onboarding/{token}/status"
    latest_completed_url = (
        f"{local_base_url}/api/onboarding/latest-completed?since={urlparse.quote(started_at_iso, safe='')}"
    )
    while time.time() < end_time:
        status_doc = _json_request(status_url, method="GET", payload=None, timeout=4.0)
        if status_doc and str(status_doc.get("status", "")).lower() == "completed":
            profile = status_doc.get("profile") or {}
            _write_onboarding_state(status="completed", url=onboarding_url, message="Profile received")
            _push_onboarding_assets_to_rpi(local_qr_path=qr_path)
            return profile if isinstance(profile, dict) else None

        # Recovery path: if user submitted a nearby form with a different token,
        # accept the most recent completed submission from this onboarding window.
        latest_doc = _json_request(latest_completed_url, method="GET", payload=None, timeout=4.0)
        if latest_doc and str(latest_doc.get("status", "")).lower() == "completed":
            profile = latest_doc.get("profile") or {}
            if isinstance(profile, dict) and str(profile.get("name", "")).strip():
                _write_onboarding_state(status="completed", url=onboarding_url, message="Profile received")
                _push_onboarding_assets_to_rpi(local_qr_path=qr_path)
                return profile
        time.sleep(2.0)

    _write_onboarding_state(status="timeout", url=onboarding_url, message="Timed out. Using voice onboarding")
    _push_onboarding_assets_to_rpi(local_qr_path=qr_path)
    return None


def listen() -> tuple[str, np.ndarray]:
    """Record one turn and transcribe with the existing local Whisper model."""
    set_assistant_activity("listening")
    try:
        audio_signal = record_chunk()
        segments, _ = whisper_model.transcribe(
            audio_signal,
            vad_filter=True,
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        user_text = " ".join(seg.text for seg in segments).strip()
        if not user_text:
            # Quiet speech can be filtered out by VAD; retry once without it.
            segments, _ = whisper_model.transcribe(
                audio_signal,
                vad_filter=False,
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=False,
            )
            user_text = " ".join(seg.text for seg in segments).strip()
        return user_text, audio_signal
    finally:
        set_assistant_activity("idle")


def _extract_avg_pitch(audio_signal: np.ndarray) -> float:
    if librosa is None or audio_signal is None:
        return 0.0
    y = np.asarray(audio_signal, dtype=np.float32).flatten()
    if y.size == 0:
        return 0.0
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    f0 = librosa.yin(y, fmin=50, fmax=300, sr=SAMPLE_RATE)
    valid = f0[np.isfinite(f0) & (f0 > 0)]
    return float(np.mean(valid)) if valid.size else 0.0


def _ask_and_capture(question: str) -> tuple[str, np.ndarray]:
    speak_blocking(question)
    return listen()


def _normalize_name(name_text: str) -> str:
    cleaned = " ".join((name_text or "").strip().split())
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    for prefix in ("my name is ", "this is ", "i am ", "i'm ", "im ", "call me "):
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break

    # Keep only name-like tokens and drop common filler words from ASR.
    words = re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)?", cleaned)
    filler = {"my", "name", "is", "i", "am", "im", "this", "me", "call", "called"}
    words = [w for w in words if w.lower() not in filler]
    if not words:
        return ""

    # Support first+last names while keeping existing first-name-only flows functional.
    selected = words[:2]
    return " ".join(part[:24].capitalize() for part in selected)


def _normalize_age(age_text: str) -> str:
    cleaned = " ".join((age_text or "").strip().split())
    if not cleaned:
        return ""
    match = re.search(r"\b(\d{1,3})\b", cleaned)
    if not match:
        return cleaned
    value = int(match.group(1))
    if 0 < value < 125:
        return str(value)
    return cleaned


def _normalize_spelled_name(name_text: str) -> str:
    cleaned = " ".join((name_text or "").strip().split())
    if not cleaned:
        return ""
    words = re.findall(r"[A-Za-z]+", cleaned)
    if not words:
        return ""

    # If user spells letter-by-letter ("p r a k a s h"), merge letters.
    if words and all(len(w) == 1 for w in words):
        merged = "".join(words)
        return merged[:24].capitalize() if merged else ""

    return _normalize_name(cleaned)


def _capture_confirmed_name(max_attempts: int = 3) -> tuple[str, list[float]]:
    pitch_values: list[float] = []

    for attempt in range(max_attempts):
        raw_name, name_audio = _ask_and_capture("What is your name?")
        pitch_values.append(_extract_avg_pitch(name_audio))
        name = _normalize_name(raw_name)
        if not name:
            speak_blocking("I did not catch the name clearly. Please say only your first name.")
            continue

        confirm_text, confirm_audio = _ask_and_capture(f"I heard {name}. Is that right? Please say yes or no.")
        pitch_values.append(_extract_avg_pitch(confirm_audio))
        if is_affirmative(confirm_text):
            return name, pitch_values
        if is_negative(confirm_text):
            if attempt < max_attempts - 1:
                speak_blocking("Okay, let's try again.")
            continue

        speak_blocking("I could not confirm that. I will ask your name once more.")

    # Voice fallback: ask user to spell their name slowly.
    spelled_raw, spelled_audio = _ask_and_capture(
        "I still could not catch your name. Please spell your first name letter by letter."
    )
    pitch_values.append(_extract_avg_pitch(spelled_audio))
    spelled_name = _normalize_spelled_name(spelled_raw)
    if spelled_name:
        confirm_text, confirm_audio = _ask_and_capture(
            f"I heard {spelled_name}. Is that right? Please say yes or no."
        )
        pitch_values.append(_extract_avg_pitch(confirm_audio))
        if is_affirmative(confirm_text):
            return spelled_name, pitch_values

    # Keyboard fallback for noisy environments and hard-to-capture names.
    speak_blocking("Please type your name on the keyboard so I can load it correctly.")
    try:
        typed_name = input("Type your name and press Enter: ").strip()
    except (EOFError, OSError):
        typed_name = ""
    typed_name = _normalize_name(typed_name)
    if typed_name:
        speak_blocking(f"Thanks {typed_name}.")
        return typed_name, pitch_values

    return "", pitch_values


def _fmt_profile_line(user: dict, key: str, default: str = "Unknown") -> str:
    value = user.get(key)
    if value is None or str(value).strip() == "":
        return default
    return str(value)


def _is_missing_profile_field(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"unknown", "n/a", "na", "none"}


def _is_session_end(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False
    end_markers = (
        "thank you",
        "thankyou",
        "thank you so much",
        "thanks",
        "thanks a lot",
        "thanks for your help",
        "i like to end up the conversation",
        "i want to end up the conversation",
        "i need to end up the conversation",
        "i like to wind up the session",
        "i want to wind up the session",
        "can we wind up the session",
        "please end the session",
        "please close the session",
        "end the conversation",
        "close the conversation",
        "end the session",
        "close the session",
        "that is all",
        "goodbye",
        "bye",
        "we are done",
    )
    return any(normalized == marker or normalized.startswith(f"{marker} ") for marker in end_markers)


def _is_explicit_windup_request(text: str) -> bool:
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized:
        return False
    explicit_markers = (
        "wind up",
        "windup",
        "end session",
        "close session",
        "stop session",
        "finish session",
        "we can stop",
        "let's stop",
        "lets stop",
        "bye",
        "goodbye",
    )
    return any(marker in normalized for marker in explicit_markers)


def _finalize_session(
    user_id,
    conversation_history: list[tuple[str, str]],
    session_pitch_values: list[float],
    latest_overall_risk: float,
    latest_primary_problem: str,
    latest_display_mood: str,
    session_max_overall_risk: float,
    session_seen_crisis: bool,
) -> None:
    metadata = {
        "overall_risk_score": latest_overall_risk,
        "primary_problem": latest_primary_problem,
        "display_mood": latest_display_mood,
        "session_max_overall_risk": session_max_overall_risk,
        "session_seen_crisis": session_seen_crisis,
    }
    save_full_session(user_id, conversation_history, session_pitch_values, metadata=metadata)
    update_user_by_id(
        user_id,
        {
            "last_session_at": _now_iso(),
            "last_overall_risk_score": latest_overall_risk,
            "last_primary_problem": latest_primary_problem,
            "last_display_mood": latest_display_mood,
            "last_session_max_overall_risk": session_max_overall_risk,
            "last_session_seen_crisis": session_seen_crisis,
        },
    )


def _set_rpi_mood(mood: str) -> None:
    normalized = (mood or "").strip().lower()
    if normalized not in {"happy", "sad", "anger"}:
        normalized = "sad"
    os.environ["RPI_MOOD"] = normalized


def _derive_display_mood(user_text: str, risk_state: str, risk_score: float) -> str:
    if (risk_state or "").upper() == "CRISIS" or float(risk_score or 0.0) >= 0.75:
        return "anger"

    text = (user_text or "").strip()
    if not text:
        return "sad"

    lowered = text.lower()
    positive_words = (
        "happy",
        "good",
        "great",
        "better",
        "fine",
        "okay",
        "calm",
        "excited",
        "relaxed",
        "grateful",
    )
    severe_words = (
        "suicide",
        "kill myself",
        "end my life",
        "die",
        "can't go on",
        "panic attack",
        "hopeless",
    )
    if any(w in lowered for w in severe_words):
        return "anger"
    if any(w in lowered for w in positive_words):
        return "happy"
    return "sad"


def _derive_session_display_mood(
    latest_user_text: str,
    latest_risk_state: str,
    latest_turn_risk: float,
    session_max_overall_risk: float,
    session_seen_crisis: bool,
    primary_problem: str,
) -> str:
    problem = (primary_problem or "").strip().lower()
    if (
        session_seen_crisis
        or session_max_overall_risk >= 0.66
        or "suicidal" in problem
        or "suicide" in problem
    ):
        return "anger"
    return _derive_display_mood(latest_user_text, latest_risk_state, latest_turn_risk)


def _append_whatsapp_log(record: dict) -> None:
    _WHATSAPP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _WHATSAPP_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _send_whatsapp_to_doctor(message: str) -> tuple[bool, str]:
    doctor_number = (os.getenv("DOCTOR_WHATSAPP_NUMBER") or "").strip()
    if not doctor_number:
        reason = "DOCTOR_WHATSAPP_NUMBER is not set."
        _append_whatsapp_log({"timestamp": _now_iso(), "status": "failed", "reason": reason, "message": message})
        return False, reason

    if pywhatkit is None:
        reason = "pywhatkit is not installed."
        _append_whatsapp_log({"timestamp": _now_iso(), "status": "failed", "reason": reason, "message": message})
        return False, reason

    try:
        # Uses WhatsApp Web in the default browser. Number must be in +<countrycode><number> format.
        pywhatkit.sendwhatmsg_instantly(
            phone_no=doctor_number,
            message=message,
            wait_time=15,
            tab_close=True,
            close_time=5,
        )
        _append_whatsapp_log({"timestamp": _now_iso(), "status": "sent", "message": message})
        return True, "sent"
    except Exception as exc:
        reason = f"{exc.__class__.__name__}: {exc}"
        _append_whatsapp_log({"timestamp": _now_iso(), "status": "failed", "reason": reason, "message": message})
        return False, reason


def _compute_overall_risk(risk_scores: list[float]) -> float:
    valid = [float(s) for s in risk_scores if isinstance(s, (int, float))]
    if not valid:
        return 0.0
    max_score = max(valid)
    avg_score = sum(valid) / len(valid)
    return min(1.0, (0.6 * max_score) + (0.4 * avg_score))


def _infer_primary_problem(user_messages: list[str]) -> str:
    text = " ".join((m or "").lower() for m in user_messages)
    buckets: list[tuple[str, tuple[str, ...]]] = [
        ("suicidal thoughts", ("suicide", "suicidal", "kill myself", "end my life", "want to die", "no reason to live")),
        ("depression symptoms", ("depress", "hopeless", "worthless", "empty", "numb")),
        ("anxiety and panic", ("anxious", "anxiety", "panic", "overwhelmed", "fear")),
        ("stress and burnout", ("stress", "burnout", "pressure", "exhausted")),
        ("relationship distress", ("relationship", "partner", "breakup", "family", "friend")),
        ("work or academic pressure", ("work", "job", "office", "exam", "study", "college", "school")),
        ("sleep disturbance", ("sleep", "insomnia", "nightmare", "can't sleep")),
    ]
    best_problem = "general emotional distress"
    best_hits = 0
    for problem, keywords in buckets:
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits > best_hits:
            best_hits = hits
            best_problem = problem
    return best_problem


def _build_doctor_message(user: dict, overall_risk: float, primary_problem: str, conversation_history: list[tuple[str, str]]) -> str:
    recent_user_lines = [text for role, text in conversation_history if role == "user" and str(text).strip()][-3:]
    excerpt = " | ".join(recent_user_lines) if recent_user_lines else "No user excerpt available."
    age_value = _fmt_profile_line(user, "age")
    if _is_missing_profile_field(age_value):
        for line in reversed(recent_user_lines):
            inferred = _normalize_age(line)
            if inferred.isdigit():
                age_value = inferred
                break
    return (
        "Mental Support Assistant Alert\n"
        f"Patient: {_fmt_profile_line(user, 'name')}\n"
        f"Age: {age_value}\n"
        f"Occupation: {_fmt_profile_line(user, 'occupation')}\n"
        f"Primary problem: {primary_problem}\n"
        f"Overall risk score: {overall_risk:.2f}\n"
        f"Time(UTC): {_now_iso()}\n"
        f"Recent statements: {excerpt}\n"
        "Action: Patient requested professional appointment booking."
    )


def _run_onboarding(prefilled_name: str | None = None) -> dict:
    _set_rpi_mood("happy")
    speak_blocking("Hello. I am your Mental support assistant. I would like to get to know you.")

    onboarding_pitch_values: list[float] = []

    if prefilled_name and prefilled_name.strip():
        name = prefilled_name.strip()
    else:
        name, name_pitch_values = _capture_confirmed_name()
        onboarding_pitch_values.extend(name_pitch_values)
        if not name:
            name = f"Guest{datetime.now().strftime('%H%M%S')}"
            speak_blocking(f"I will use temporary name {name}. We can update it later.")

    age, age_audio = _ask_and_capture("How old are you?")
    onboarding_pitch_values.append(_extract_avg_pitch(age_audio))

    occupation, occupation_audio = _ask_and_capture("What is your occupation?")
    onboarding_pitch_values.append(_extract_avg_pitch(occupation_audio))

    field_of_study = None
    if "student" in (occupation or "").strip().lower():
        field_of_study, field_audio = _ask_and_capture("What are you studying?")
        onboarding_pitch_values.append(_extract_avg_pitch(field_audio))

    week_text, week_audio = _ask_and_capture("How was your week?")
    onboarding_pitch_values.append(_extract_avg_pitch(week_audio))

    week_prompt = (
        "You are a friendly Mental support assistant. "
        "Respond meaningfully to the user's weekly update.\n"
        f"User name: {(name or '').strip() or 'User'}\n"
        f"User weekly update: {(week_text or '').strip()}\n"
        "Keep response short, natural, and empathetic."
    )
    week_reply = generate_llama_response(week_prompt)
    if week_reply:
        speak_blocking(week_reply)

    valid_baseline = [v for v in onboarding_pitch_values if isinstance(v, (int, float)) and v > 0]
    pitch_baseline = float(sum(valid_baseline) / len(valid_baseline)) if valid_baseline else 0.0

    user_doc = {
        "name": (name or "").strip() or "User",
        "age": _normalize_age(age),
        "occupation": (occupation or "").strip(),
        "pitch_baseline": pitch_baseline,
        "created_at": _now_iso(),
        "how_was_your_week": (week_text or "").strip(),
        "onboarding_complete": True,
    }
    if field_of_study and field_of_study.strip():
        user_doc["field_of_study"] = field_of_study.strip()

    new_user = create_user(user_doc)
    speak_blocking(f"Hi {new_user.get('name', 'there')}, how was your day?")
    return new_user


def _create_or_update_user_from_profile(profile: dict) -> dict | None:
    if not isinstance(profile, dict):
        return None
    raw_name = str(profile.get("name", "")).strip()
    name = _normalize_name(raw_name)
    if not name:
        return None

    existing = get_user_by_name(name)
    if not existing and " " in name:
        existing = get_user_by_name(name.split(" ", 1)[0])

    age = _normalize_age(str(profile.get("age", "")).strip())
    occupation = str(profile.get("occupation", "")).strip()
    field_of_study = str(profile.get("field_of_study", "")).strip()
    notes = str(profile.get("notes", "")).strip()

    if existing:
        updates: dict[str, str] = {}
        if age and _is_missing_profile_field(existing.get("age")):
            updates["age"] = age
        if occupation and _is_missing_profile_field(existing.get("occupation")):
            updates["occupation"] = occupation
        if field_of_study and _is_missing_profile_field(existing.get("field_of_study")):
            updates["field_of_study"] = field_of_study
        if notes:
            updates["intake_notes"] = notes
        if updates:
            update_user_by_id(existing["_id"], updates)
            existing.update(updates)
        return existing

    user_doc = {
        "name": name,
        "age": age,
        "occupation": occupation,
        "field_of_study": field_of_study,
        "intake_notes": notes,
        "pitch_baseline": 0.0,
        "created_at": _now_iso(),
        "onboarding_complete": True,
    }
    return create_user(user_doc)


def identify_or_onboard() -> dict:
    _set_rpi_mood("happy")
    speak_blocking("Welcome. Please scan the QR code on the display to share your basic details.")
    profile = _collect_profile_from_qr(timeout_seconds=900)
    if profile:
        user = _create_or_update_user_from_profile(profile)
        if user:
            _clear_onboarding_files()
            speak_blocking(f"Thanks {_fmt_profile_line(user, 'name', 'there')}. Profile received.")
            return user

    _clear_onboarding_files()
    speak_blocking("I could not get details from QR. Please tell me your first name so I can load your profile.")
    name, _ = _capture_confirmed_name(max_attempts=2)
    if not name:
        speak_blocking("I could not capture your name clearly. I will create a new profile and we can update your name later.")
        return _run_onboarding(prefilled_name=None)

    existing = get_user_by_name(name)
    if not existing and " " in name:
        existing = get_user_by_name(name.split(" ", 1)[0])
    if existing:
        updates: dict[str, str] = {}
        if _is_missing_profile_field(existing.get("age")):
            age_text, _ = _ask_and_capture("I do not have your age yet. How old are you?")
            normalized_age = _normalize_age(age_text)
            if normalized_age:
                updates["age"] = normalized_age
        if _is_missing_profile_field(existing.get("occupation")):
            occupation_text, _ = _ask_and_capture("I do not have your occupation yet. What do you do?")
            if (occupation_text or "").strip():
                updates["occupation"] = occupation_text.strip()
        if updates:
            update_user_by_id(existing["_id"], updates)
            existing.update(updates)
        speak_blocking(f"Welcome back, {_fmt_profile_line(existing, 'name', 'there')}.")
        return existing
    speak_blocking("I do not have your profile yet. I will create one now.")
    return _run_onboarding(prefilled_name=name)


def build_prompt(user: dict, conversation_history: list[tuple[str, str]], session_pitch_values: list[float]) -> str:
    baseline = float(user.get("pitch_baseline") or 0.0)
    latest_pitch = float(session_pitch_values[-1]) if session_pitch_values else 0.0

    if baseline > 0 and latest_pitch > 0:
        ratio = latest_pitch / baseline
        if ratio >= 1.15:
            pitch_comparison = "current pitch is higher than baseline"
        elif ratio <= 0.85:
            pitch_comparison = "current pitch is lower than baseline"
        else:
            pitch_comparison = "current pitch is near baseline"
    else:
        pitch_comparison = "current pitch comparison unavailable"

    recent_turns = conversation_history[-5:]
    if recent_turns:
        history_text = "\n".join([f"{role}: {text}" for role, text in recent_turns if str(text).strip()])
    else:
        history_text = "No prior turns yet. Start naturally and invite the user to speak."

    occupation = str(user.get("occupation", "")).strip()
    student_hint = (
        "If occupation is student, relate advice to their studies."
        if "student" in occupation.lower()
        else "No student-specific guidance needed."
    )

    return (
        "You are a Mental support assistant powered by Llama.\n\n"
        "User Profile:\n"
        f"- Name: {_fmt_profile_line(user, 'name')}\n"
        f"- Age: {_fmt_profile_line(user, 'age')}\n"
        f"- Occupation: {_fmt_profile_line(user, 'occupation')}\n"
        f"- Field of study: {_fmt_profile_line(user, 'field_of_study', 'N/A')}\n"
        f"- Pitch baseline: {baseline:.2f}\n"
        f"- Current pitch comparison: {pitch_comparison}\n\n"
        "Conversation History (last 5 turns):\n"
        f"{history_text}\n\n"
        "Instructions:\n"
        "- Personalize using stored name.\n"
        f"- {student_hint}\n"
        "- Use pitch comparison only as tone guidance.\n"
        "- Do not generate random filler responses.\n"
        "- Always respond meaningfully to the user's last message.\n"
        "- Continue conversation unless user says thank you."
    )


def _run_initial_checkin(
    user: dict,
    conversation_history: list[tuple[str, str]],
    session_pitch_values: list[float],
    risk_scores: list[float],
    user_problem_texts: list[str],
) -> None:
    prompts = (
        "How was your day today?",
        "How are you feeling right now?",
    )
    for question in prompts:
        answer_text, answer_audio = _ask_and_capture(question)
        if not (answer_text or "").strip():
            continue
        conversation_history.append(("assistant", question))
        conversation_history.append(("user", answer_text))
        user_problem_texts.append(answer_text)

        pitch = _extract_avg_pitch(answer_audio)
        session_pitch_values.append(pitch)
        log_pitch(user["_id"], pitch)
        risk_scores.append(detect_risk(answer_text))

    valid_pitches = [p for p in session_pitch_values if isinstance(p, (int, float)) and p > 0]
    if valid_pitches:
        baseline = float(sum(valid_pitches) / len(valid_pitches))
        user["pitch_baseline"] = baseline
        update_user_by_id(user["_id"], {"pitch_baseline": baseline})


def run_session():
    user = identify_or_onboard()

    conversation_history: list[tuple[str, str]] = []
    session_pitch_values: list[float] = []
    risk_scores: list[float] = []
    user_problem_texts: list[str] = []
    pending_wrapup_confirmation = False
    pending_appointment_confirmation = False
    appointment_offer_sent = False
    latest_overall_risk = 0.0
    latest_primary_problem = "general emotional distress"
    latest_display_mood = "happy"
    latest_turn_text = ""
    latest_turn_risk_score = 0.0
    latest_risk_state = "SAFE"
    session_max_overall_risk = 0.0
    session_seen_crisis = False

    _run_initial_checkin(
        user=user,
        conversation_history=conversation_history,
        session_pitch_values=session_pitch_values,
        risk_scores=risk_scores,
        user_problem_texts=user_problem_texts,
    )

    opening = (
        f"Hi {_fmt_profile_line(user, 'name', 'there')}. "
        "I'm listening. Tell me what's on your mind."
    )
    _set_rpi_mood("happy")
    speak_blocking(opening)
    conversation_history.append(("assistant", opening))

    while True:
        user_text, audio_signal = listen()

        if not (user_text or "").strip():
            continue

        if pending_wrapup_confirmation:
            if is_affirmative(user_text):
                close_msg = _pick_session_message(_SESSION_CLOSE_MESSAGES)
                speak_blocking(close_msg)
                conversation_history.append(("assistant", close_msg))
                _finalize_session(
                    user_id=user["_id"],
                    conversation_history=conversation_history,
                    session_pitch_values=session_pitch_values,
                    latest_overall_risk=latest_overall_risk,
                    latest_primary_problem=latest_primary_problem,
                    latest_display_mood=latest_display_mood,
                    session_max_overall_risk=session_max_overall_risk,
                    session_seen_crisis=session_seen_crisis,
                )
                break
            if is_negative(user_text):
                pending_wrapup_confirmation = False
                speak_blocking("Okay, we can continue. Tell me what else you want to discuss.")
                continue
            speak_blocking(_pick_session_message(_SESSION_WRAPUP_REPROMPTS))
            continue

        if pending_appointment_confirmation:
            if is_affirmative(user_text):
                doctor_message = _build_doctor_message(
                    user=user,
                    overall_risk=latest_overall_risk,
                    primary_problem=latest_primary_problem,
                    conversation_history=conversation_history,
                )
                sent, reason = _send_whatsapp_to_doctor(doctor_message)
                if sent:
                    speak_blocking("I have sent your details and appointment request to the doctor on WhatsApp.")
                else:
                    speak_blocking("I could not send WhatsApp right now, but your request has been logged for follow-up.")
                conversation_history.append(("assistant", f"Doctor booking request status: {'sent' if sent else reason}"))
                pending_appointment_confirmation = False
                continue
            if is_negative(user_text):
                pending_appointment_confirmation = False
                speak_blocking("Okay. We will continue here, and I am with you.")
                continue
            speak_blocking("Please say yes if you want me to book that appointment, or no to continue here.")
            continue

        if _is_explicit_windup_request(user_text):
            close_msg = _pick_session_message(_SESSION_CLOSE_MESSAGES)
            speak_blocking(close_msg)
            conversation_history.append(("assistant", close_msg))
            _finalize_session(
                user_id=user["_id"],
                conversation_history=conversation_history,
                session_pitch_values=session_pitch_values,
                latest_overall_risk=latest_overall_risk,
                latest_primary_problem=latest_primary_problem,
                latest_display_mood=latest_display_mood,
                session_max_overall_risk=session_max_overall_risk,
                session_seen_crisis=session_seen_crisis,
            )
            break

        if _is_session_end(user_text):
            close_msg = _pick_session_message(_SESSION_CLOSE_MESSAGES)
            speak_blocking(close_msg)
            conversation_history.append(("assistant", close_msg))
            _finalize_session(
                user_id=user["_id"],
                conversation_history=conversation_history,
                session_pitch_values=session_pitch_values,
                latest_overall_risk=latest_overall_risk,
                latest_primary_problem=latest_primary_problem,
                latest_display_mood=latest_display_mood,
                session_max_overall_risk=session_max_overall_risk,
                session_seen_crisis=session_seen_crisis,
            )
            break

        conversation_history.append(("user", user_text))
        user_problem_texts.append(user_text)
        latest_turn_text = user_text

        pitch_vals = _extract_avg_pitch(audio_signal)
        session_pitch_values.append(pitch_vals)
        log_pitch(user["_id"], pitch_vals)
        turn_risk_score = detect_risk(user_text)
        latest_turn_risk_score = turn_risk_score
        risk_scores.append(turn_risk_score)
        latest_overall_risk = _compute_overall_risk(risk_scores)
        session_max_overall_risk = max(session_max_overall_risk, latest_overall_risk)
        latest_primary_problem = _infer_primary_problem(user_problem_texts)
        risk_state = apply_rules(turn_risk_score)
        latest_risk_state = risk_state
        if risk_state == "CRISIS":
            session_seen_crisis = True
        latest_display_mood = _derive_session_display_mood(
            latest_user_text=latest_turn_text,
            latest_risk_state=latest_risk_state,
            latest_turn_risk=latest_turn_risk_score,
            session_max_overall_risk=session_max_overall_risk,
            session_seen_crisis=session_seen_crisis,
            primary_problem=latest_primary_problem,
        )
        _set_rpi_mood(latest_display_mood)
        update_user_by_id(
            user["_id"],
            {
                "last_session_at": _now_iso(),
                "last_overall_risk_score": latest_overall_risk,
                "last_primary_problem": latest_primary_problem,
                "last_display_mood": latest_display_mood,
                "last_session_max_overall_risk": session_max_overall_risk,
                "last_session_seen_crisis": session_seen_crisis,
            },
        )

        context = [f"{role}: {text}" for role, text in conversation_history if str(text).strip()]
        set_assistant_activity("thinking")
        try:
            llama_response = generate_response(
                user_text=user_text,
                context=context,
                audio_signal=audio_signal,
                sample_rate=SAMPLE_RATE,
                user_profile=user,
            )
        finally:
            set_assistant_activity("idle")

        speak_blocking(llama_response)
        conversation_history.append(("assistant", llama_response))

        if risk_state == "CRISIS" and not appointment_offer_sent:
            appointment_offer_sent = True
            pending_appointment_confirmation = True
            if not re.search(r"\b(appointment|book|doctor)\b", (llama_response or "").lower()):
                offer_text = "Would you like me to book an urgent appointment with our mental health experts?"
                speak_blocking(offer_text)
                conversation_history.append(("assistant", offer_text))
