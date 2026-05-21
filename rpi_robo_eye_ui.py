import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse as urlparse
from urllib import request as urlrequest

import pygame

from eye_engine import EyeEngine


MOOD_FILE = Path("/tmp/llm_tts/mood.txt")
TTS_FLAG_FILE = Path("/tmp/llm_tts/tts_playing.flag")
ACTIVITY_FILE = Path("/tmp/llm_tts/assistant_state.txt")
ONBOARDING_STATE_FILE = Path("/tmp/llm_tts/onboarding_state.json")
ONBOARDING_QR_FILE = Path("/tmp/llm_tts/onboarding_qr.png")
BASE_WIDTH = 800
BASE_HEIGHT = 480
SAFE_MARGIN_X = 60
SAFE_MARGIN_Y = 44

try:
    import qrcode
except ModuleNotFoundError:
    qrcode = None


def _read_mood(default: str = "neutral") -> str:
    try:
        mood = MOOD_FILE.read_text(encoding="utf-8").strip().lower()
    except Exception:
        return default

    if mood in {"neutral", "happy", "sad", "surprised"}:
        return mood
    if mood in {"anger", "critical"}:
        return "sad"
    return default


def _is_tts_playing() -> bool:
    if not TTS_FLAG_FILE.exists():
        return False
    try:
        age_seconds = (datetime.now(timezone.utc) - datetime.fromtimestamp(TTS_FLAG_FILE.stat().st_mtime, timezone.utc)).total_seconds()
        # Ignore stale flags so onboarding overlay can appear if playback crashed.
        return age_seconds < 30
    except Exception:
        return True


def _read_activity(default: str = "idle") -> str:
    try:
        state = ACTIVITY_FILE.read_text(encoding="utf-8").strip().lower()
    except Exception:
        return default

    if state in {"idle", "listening", "thinking", "speaking"}:
        return state
    return default


def _read_onboarding_state() -> dict:
    try:
        raw = ONBOARDING_STATE_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception:
        # If QR file exists but state is missing, still show overlay.
        if ONBOARDING_QR_FILE.exists():
            return {"status": "waiting", "url": "", "message": "Scan QR and submit profile"}
        return {"status": "idle", "url": "", "message": ""}
    if not isinstance(data, dict):
        if ONBOARDING_QR_FILE.exists():
            return {"status": "waiting", "url": "", "message": "Scan QR and submit profile"}
        return {"status": "idle", "url": "", "message": ""}
    state = {
        "status": str(data.get("status", "idle")).lower(),
        "url": str(data.get("url", "")),
        "message": str(data.get("message", "")),
        "updated_at": str(data.get("updated_at", "")),
    }
    updated_at = state.get("updated_at", "")
    if updated_at:
        try:
            ts = datetime.fromisoformat(updated_at)
            age_seconds = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
            # Ignore stale onboarding overlay from previous sessions.
            if age_seconds > 1800:
                state["status"] = "idle"
        except Exception:
            pass
    return state


def _ensure_onboarding_qr(overlay: dict) -> bool:
    """Ensure a QR image exists on the Pi, even if SCP failed."""
    url = str(overlay.get("url", "")).strip()
    if not url:
        return False
    if ONBOARDING_QR_FILE.exists():
        return True

    ONBOARDING_QR_FILE.parent.mkdir(parents=True, exist_ok=True)

    if qrcode is not None:
        try:
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            img.save(ONBOARDING_QR_FILE)
            return True
        except Exception:
            pass

    try:
        encoded = urlparse.quote(url, safe="")
        qr_url = (
            "https://api.qrserver.com/v1/create-qr-code/"
            f"?size=420x420&ecc=H&qzone=4&format=png&color=000-000-000&bgcolor=255-255-255&data={encoded}"
        )
        with urlrequest.urlopen(qr_url, timeout=6) as resp:
            data = resp.read()
        ONBOARDING_QR_FILE.write_bytes(data)
        return True
    except Exception:
        return False


def _render_onboarding_overlay(screen: pygame.Surface, overlay: dict, qr_cache: dict) -> None:
    screen.fill((4, 10, 16))
    font_title = pygame.font.SysFont("dejavusans", 46, bold=True)
    font_body = pygame.font.SysFont("dejavusans", 27, bold=True)
    font_small = pygame.font.SysFont("dejavusans", 23)

    # Keep all overlay elements inside a padded "content safe" rectangle.
    safe = pygame.Rect(30, 30, BASE_WIDTH - 60, BASE_HEIGHT - 60)
    panel = safe.inflate(-20, -16)
    pygame.draw.rect(screen, (10, 21, 30), panel, border_radius=24)
    pygame.draw.rect(screen, (23, 56, 74), panel, width=2, border_radius=24)

    qr_top_left = (panel.x + 32, panel.y + 130)
    qr_max_dim = 218
    right_x = panel.x + 318
    right_width = panel.right - right_x - 24

    title = font_title.render("Scan To Continue", True, (228, 248, 255))
    screen.blit(title, (panel.x + 28, panel.y + 22))

    message = overlay.get("message", "Scan QR and submit profile")
    message_surface = font_body.render(message, True, (166, 222, 242))
    screen.blit(message_surface, (panel.x + 30, panel.y + 76))

    fingerprint = f"{overlay.get('url','')}|{overlay.get('updated_at','')}"
    if qr_cache.get("fingerprint") != fingerprint:
        qr_cache.clear()
        qr_cache["fingerprint"] = fingerprint

    qr_surface = qr_cache.get("surface")
    qr_rect = qr_cache.get("rect")
    if qr_surface is None:
        try:
            if _ensure_onboarding_qr(overlay) and ONBOARDING_QR_FILE.exists():
                raw_qr = pygame.image.load(str(ONBOARDING_QR_FILE)).convert()
                # Keep QR readable but cap size so it doesn't overwhelm the layout.
                w, h = raw_qr.get_size()
                if max(w, h) > qr_max_dim:
                    scale = qr_max_dim / max(w, h)
                    new_size = (int(w * scale), int(h * scale))
                    qr_surface = pygame.transform.smoothscale(raw_qr, new_size)
                else:
                    qr_surface = raw_qr
                qr_cache["surface"] = qr_surface
                qr_cache["rect"] = qr_surface.get_rect(topleft=qr_top_left)
        except Exception:
            qr_surface = None

    if qr_surface is not None and qr_rect is not None:
        panel = qr_rect.inflate(24, 24)
        pygame.draw.rect(screen, (255, 255, 255), panel, border_radius=10)
        screen.blit(qr_surface, qr_rect.topleft)
        tip = font_small.render("Use your phone camera", True, (124, 195, 224))
        tip_rect = tip.get_rect(midtop=(panel.centerx, panel.bottom + 14))
        tip_rect.x = max(12, tip_rect.x)
        tip_rect.y = min(BASE_HEIGHT - tip_rect.height - 8, tip_rect.y)
        screen.blit(tip, tip_rect.topleft)

    url = overlay.get("url", "")
    if url:
        def _wrap_for_width(text: str, max_width: int) -> list[str]:
            words = text.split()
            if not words:
                return [text]
            lines: list[str] = []
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                if font_small.size(candidate)[0] <= max_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
            # Fallback for long tokens without spaces (URLs).
            final: list[str] = []
            for line in lines:
                if font_small.size(line)[0] <= max_width:
                    final.append(line)
                    continue
                token = line
                chunk = ""
                for ch in token:
                    test = chunk + ch
                    if font_small.size(test)[0] <= max_width:
                        chunk = test
                    else:
                        if chunk:
                            final.append(chunk)
                        chunk = ch
                if chunk:
                    final.append(chunk)
            return final

        wrapped = _wrap_for_width(url, right_width)
        label_y = panel.y + 136
        if qr_rect is not None:
            label_y = max(140, qr_rect.top + 34)
        y = label_y + 38
        label = font_small.render("If QR scan fails, open this URL:", True, (180, 225, 240))
        screen.blit(label, (right_x, label_y))
        for line in wrapped[:7]:
            line_surface = font_small.render(line, True, (226, 244, 252))
            screen.blit(line_surface, (right_x, y))
            y += 28


def _safe_render_rect(display_width: int, display_height: int) -> pygame.Rect:
    """Inset UI from panel edges so bezel/cutouts do not clip content."""
    margin_x = min(SAFE_MARGIN_X, max(16, display_width // 10))
    margin_y = min(SAFE_MARGIN_Y, max(12, display_height // 10))
    rect = pygame.Rect(
        margin_x,
        margin_y,
        max(120, display_width - margin_x * 2),
        max(80, display_height - margin_y * 2),
    )
    if rect.width < 120 or rect.height < 80:
        return pygame.Rect(0, 0, display_width, display_height)
    return rect


def _present_canvas(screen: pygame.Surface, canvas: pygame.Surface) -> None:
    display_width, display_height = screen.get_size()
    screen.fill((0, 0, 0))

    safe_rect = _safe_render_rect(display_width, display_height)
    scale = min(safe_rect.width / BASE_WIDTH, safe_rect.height / BASE_HEIGHT)
    scaled_w = max(1, int(BASE_WIDTH * scale))
    scaled_h = max(1, int(BASE_HEIGHT * scale))
    scaled = pygame.transform.smoothscale(canvas, (scaled_w, scaled_h))

    x = safe_rect.x + (safe_rect.width - scaled_w) // 2
    y = safe_rect.y + (safe_rect.height - scaled_h) // 2
    screen.blit(scaled, (x, y))


def main() -> None:
    pygame.init()
    os.environ.setdefault("SDL_VIDEO_CENTERED", "0")
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.NOFRAME)
    display_width, display_height = screen.get_size()
    canvas = pygame.Surface((BASE_WIDTH, BASE_HEIGHT))
    pygame.display.set_caption("Robo Eye UI")
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    eyes = EyeEngine(canvas)
    assistant_state = "neutral"
    running = True
    qr_cache: dict = {}

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        activity_state = _read_activity(default="idle")
        mood_state = _read_mood(default=assistant_state)
        onboarding = _read_onboarding_state()
        tts_playing = _is_tts_playing()

        # During speech, keep expressive eye animation; show QR after TTS line completes.
        if not (tts_playing or activity_state == "speaking" or activity_state == "thinking"):
            if onboarding.get("status") in {"waiting", "timeout"}:
                _render_onboarding_overlay(canvas, onboarding, qr_cache)
                _present_canvas(screen, canvas)
                pygame.display.flip()
                continue
            qr_cache.clear()

        if activity_state == "thinking":
            assistant_state = "thinking"
        elif activity_state == "listening":
            assistant_state = "listening"
        elif activity_state == "speaking":
            assistant_state = "talking" if tts_playing else "thinking"
        elif tts_playing:
            assistant_state = "talking"
        else:
            assistant_state = mood_state

        eyes.set_state(assistant_state)
        eyes.update(dt, assistant_state == "talking")
        eyes.draw()
        _present_canvas(screen, canvas)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
