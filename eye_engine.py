import math
import random

import pygame


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class EyeEngine:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.sw, self.sh = self.screen.get_size()

        self.base_w = 150.0
        self.base_h = 86.0
        self.base_gap = 80.0
        self.base_radius = 40

        self.current_state = "neutral"
        self.target_state = "neutral"

        self.current = {
            "w": 1.0,
            "h": 1.0,
            "tilt": 0.0,
            "arc": 0.0,
            "bright": 1.0,
        }
        self.target = dict(self.current)

        self.state_profiles = {
            "neutral": {"w": 1.00, "h": 1.00, "tilt": 0.0, "arc": 0.0, "bright": 1.00},
            "happy": {"w": 1.08, "h": 0.58, "tilt": 0.0, "arc": 1.0, "bright": 1.15},
            "sad": {"w": 0.96, "h": 0.90, "tilt": -7.0, "arc": 0.0, "bright": 0.90},
            "surprised": {"w": 1.22, "h": 1.12, "tilt": 0.0, "arc": 0.0, "bright": 1.20},
            "listening": {"w": 1.06, "h": 1.08, "tilt": 0.0, "arc": 0.0, "bright": 1.18},
            "thinking": {"w": 0.94, "h": 0.82, "tilt": 3.0, "arc": 0.0, "bright": 0.86},
            "talking": {"w": 1.03, "h": 0.98, "tilt": 0.0, "arc": 0.0, "bright": 1.08},
        }

        self.time = 0.0
        self.talk_phase = 0.0
        self.talk_amount = 0.0
        self.idle_phase = 0.0
        self.idle_jitter = 0.0
        self.mouth_open = 0.0

        self.blink_duration = 0.20
        self.blink_t = 0.0
        self.is_blinking = False
        self.next_blink_in = random.uniform(3.0, 6.0)

        self._thinking_symbols = ["?", "!", "?!", "!?"]
        self._left_think = random.choice(self._thinking_symbols)
        self._right_think = random.choice(self._thinking_symbols)
        self._think_swap_in = random.uniform(0.25, 0.7)
        self._think_jitter = 0.0
        self._bubble_bob = 0.0

    def set_state(self, state_name: str):
        if state_name == "blink":
            self.is_blinking = True
            self.blink_t = 0.0
            self.next_blink_in = random.uniform(3.0, 6.0)
            return

        if state_name not in self.state_profiles:
            state_name = "neutral"

        self.target_state = state_name
        self.target = dict(self.state_profiles[state_name])

    def update(self, dt: float, is_talking: bool):
        self.time += dt
        self.idle_phase += dt * 0.8
        self.idle_jitter = _lerp(self.idle_jitter, random.uniform(-1.0, 1.0), 1.0 - math.exp(-2.0 * dt))

        # Smooth state transition.
        blend = 1.0 - math.exp(-8.0 * dt)
        for k in self.current.keys():
            self.current[k] = _lerp(self.current[k], self.target[k], blend)
        self.current_state = self.target_state

        # Random blink every 3-6s; blink lasts 200ms.
        self.next_blink_in -= dt
        if self.next_blink_in <= 0.0 and not self.is_blinking:
            self.is_blinking = True
            self.blink_t = 0.0
            self.next_blink_in = random.uniform(3.0, 6.0)

        if self.is_blinking:
            self.blink_t += dt
            if self.blink_t >= self.blink_duration:
                self.blink_t = 0.0
                self.is_blinking = False

        # Talking pulse only while TTS is active.
        if is_talking:
            self.talk_phase += dt * 8.0
            self.talk_amount = _lerp(self.talk_amount, 1.0, 1.0 - math.exp(-10.0 * dt))
            self.mouth_open = _lerp(self.mouth_open, 1.0, 1.0 - math.exp(-14.0 * dt))
        else:
            self.talk_phase += dt * 2.5
            self.talk_amount = _lerp(self.talk_amount, 0.0, 1.0 - math.exp(-8.0 * dt))
            self.mouth_open = _lerp(self.mouth_open, 0.0, 1.0 - math.exp(-12.0 * dt))

        if self.target_state == "thinking":
            self._think_swap_in -= dt
            self._think_jitter = _lerp(self._think_jitter, random.uniform(-1.2, 1.2), 1.0 - math.exp(-4.0 * dt))
            if self._think_swap_in <= 0.0:
                self._left_think = random.choice(self._thinking_symbols)
                self._right_think = random.choice(self._thinking_symbols)
                self._think_swap_in = random.uniform(0.25, 0.7)
        self._bubble_bob += dt * 2.6

    def _blink_open_scale(self) -> float:
        if not self.is_blinking:
            return 1.0
        # 0 -> 1 -> 0 curve over blink duration.
        p = self.blink_t / self.blink_duration
        curve = math.sin(math.pi * p)
        return 1.0 - 0.92 * curve

    def _draw_eye(self, cx: float, cy: float, w: float, h: float, tilt_deg: float, arcness: float, brightness: float):
        pad = 52
        surf_w = int(w + pad * 2)
        surf_h = int(max(h * 2.0, h + pad * 2))
        eye_surf = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)

        ex = surf_w // 2 - int(w // 2)
        ey = surf_h // 2 - int(h // 2)
        rect = pygame.Rect(ex, ey, int(w), int(h))

        # Glow layers.
        glow_rgb = (0, 245, 255)
        for i in range(5, 0, -1):
            inflate = i * 10
            alpha = int(16 * i * max(0.75, brightness))
            r = rect.inflate(inflate, inflate)
            rr = min(self.base_radius + i * 3, max(8, r.h // 2))
            pygame.draw.rect(eye_surf, (*glow_rgb, alpha), r, border_radius=rr)

        # Main shape without iris/pupil.
        main_rgb = (
            min(255, int(0 * brightness)),
            min(255, int(245 * brightness)),
            min(255, int(255 * brightness)),
        )

        if arcness < 0.98:
            rr = min(self.base_radius, max(8, rect.h // 2))
            alpha = int(255 * (1.0 - 0.35 * arcness))
            pygame.draw.rect(eye_surf, (*main_rgb, alpha), rect, border_radius=rr)

        # Blend toward smile-eye arc for happy.
        if arcness > 0.02:
            arc_rect = pygame.Rect(
                ex,
                int(ey - h * 0.55),
                int(w),
                int(h * 1.8),
            )
            thick = max(6, int(h * (0.25 + 0.45 * arcness)))
            arc_alpha = int(220 * arcness)
            pygame.draw.arc(
                eye_surf,
                (main_rgb[0], main_rgb[1], main_rgb[2], arc_alpha),
                arc_rect,
                math.radians(195),
                math.radians(345),
                thick,
            )

        rotated = pygame.transform.rotozoom(eye_surf, tilt_deg, 1.0)
        rx, ry = rotated.get_size()
        self.screen.blit(rotated, (cx - rx / 2, cy - ry / 2))

    def _draw_mouth(self, cx: float, cy: float):
        base_y = cy + 95
        mouth_w = 140
        # Strong visible lip-sync style movement while speaking.
        talking_wave = math.sin(self.talk_phase * 1.2)
        talking_wave_2 = math.sin(self.talk_phase * 2.2 + 0.6)
        open_amt = self.mouth_open * (0.25 + 0.75 * abs(0.7 * talking_wave + 0.3 * talking_wave_2))
        mouth_h = int(8 + 42 * open_amt)
        mouth_w = int(mouth_w * (0.92 + 0.14 * abs(talking_wave)))

        # Glow + main tone tied to eye brightness.
        glow_color = (0, 245, 255, 70)
        line_color = (0, 245, 255, 220)

        # While speaking, always use animated open/close mouth so speech is obvious.
        if self.mouth_open > 0.06:
            rect = pygame.Rect(int(cx - mouth_w // 2), int(base_y - 4), mouth_w, max(10, mouth_h))
            for i in range(4, 0, -1):
                r = rect.inflate(i * 12, i * 10)
                pygame.draw.ellipse(self.screen, glow_color, r)

            pygame.draw.ellipse(self.screen, line_color, rect)

            return

        if self.target_state == "happy":
            arc_rect = pygame.Rect(int(cx - mouth_w // 2), int(base_y - 18), mouth_w, 52)
            pygame.draw.arc(self.screen, (0, 245, 255), arc_rect, math.radians(200), math.radians(340), 6)
        elif self.target_state == "sad":
            arc_rect = pygame.Rect(int(cx - mouth_w // 2), int(base_y - 2), mouth_w, 52)
            pygame.draw.arc(self.screen, (0, 220, 235), arc_rect, math.radians(20), math.radians(160), 6)
        elif self.target_state == "surprised":
            r = int(14 + 12 * max(0.1, open_amt))
            pygame.draw.circle(self.screen, (0, 245, 255, 120), (int(cx), int(base_y + 8)), r + 8)
            pygame.draw.circle(self.screen, (0, 245, 255), (int(cx), int(base_y + 8)), r, width=4)
        else:
            # Neutral/talking mouth: rounded glowing capsule that opens while speaking.
            rect = pygame.Rect(int(cx - mouth_w // 2), int(base_y), mouth_w, max(8, mouth_h))
            for i in range(3, 0, -1):
                r = rect.inflate(i * 10, i * 8)
                rr = min(24, max(6, r.h // 2))
                pygame.draw.rect(self.screen, glow_color, r, border_radius=rr)
            rr = min(22, max(6, rect.h // 2))
            pygame.draw.rect(self.screen, line_color, rect, border_radius=rr, width=0)
            # Inner dark cut gives clearer open/close perception.
            if self.mouth_open > 0.05:
                inner_h = max(2, int(rect.h * 0.45))
                inner = pygame.Rect(rect.x + 8, rect.y + rect.h // 2 - inner_h // 2, max(10, rect.w - 16), inner_h)
                pygame.draw.rect(self.screen, (2, 12, 14), inner, border_radius=max(3, inner_h // 2))

    def _draw_thinking_marks(self, left_cx: float, right_cx: float, cy: float, h: float) -> None:
        font = pygame.font.SysFont("arial", 34, bold=True)
        wave = math.sin(self.time * 4.0)
        base_y = cy - h * 0.95 - 36 + wave * 3.0 + self._think_jitter
        for text, x in ((self._left_think, left_cx - 28), (self._right_think, right_cx + 4)):
            glow = font.render(text, True, (0, 135, 145))
            glyph = font.render(text, True, (0, 245, 255))
            self.screen.blit(glow, (x + 2, base_y + 2))
            self.screen.blit(glyph, (x, base_y))

    def _draw_mic_icon(self, cx: float, cy: float) -> None:
        neon = (0, 245, 255)

        # Mic head: rounded capsule.
        head = pygame.Rect(int(cx - 12), int(cy - 24), 24, 30)
        pygame.draw.rect(self.screen, neon, head, width=3, border_radius=12)

        # Grille lines.
        for yy in (cy - 15, cy - 10, cy - 5):
            pygame.draw.line(self.screen, neon, (cx - 7, yy), (cx + 7, yy), 2)

        # Yoke/U-bracket around lower head.
        yoke = pygame.Rect(int(cx - 18), int(cy - 8), 36, 24)
        pygame.draw.arc(self.screen, neon, yoke, math.radians(200), math.radians(340), 3)

        # Stem and base stand.
        stem = pygame.Rect(int(cx - 2), int(cy + 12), 4, 10)
        base = pygame.Rect(int(cx - 13), int(cy + 22), 26, 4)
        pygame.draw.rect(self.screen, neon, stem, border_radius=2)
        pygame.draw.rect(self.screen, neon, base, border_radius=2)

    def _draw_mindvoice_bubble(self, cx: float, cy: float) -> None:
        if self.target_state not in {"thinking", "listening"}:
            return

        bob_y = math.sin(self._bubble_bob) * 2.5
        bubble_w = 122
        bubble_h = 74
        bx = int(cx - bubble_w / 2 + 80)
        by = int(cy - 170 + bob_y)
        bubble = pygame.Rect(bx, by, bubble_w, bubble_h)

        for i in range(3, 0, -1):
            glow = bubble.inflate(i * 10, i * 10)
            pygame.draw.rect(self.screen, (0, 145, 155, 26 * i), glow, border_radius=20)

        pygame.draw.rect(self.screen, (8, 20, 24), bubble, border_radius=16)
        pygame.draw.rect(self.screen, (0, 245, 255), bubble, width=3, border_radius=16)

        tail = [(bx + 22, by + bubble_h - 1), (bx + 38, by + bubble_h - 1), (bx + 30, by + bubble_h + 16)]
        pygame.draw.polygon(self.screen, (8, 20, 24), tail)
        pygame.draw.lines(self.screen, (0, 245, 255), False, tail, 3)

        if self.target_state == "thinking":
            symbol = self._left_think if math.sin(self.time * 2.4) >= 0 else self._right_think
            font = pygame.font.SysFont("arial", 40, bold=True)
            glyph = font.render(symbol, True, (0, 245, 255))
            glow = font.render(symbol, True, (0, 120, 130))
            gx = bx + bubble_w // 2 - glyph.get_width() // 2
            gy = by + bubble_h // 2 - glyph.get_height() // 2 - 2
            self.screen.blit(glow, (gx + 2, gy + 2))
            self.screen.blit(glyph, (gx, gy))
        else:
            self._draw_mic_icon(bx + bubble_w // 2, by + bubble_h // 2)

    def draw(self):
        self.screen.fill((0, 0, 0))

        # Idle micro movement to keep eyes alive.
        micro_x = math.sin(self.time * 1.2) * 1.8 + self.idle_jitter * 0.8
        micro_y = math.sin(self.time * 1.7) * 2.2 + math.sin(self.idle_phase * 2.3) * 0.8

        if self.target_state == "listening":
            micro_x += math.sin(self.time * 4.0) * 1.0
            micro_y += math.sin(self.time * 3.6) * 1.2
        elif self.target_state == "thinking":
            micro_x += math.sin(self.time * 0.9) * 3.4
            micro_y -= 2.0

        # Keep eye shapes stable during speech; mouth carries the talking animation.
        talk_x = 1.0
        talk_h = 1.0
        talk_bright = 1.0

        open_scale = self._blink_open_scale()

        w = self.base_w * self.current["w"] * talk_x
        h = self.base_h * self.current["h"] * talk_h * open_scale
        tilt = self.current["tilt"]
        arcness = self.current["arc"]
        bright = self.current["bright"] * talk_bright

        cx = self.sw * 0.5 + micro_x
        cy = self.sh * 0.5 + micro_y
        offset = self.base_w * 0.5 + self.base_gap * 0.5

        left_cx = cx - offset
        right_cx = cx + offset

        # Slight mirror tilt in sad for more expression.
        if self.target_state == "sad":
            self._draw_eye(left_cx, cy, w, h, tilt - 2.0, arcness, bright)
            self._draw_eye(right_cx, cy, w, h, tilt + 2.0, arcness, bright)
        else:
            self._draw_eye(left_cx, cy, w, h, tilt, arcness, bright)
            self._draw_eye(right_cx, cy, w, h, tilt, arcness, bright)

        self._draw_mindvoice_bubble(cx, cy)

        self._draw_mouth(cx, cy)
