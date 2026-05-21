from __future__ import annotations

import json
import os
import secrets
import threading
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from database.mongo_db import create_user, get_all_users, get_sessions_by_user_id, get_user_by_name, update_user_by_id

_STORE_PATH = Path(__file__).resolve().parents[1] / "database" / "onboarding_sessions.json"
_LOCK = threading.Lock()
_SERVER_THREAD: threading.Thread | None = None
_SERVER: ThreadingHTTPServer | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _load_store() -> dict:
    if not _STORE_PATH.exists():
        return {"sessions": {}}
    try:
        raw = json.loads(_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"sessions": {}}
    if not isinstance(raw, dict):
        return {"sessions": {}}
    sessions = raw.get("sessions")
    if not isinstance(sessions, dict):
        sessions = {}
    return {"sessions": sessions}


def _save_store(store: dict) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(store, ensure_ascii=True, indent=2), encoding="utf-8")


def _cleanup_expired(store: dict) -> None:
    now = _now()
    sessions = store.get("sessions", {})
    expired = []
    for token, doc in sessions.items():
        exp = _parse_iso(str(doc.get("expires_at", "")))
        if exp is not None and exp < now:
            expired.append(token)
    for token in expired:
        sessions.pop(token, None)


def create_onboarding_session(ttl_minutes: int = 10, base_url: str = "") -> dict:
    token = secrets.token_urlsafe(16)
    now = _now()
    exp = now + timedelta(minutes=max(1, int(ttl_minutes)))
    doc = {
        "token": token,
        "status": "pending",
        "created_at": _iso(now),
        "expires_at": _iso(exp),
        "profile": {},
    }
    with _LOCK:
        store = _load_store()
        _cleanup_expired(store)
        store["sessions"][token] = doc
        _save_store(store)

    normalized_base = (base_url or "").rstrip("/")
    onboarding_url = f"{normalized_base}/onboarding/{token}" if normalized_base else f"/onboarding/{token}"
    return {
        "token": token,
        "status": doc["status"],
        "created_at": doc["created_at"],
        "expires_at": doc["expires_at"],
        "onboarding_url": onboarding_url,
    }


def get_onboarding_session(token: str) -> dict | None:
    with _LOCK:
        store = _load_store()
        _cleanup_expired(store)
        doc = store.get("sessions", {}).get(token)
        if doc is None:
            _save_store(store)
            return None
        return dict(doc)


def complete_onboarding_session(token: str, profile: dict) -> dict | None:
    normalized = {
        "name": str(profile.get("name", "")).strip(),
        "age": str(profile.get("age", "")).strip(),
        "occupation": str(profile.get("occupation", "")).strip(),
        "field_of_study": str(profile.get("field_of_study", "")).strip(),
        "notes": str(profile.get("notes", "")).strip(),
    }
    with _LOCK:
        store = _load_store()
        _cleanup_expired(store)
        doc = store.get("sessions", {}).get(token)
        if not doc:
            _save_store(store)
            return None
        doc["status"] = "completed"
        doc["completed_at"] = _iso(_now())
        doc["profile"] = normalized
        _save_store(store)
        return dict(doc)


def get_latest_completed_session(since_iso: str | None = None) -> dict | None:
    since_dt = _parse_iso(since_iso) if since_iso else None
    with _LOCK:
        store = _load_store()
        _cleanup_expired(store)
        best: tuple[datetime, dict] | None = None
        for doc in store.get("sessions", {}).values():
            if str(doc.get("status", "")).lower() != "completed":
                continue
            completed_at = _parse_iso(str(doc.get("completed_at", "")))
            if completed_at is None:
                continue
            if since_dt is not None and completed_at < since_dt:
                continue
            if best is None or completed_at > best[0]:
                best = (completed_at, dict(doc))
        _save_store(store)
        return best[1] if best else None


def get_latest_pending_session() -> dict | None:
    with _LOCK:
        store = _load_store()
        _cleanup_expired(store)
        best: tuple[datetime, dict] | None = None
        for doc in store.get("sessions", {}).values():
            if str(doc.get("status", "")).lower() != "pending":
                continue
            created_at = _parse_iso(str(doc.get("created_at", "")))
            if created_at is None:
                continue
            if best is None or created_at > best[0]:
                best = (created_at, dict(doc))
        _save_store(store)
        return best[1] if best else None


def _normalize_name(value: str) -> str:
    cleaned = " ".join((value or "").strip().split())
    if not cleaned:
        return ""
    parts = [p for p in cleaned.split(" ") if p]
    return " ".join(parts[:2])


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _serialize_user(user: dict) -> dict:
    return {
        "id": str(user.get("_id", "")),
        "name": str(user.get("name", "")).strip(),
        "age": str(user.get("age", "")).strip(),
        "occupation": str(user.get("occupation", "")).strip(),
        "field_of_study": str(user.get("field_of_study", "")).strip(),
        "created_at": str(user.get("created_at", "")).strip(),
        "last_session_at": str(user.get("last_session_at", "")).strip(),
    }


def _session_summary(session: dict) -> dict:
    metadata = session.get("metadata") if isinstance(session.get("metadata"), dict) else {}
    ts = str(session.get("timestamp", "")).strip()
    overall_risk = _safe_float(metadata.get("overall_risk_score"), default=0.0)
    primary_problem = str(metadata.get("primary_problem", "")).strip() or "unknown"
    mood = _resolve_display_mood(metadata, overall_risk=overall_risk, primary_problem=primary_problem)
    return {
        "timestamp": ts,
        "date": ts.split("T")[0] if "T" in ts else ts,
        "overall_risk_score": round(max(0.0, min(1.0, overall_risk)), 4),
        "display_mood": mood,
        "primary_problem": primary_problem,
    }


def _resolve_display_mood(metadata: dict, overall_risk: float, primary_problem: str) -> str:
    raw = str(metadata.get("display_mood", "")).strip().lower()
    mood_aliases = {
        "joy": "happy",
        "calm": "happy",
        "neutral": "happy",
        "sadness": "sad",
        "anxiety": "sad",
        "crisis": "anger",
    }
    mood = mood_aliases.get(raw, raw)
    if mood not in {"happy", "sad", "anger"}:
        mood = "unknown"

    max_overall = _safe_float(metadata.get("session_max_overall_risk"), default=overall_risk)
    seen_crisis = bool(metadata.get("session_seen_crisis"))
    problem = (primary_problem or "").strip().lower()
    if seen_crisis or max_overall >= 0.66 or "suicidal" in problem or "suicide" in problem:
        return "anger"
    return mood


def _build_user_dashboard_payload(user: dict) -> dict:
    sessions = get_sessions_by_user_id(user.get("_id"))
    sessions = sorted(sessions, key=lambda s: str(s.get("timestamp", "")))
    summaries = [_session_summary(s) for s in sessions]

    current = summaries[-1] if summaries else None
    previous = summaries[-2] if len(summaries) > 1 else None

    if current is None:
        fallback_risk = round(max(0.0, min(1.0, _safe_float(user.get("last_overall_risk_score"), 0.0))), 4)
        fallback_problem = str(user.get("last_primary_problem", "")).strip() or "unknown"
        fallback_mood = _resolve_display_mood(
            {
                "display_mood": str(user.get("last_display_mood", "")).strip().lower(),
                "session_max_overall_risk": _safe_float(user.get("last_session_max_overall_risk"), fallback_risk),
                "session_seen_crisis": bool(user.get("last_session_seen_crisis")),
            },
            overall_risk=fallback_risk,
            primary_problem=fallback_problem,
        )
        fallback_timestamp = str(user.get("last_session_at", "")).strip()
        if fallback_timestamp:
            current = {
                "timestamp": fallback_timestamp,
                "date": fallback_timestamp.split("T")[0] if "T" in fallback_timestamp else fallback_timestamp,
                "overall_risk_score": fallback_risk,
                "display_mood": fallback_mood,
                "primary_problem": fallback_problem,
            }
    risk_change = None
    mood_changed = None
    if current and previous:
        risk_change = round(current["overall_risk_score"] - previous["overall_risk_score"], 4)
        mood_changed = current["display_mood"] != previous["display_mood"]

    return {
        "user": _serialize_user(user),
        "session_count": len(summaries),
        "current_session": current,
        "previous_session": previous,
        "comparison": {
            "risk_change": risk_change,
            "mood_changed": mood_changed,
        },
        "charts": {
            "labels": [s["date"] for s in summaries],
            "risk_series": [s["overall_risk_score"] for s in summaries],
            "mood_series": [s["display_mood"] for s in summaries],
        },
        "sessions": summaries,
    }


def _detect_or_create_user(payload: dict) -> tuple[str, dict | None]:
    name = _normalize_name(str(payload.get("name", "")))
    if not name:
        return "invalid", None

    age = str(payload.get("age", "")).strip()
    occupation = str(payload.get("occupation", "")).strip()
    field_of_study = str(payload.get("field_of_study", "")).strip()

    existing = get_user_by_name(name)
    if not existing and " " in name:
        existing = get_user_by_name(name.split(" ", 1)[0])
    if existing:
        updates: dict[str, str] = {}
        if age and not str(existing.get("age", "")).strip():
            updates["age"] = age
        if occupation and not str(existing.get("occupation", "")).strip():
            updates["occupation"] = occupation
        if field_of_study and not str(existing.get("field_of_study", "")).strip():
            updates["field_of_study"] = field_of_study
        if updates:
            update_user_by_id(existing.get("_id"), updates)
            existing.update(updates)
        return "existing", existing

    doc = {
        "name": name,
        "age": age,
        "occupation": occupation,
        "field_of_study": field_of_study,
        "created_at": _iso(_now()),
        "onboarding_complete": True,
        "pitch_baseline": 0.0,
    }
    return "new", create_user(doc)


def _dashboard_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>User Mood/Risk Dashboard</title>
  <style>
    :root {
      --bg: #ece6d9;
      --card: #fffdfa;
      --ink: #15221f;
      --accent: #0f766e;
      --accent-soft: #d8f1ee;
      --warn: #b45309;
      --danger: #b91c1c;
      --muted: #5f6b69;
      --line: #d9d4ca;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Trebuchet MS", "Lucida Sans", sans-serif;
      background:
        radial-gradient(circle at 12% 18%, #fffdf9 0%, transparent 40%),
        radial-gradient(circle at 88% 4%, #d8f1ee 0%, transparent 28%),
        linear-gradient(165deg, #f3ede1 0%, var(--bg) 52%, #e8e1d2 100%);
    }
    .wrap { max-width: 1160px; margin: 0 auto; padding: 22px 16px 28px; }
    .title { margin: 0 0 12px; font-size: 24px; letter-spacing: 0.3px; }
    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 26px rgba(33, 34, 31, 0.04);
      padding: 14px;
    }
    .top { display: grid; grid-template-columns: 1.4fr 1fr; gap: 12px; }
    .controls { display: grid; grid-template-columns: 1.3fr 0.8fr 0.9fr; gap: 8px; }
    label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    select, button {
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      padding: 10px;
      background: #fff;
      color: var(--ink);
      font-weight: 600;
    }
    button {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      cursor: pointer;
    }
    .toggle {
      display: flex;
      align-items: center;
      gap: 8px;
      height: 42px;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 0 10px;
      background: #fff;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }
    .status {
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
      padding: 8px 10px;
      border: 1px dashed var(--line);
      border-radius: 9px;
      background: #fff;
    }
    .details-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 8px;
      margin-top: 12px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fff;
    }
    .k { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.35px; }
    .v { font-size: 18px; font-weight: 800; margin-top: 2px; }
    .ok { color: var(--accent); }
    .bad { color: var(--danger); }
    .warn { color: var(--warn); }
    .chart-row { margin-top: 12px; }
    .subtle { color: var(--muted); font-size: 12px; margin-top: 6px; }
    canvas {
      width: 100%;
      height: 260px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 13px;
      background: #fff;
    }
    th, td {
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
    }
    th { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.3px; }
    .sessions-wrap {
      margin-top: 12px;
      max-height: 320px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      padding: 6px 10px;
    }
    .chip {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.3px;
      background: var(--accent-soft);
      color: #0b5f59;
    }
    .chip.bad { background: #fee2e2; color: #991b1b; }
    .chip.warn { background: #ffedd5; color: #92400e; }
    @media (max-width: 980px) {
      .top, .metrics, .controls, .details-grid { grid-template-columns: 1fr; }
      canvas { height: 230px; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1 class="title">Mental Support - User Dashboard</h1>
    <div class="top">
      <div class="card">
        <h3 style="margin:0 0 10px;">Assistant Session Data</h3>
        <div class="controls">
          <div>
            <label>Select User</label>
            <select id="userSelect"></select>
          </div>
          <div>
            <label>Actions</label>
            <button id="refreshBtn">Refresh Now</button>
          </div>
          <div>
            <label>Live Updates</label>
            <div class="toggle">
              <input id="autoRefresh" type="checkbox" />
              <span>Auto refresh 30s</span>
            </div>
          </div>
        </div>
        <div id="status" class="status">Loading users from assistant sessions...</div>
      </div>
      <div class="card">
        <h3 style="margin:0 0 10px;">User Details</h3>
        <div id="details" class="details-grid">
          <div><b>Name:</b> -</div>
          <div><b>Age:</b> -</div>
          <div><b>Occupation:</b> -</div>
          <div><b>Field:</b> -</div>
          <div><b>Created:</b> -</div>
          <div><b>Last Session:</b> -</div>
        </div>
      </div>
    </div>
    <div class="metrics">
      <div class="metric"><div class="k">Current Mood</div><div id="moodNow" class="v">-</div></div>
      <div class="metric"><div class="k">Current Risk</div><div id="riskNow" class="v">-</div></div>
      <div class="metric"><div class="k">Previous Risk</div><div id="riskPrev" class="v">-</div></div>
      <div class="metric"><div class="k">Risk Change</div><div id="riskChange" class="v">-</div></div>
      <div class="metric"><div class="k">Session Count</div><div id="sessionCount" class="v">0</div></div>
    </div>
    <div class="chart-row card">
      <h3 style="margin:0 0 6px;">Risk Trend (0.00 to 1.00)</h3>
      <canvas id="riskChart"></canvas>
      <div class="subtle">Fixed-scale graph makes risk movement readable across users and sessions.</div>
    </div>
    <div class="card" style="margin-top:12px;">
      <h3 style="margin:0 0 8px;">Session Comparison (Current vs Previous)</h3>
      <table>
        <thead><tr><th>Metric</th><th>Current</th><th>Previous</th></tr></thead>
        <tbody id="cmpRows"><tr><td colspan="3">No comparison data.</td></tr></tbody>
      </table>
    </div>
    <div class="card" style="margin-top:12px;">
      <h3 style="margin:0;">Session History</h3>
      <div class="sessions-wrap">
        <table>
          <thead><tr><th>Date</th><th>Mood</th><th>Risk</th><th>Primary Problem</th></tr></thead>
          <tbody id="sessionRows"><tr><td colspan="4">No sessions yet.</td></tr></tbody>
        </table>
      </div>
    </div>
  </div>
  <script>
    const statusEl = document.getElementById("status");
    const detailsEl = document.getElementById("details");
    const userSelectEl = document.getElementById("userSelect");
    const moodNowEl = document.getElementById("moodNow");
    const riskNowEl = document.getElementById("riskNow");
    const riskPrevEl = document.getElementById("riskPrev");
    const riskChangeEl = document.getElementById("riskChange");
    const sessionCountEl = document.getElementById("sessionCount");
    const cmpRowsEl = document.getElementById("cmpRows");
    const sessionRowsEl = document.getElementById("sessionRows");
    const autoRefreshEl = document.getElementById("autoRefresh");

    const USERS_CACHE_KEY = "dashboard_users_cache_v2";
    const SELECTED_USER_KEY = "dashboard_selected_user_v2";
    const USER_CACHE_PREFIX = "dashboard_user_cache_v2::";

    function userCacheKey(name) { return USER_CACHE_PREFIX + name; }
    function toFixed2(value) {
      const num = Number(value);
      return Number.isFinite(num) ? num.toFixed(2) : "-";
    }
    function moodClass(mood) {
      const m = String(mood || "").toLowerCase();
      if (m === "anger") return "bad";
      if (m === "sad") return "warn";
      if (m === "happy") return "ok";
      return "";
    }
    function setStatus(text) { statusEl.textContent = text; }

    function saveJson(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) {}
    }
    function loadJson(key, fallbackValue) {
      try {
        const raw = localStorage.getItem(key);
        if (!raw) return fallbackValue;
        return JSON.parse(raw);
      } catch (e) {
        return fallbackValue;
      }
    }

    function dateFromISO(iso) {
      if (!iso) return "-";
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return iso;
      return d.toLocaleString();
    }

    function drawRiskChart(labels, values) {
      const canvas = document.getElementById("riskChart");
      const ctx = canvas.getContext("2d");
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const cssWidth = Math.max(280, canvas.clientWidth || 960);
      const cssHeight = Math.max(220, canvas.clientHeight || 260);
      canvas.width = Math.floor(cssWidth * dpr);
      canvas.height = Math.floor(cssHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const w = cssWidth;
      const h = cssHeight;
      const left = 46;
      const right = w - 16;
      const top = 14;
      const bottom = h - 30;
      const plotW = Math.max(1, right - left);
      const plotH = Math.max(1, bottom - top);

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, w, h);

      const yTicks = [0, 0.25, 0.5, 0.75, 1];
      yTicks.forEach((tick) => {
        const y = bottom - (tick * plotH);
        ctx.strokeStyle = tick >= 0.75 ? "#fecaca" : "#e5e7eb";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        ctx.fillStyle = "#6b7280";
        ctx.font = "11px Trebuchet MS";
        ctx.fillText(tick.toFixed(2), 8, y + 4);
      });

      ctx.strokeStyle = "#cbd5e1";
      ctx.beginPath();
      ctx.moveTo(left, top);
      ctx.lineTo(left, bottom);
      ctx.lineTo(right, bottom);
      ctx.stroke();

      if (!values || values.length === 0) {
        ctx.fillStyle = "#6b7280";
        ctx.font = "13px Trebuchet MS";
        ctx.fillText("No session series available yet.", left + 8, top + 20);
        return;
      }

      const cleaned = values.map((v) => Math.max(0, Math.min(1, Number(v) || 0)));
      const xFor = (i) => left + (i * plotW / Math.max(1, cleaned.length - 1));
      const yFor = (v) => bottom - (v * plotH);

      ctx.strokeStyle = "#0f766e";
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      cleaned.forEach((v, i) => {
        const x = xFor(i);
        const y = yFor(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();

      cleaned.forEach((v, i) => {
        const x = xFor(i);
        const y = yFor(v);
        ctx.beginPath();
        ctx.fillStyle = v >= 0.75 ? "#b91c1c" : (v >= 0.5 ? "#b45309" : "#0f766e");
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();
      });

      const step = Math.max(1, Math.ceil(cleaned.length / 6));
      ctx.fillStyle = "#6b7280";
      ctx.font = "11px Trebuchet MS";
      for (let i = 0; i < cleaned.length; i += step) {
        const x = xFor(i);
        const label = (labels && labels[i]) ? String(labels[i]).slice(5) : String(i + 1);
        ctx.fillText(label, Math.max(left, x - 12), h - 10);
      }
    }

    function setComparisonRows(current, previous) {
      if (!current) {
        cmpRowsEl.innerHTML = "<tr><td colspan='3'>No session data.</td></tr>";
        return;
      }
      const prev = previous || {};
      cmpRowsEl.innerHTML = `
        <tr><td>Date</td><td>${current.date || "-"}</td><td>${prev.date || "-"}</td></tr>
        <tr><td>Mood</td><td><span class="chip ${moodClass(current.display_mood)}">${current.display_mood || "-"}</span></td><td>${prev.display_mood || "-"}</td></tr>
        <tr><td>Overall Risk</td><td>${toFixed2(current.overall_risk_score)}</td><td>${toFixed2(prev.overall_risk_score)}</td></tr>
        <tr><td>Primary Problem</td><td>${current.primary_problem || "-"}</td><td>${prev.primary_problem || "-"}</td></tr>
      `;
    }

    function setSessionRows(sessions) {
      const rows = Array.isArray(sessions) ? sessions : [];
      if (!rows.length) {
        sessionRowsEl.innerHTML = "<tr><td colspan='4'>No sessions yet.</td></tr>";
        return;
      }
      sessionRowsEl.innerHTML = rows.slice().reverse().map((s) => {
        const mood = String(s.display_mood || "-");
        const klass = moodClass(mood);
        return `<tr>
          <td>${s.date || "-"}</td>
          <td><span class="chip ${klass}">${mood}</span></td>
          <td>${toFixed2(s.overall_risk_score)}</td>
          <td>${s.primary_problem || "-"}</td>
        </tr>`;
      }).join("");
    }

    function updateUI(payload) {
      const u = payload.user || {};
      detailsEl.innerHTML = `
        <div><b>Name:</b> ${u.name || "-"}</div>
        <div><b>Age:</b> ${u.age || "-"}</div>
        <div><b>Occupation:</b> ${u.occupation || "-"}</div>
        <div><b>Field:</b> ${u.field_of_study || "-"}</div>
        <div><b>Created:</b> ${dateFromISO(u.created_at)}</div>
        <div><b>Last Session:</b> ${dateFromISO(u.last_session_at)}</div>
      `;

      const curr = payload.current_session;
      const prev = payload.previous_session;
      const cmp = payload.comparison || {};
      moodNowEl.textContent = curr ? (curr.display_mood || "-") : "-";
      moodNowEl.className = "v " + moodClass(curr ? curr.display_mood : "");
      riskNowEl.textContent = curr ? toFixed2(curr.overall_risk_score) : "-";
      riskPrevEl.textContent = prev ? toFixed2(prev.overall_risk_score) : "-";
      sessionCountEl.textContent = String(payload.session_count || 0);

      if (cmp.risk_change == null) {
        riskChangeEl.textContent = "-";
        riskChangeEl.className = "v";
      } else {
        const value = Number(cmp.risk_change) || 0;
        riskChangeEl.textContent = (value > 0 ? "+" : "") + value.toFixed(2);
        riskChangeEl.className = "v " + (value > 0.08 ? "bad" : (value < -0.08 ? "ok" : "warn"));
      }

      setComparisonRows(curr, prev);
      setSessionRows(payload.sessions || []);

      const charts = payload.charts || {};
      drawRiskChart(charts.labels || [], charts.risk_series || []);
    }

    function sortUsersByRecency(users) {
      const copy = [...(users || [])];
      copy.sort((a, b) => {
        const ad = (a.user && (a.user.last_session_at || a.user.created_at)) || "";
        const bd = (b.user && (b.user.last_session_at || b.user.created_at)) || "";
        return ad < bd ? 1 : -1;
      });
      return copy;
    }

    function populateUserSelect(users, selectedName) {
      const rows = sortUsersByRecency(users);
      userSelectEl.innerHTML = "";
      if (!rows.length) {
        userSelectEl.innerHTML = "<option value=''>No users yet</option>";
        return "";
      }
      rows.forEach((row) => {
        const name = (row.user && row.user.name) || "";
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = `${name || "-"} (${row.session_count || 0} sessions)`;
        userSelectEl.appendChild(opt);
      });

      const validNames = new Set(rows.map((row) => (row.user && row.user.name) || ""));
      let target = selectedName || "";
      if (!validNames.has(target)) {
        target = rows[0] && rows[0].user ? rows[0].user.name : "";
      }
      userSelectEl.value = target || "";
      return target || "";
    }

    async function loadUserByName(name, preferCache = true) {
      if (!name) {
        setStatus("No user selected.");
        return;
      }
      localStorage.setItem(SELECTED_USER_KEY, name);
      setStatus(`Loading ${name}...`);
      try {
        const r = await fetch(`/api/dashboard/user?name=${encodeURIComponent(name)}`);
        const data = await r.json();
        if (!r.ok || !data.ok) {
          throw new Error(data.error || "failed_to_load");
        }
        const dashboard = data.dashboard || {};
        updateUI(dashboard);
        saveJson(userCacheKey(name), dashboard);
        setStatus(`Loaded assistant data for ${name}.`);
      } catch (err) {
        const cached = preferCache ? loadJson(userCacheKey(name), null) : null;
        if (cached) {
          updateUI(cached);
          setStatus(`Showing cached data for ${name} (live connection unavailable).`);
          return;
        }
        setStatus(`Network error while loading ${name}.`);
      }
    }

    async function refreshUsersAndCurrent() {
      const previousSelected = userSelectEl.value || localStorage.getItem(SELECTED_USER_KEY) || "";
      setStatus("Refreshing users...");
      let users = [];
      try {
        const r = await fetch("/api/dashboard/users");
        const data = await r.json();
        if (!r.ok || !data.ok) throw new Error("users_fetch_failed");
        users = data.users || [];
        saveJson(USERS_CACHE_KEY, users);
      } catch (err) {
        users = loadJson(USERS_CACHE_KEY, []);
        if (!users.length) {
          userSelectEl.innerHTML = "<option value=''>No users yet</option>";
          setStatus("No user list available (live and cache unavailable).");
          return;
        }
        setStatus("Using cached user list (server not reachable).");
      }

      const selected = populateUserSelect(users, previousSelected);
      if (!selected) {
        setStatus("No users available yet.");
        return;
      }
      await loadUserByName(selected, true);
    }

    userSelectEl.addEventListener("change", async () => {
      await loadUserByName(userSelectEl.value, true);
    });

    document.getElementById("refreshBtn").addEventListener("click", async () => {
      await refreshUsersAndCurrent();
    });

    setInterval(async () => {
      if (!autoRefreshEl.checked) return;
      const name = userSelectEl.value || localStorage.getItem(SELECTED_USER_KEY) || "";
      if (!name) return;
      await loadUserByName(name, true);
    }, 30000);

    window.addEventListener("resize", () => {
      const selected = userSelectEl.value || localStorage.getItem(SELECTED_USER_KEY) || "";
      if (!selected) return;
      const cached = loadJson(userCacheKey(selected), null);
      if (cached) {
        const charts = cached.charts || {};
        drawRiskChart(charts.labels || [], charts.risk_series || []);
      }
    });

    refreshUsersAndCurrent();
  </script>
</body>
</html>"""


def _html_form(token: str, status_message: str = "") -> str:
    status_block = f"<p class='status'>{status_message}</p>" if status_message else ""
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Mental Support Intake</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #0e141b; color: #f5f7fa; }}
    .card {{ max-width: 680px; margin: 16px auto; padding: 20px; background: #17222d; border-radius: 12px; }}
    h1 {{ margin-top: 0; font-size: 22px; }}
    label {{ display: block; margin: 14px 0 6px; font-size: 14px; }}
    input, textarea {{ width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #2d3b4a; border-radius: 8px; background: #0f1822; color: #f5f7fa; }}
    button {{ margin-top: 16px; padding: 10px 14px; border: 0; border-radius: 8px; background: #1db8ff; color: #06202d; font-weight: 700; }}
    .hint {{ font-size: 12px; color: #9ab3c8; }}
    .status {{ padding: 8px 10px; border-radius: 6px; background: #203445; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Mental Support Assistant - Quick Profile</h1>
    <p class="hint">This information helps the assistant personalize support.</p>
    {status_block}
    <form method="post" action="/api/onboarding/{token}/submit">
      <label for="name">Full name</label>
      <input id="name" name="name" required maxlength="80" />
      <label for="age">Age</label>
      <input id="age" name="age" inputmode="numeric" maxlength="8" />
      <label for="occupation">Occupation</label>
      <input id="occupation" name="occupation" maxlength="120" />
      <label for="field_of_study">Field of study (if student)</label>
      <input id="field_of_study" name="field_of_study" maxlength="120" />
      <label for="notes">Optional details</label>
      <textarea id="notes" name="notes" rows="3" maxlength="500"></textarea>
      <button type="submit">Submit</button>
    </form>
  </div>
</body>
</html>"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "OnboardingHTTP/1.0"

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_redirect(self, location: str, status: int = 302) -> None:
        self.send_response(status)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _base_url(self) -> str:
        public = (os.getenv("ONBOARDING_PUBLIC_BASE_URL") or "").strip().rstrip("/")
        if public:
            return public
        host = self.headers.get("Host", "")
        proto = self.headers.get("X-Forwarded-Proto", "http")
        return f"{proto}://{host}" if host else ""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/dashboard":
            self._send_html(_dashboard_html())
            return

        if path == "/api/dashboard/users":
            users = get_all_users()
            payload = []
            for user in users:
                dashboard = _build_user_dashboard_payload(user)
                payload.append(
                    {
                        "user": dashboard["user"],
                        "session_count": dashboard["session_count"],
                        "current_session": dashboard["current_session"],
                    }
                )
            self._send_json({"ok": True, "users": payload})
            return

        if path == "/api/dashboard/user":
            qs = parse_qs(parsed.query or "", keep_blank_values=False)
            query_name = str((qs.get("name") or [""])[0]).strip()
            if not query_name:
                self._send_json({"ok": False, "error": "name_required"}, status=400)
                return
            user = get_user_by_name(query_name)
            if not user and " " in query_name:
                user = get_user_by_name(query_name.split(" ", 1)[0])
            if not user:
                self._send_json({"ok": False, "error": "user_not_found"}, status=404)
                return
            self._send_json({"ok": True, "dashboard": _build_user_dashboard_payload(user)})
            return

        if path == "/health":
            self._send_json({"ok": True, "service": "onboarding"})
            return

        if path == "/api/onboarding/latest-completed":
            qs = parse_qs(parsed.query or "", keep_blank_values=False)
            since = ""
            values = qs.get("since") or []
            if values:
                since = str(values[0]).strip()
            doc = get_latest_completed_session(since_iso=since or None)
            if not doc:
                self._send_json({"status": "none"})
                return
            self._send_json(
                {
                    "status": "completed",
                    "token": doc.get("token", ""),
                    "completed_at": doc.get("completed_at", ""),
                    "profile": doc.get("profile", {}),
                }
            )
            return

        if path.startswith("/api/onboarding/") and path.endswith("/status"):
            token = path.split("/")[-2]
            doc = get_onboarding_session(token)
            if not doc:
                self._send_json({"error": "session_not_found"}, status=404)
                return
            self._send_json(
                {
                    "token": token,
                    "status": doc.get("status", "pending"),
                    "expires_at": doc.get("expires_at"),
                    "profile": doc.get("profile", {}),
                }
            )
            return

        if path == "/onboarding/latest":
            doc = get_latest_pending_session()
            if not doc:
                self._send_html("<h2>No active onboarding session right now.</h2>", status=404)
                return
            token = str(doc.get("token", "")).strip()
            if not token:
                self._send_html("<h2>No active onboarding session right now.</h2>", status=404)
                return
            self._send_redirect(f"/onboarding/{token}")
            return

        if path.startswith("/onboarding/"):
            token = path.split("/")[-1]
            doc = get_onboarding_session(token)
            if not doc:
                latest = get_latest_pending_session()
                latest_token = str((latest or {}).get("token", "")).strip()
                if latest_token:
                    self._send_redirect(f"/onboarding/{latest_token}")
                    return
                self._send_html("<h2>Session expired or not found.</h2>", status=404)
                return
            if doc.get("status") == "completed":
                self._send_html(_html_form(token, status_message="Already submitted. You can close this page."))
                return
            self._send_html(_html_form(token))
            return

        self._send_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/api/dashboard/user/identify":
            body = self._read_body().decode("utf-8", errors="ignore").strip()
            payload = {}
            if body:
                try:
                    payload = json.loads(body)
                except Exception:
                    payload = {}
            user_state, user = _detect_or_create_user(payload)
            if user_state == "invalid" or not user:
                self._send_json({"ok": False, "error": "valid_name_required"}, status=400)
                return
            self._send_json(
                {
                    "ok": True,
                    "user_state": user_state,
                    "dashboard": _build_user_dashboard_payload(user),
                }
            )
            return

        if path == "/api/onboarding/session":
            body = self._read_body().decode("utf-8", errors="ignore").strip()
            payload = {}
            if body:
                try:
                    payload = json.loads(body)
                except Exception:
                    payload = {}
            ttl = int(payload.get("ttl_minutes", 10) or 10)
            session = create_onboarding_session(ttl_minutes=ttl, base_url=self._base_url())
            self._send_json(session, status=201)
            return

        if path.startswith("/api/onboarding/") and path.endswith("/submit"):
            token = path.split("/")[-2]
            content_type = (self.headers.get("Content-Type") or "").lower()
            raw = self._read_body().decode("utf-8", errors="ignore")

            if "application/json" in content_type:
                try:
                    profile = json.loads(raw)
                except Exception:
                    profile = {}
            else:
                form = parse_qs(raw, keep_blank_values=True)
                profile = {k: (v[0] if isinstance(v, list) and v else "") for k, v in form.items()}

            updated = complete_onboarding_session(token, profile)
            if not updated:
                self._send_json({"error": "session_not_found"}, status=404)
                return

            if "application/json" in content_type:
                self._send_json({"ok": True, "status": "completed"})
            else:
                self._send_html("<h2>Submitted successfully. You can return to the assistant now.</h2>")
            return

        self._send_json({"error": "not_found"}, status=404)

    def log_message(self, format: str, *args) -> None:
        return


def ensure_server_started() -> tuple[str, int]:
    global _SERVER_THREAD, _SERVER
    if _SERVER_THREAD and _SERVER_THREAD.is_alive() and _SERVER is not None:
        host, port = _SERVER.server_address
        return str(host), int(port)

    host = (os.getenv("ONBOARDING_SERVER_HOST") or "0.0.0.0").strip()
    port = int(os.getenv("ONBOARDING_SERVER_PORT") or 8765)
    _SERVER = ThreadingHTTPServer((host, port), _Handler)

    def _serve() -> None:
        assert _SERVER is not None
        _SERVER.serve_forever(poll_interval=0.3)

    _SERVER_THREAD = threading.Thread(target=_serve, name="onboarding-server", daemon=True)
    _SERVER_THREAD.start()
    return host, port


def main() -> None:
    host, port = ensure_server_started()
    print(f"Onboarding server running on {host}:{port}")
    try:
        while True:
            threading.Event().wait(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
