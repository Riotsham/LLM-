import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import librosa
except ModuleNotFoundError:
    librosa = None

# Ensure project root is on sys.path so sibling packages (like `audio`) can be imported.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

try:
    from audio.record import SAMPLE_RATE, model as whisper_model, record_chunk
    from audio.tts import speak, speak_blocking as edge_speak_blocking
except ModuleNotFoundError:
    print("Could not import 'audio' package. Make sure you're running the script from the project root.")
    raise

try:
    from emotion.text_emotion import detect_text_emotion, format_emotion_context
except Exception:
    detect_text_emotion = None
    format_emotion_context = None

try:
    from emotion.voice_emotion import detect_voice_emotion, format_voice_emotion_context
except Exception:
    detect_voice_emotion = None
    format_voice_emotion_context = None

from llm.prompt_engineering import decide_action
from risk.risk_model import detect_risk
from risk.rules import apply_rules
from rag.retriever import retrieve
from rag.voice_features import extract_voice_indicators
from conversation_state import STATE
from helpers import (
    decline_exercise_response,
    detect_breathing_offer,
    is_affirmative,
    is_greeting,
    is_negative,
    is_out_of_scope,
    redirect_message,
    run_breathing_sequence,
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
_SELECTED_MODEL: str | None = None
_NO_MODELS_WARNED = False
QUESTION_LOOP_THRESHOLD = 2  # If the last N assistant replies are questions, force the next reply to be guidance.
LLM_MAX_RETRIES = 2


def speak_blocking(text: str) -> None:
    if not text:
        return
    edge_speak_blocking(text)


def _fallback_support_reply(user_text: str = "") -> str:
    text = (user_text or "").strip()
    if text:
        return (
            "I am here with you. My local model is not ready right now, "
            "but I still want to support you. Please share one small thing that feels hardest at this moment."
        )
    return (
        "I am here with you. My local model is not ready right now, "
        "but we can still take this one step at a time."
    )


def _list_available_models(ollama: OpenAI) -> list[str]:
    """Return installed model ids from Ollama's OpenAI-compatible model endpoint."""
    try:
        models = ollama.models.list()
        return [m.id for m in getattr(models, "data", []) if getattr(m, "id", None)]
    except Exception:
        return []


def _resolve_model(ollama: OpenAI, force_refresh: bool = False) -> str:
    """Resolve a usable model name, preferring env-configured model then installed models."""
    global _SELECTED_MODEL

    if _SELECTED_MODEL and not force_refresh:
        return _SELECTED_MODEL

    available = _list_available_models(ollama)
    preferred = (MODEL or "").strip()

    if preferred and preferred in available:
        _SELECTED_MODEL = preferred
        return _SELECTED_MODEL

    if preferred and available:
        preferred_base = preferred.split(":", 1)[0]
        for installed in available:
            if installed.split(":", 1)[0] == preferred_base:
                _SELECTED_MODEL = installed
                print(f"[LLM] Model '{preferred}' not found. Falling back to installed model '{installed}'.")
                return _SELECTED_MODEL

    if available:
        _SELECTED_MODEL = available[0]
        if preferred and preferred != _SELECTED_MODEL:
            print(f"[LLM] Model '{preferred}' not found. Using installed model '{_SELECTED_MODEL}'.")
        return _SELECTED_MODEL

    _SELECTED_MODEL = preferred or "llama3.1:8b"
    return _SELECTED_MODEL


def _safe_chat_completion(messages: list[dict[str, str]]) -> str | None:
    """Run local Ollama chat completion with short retries and graceful failure."""
    global _SELECTED_MODEL, _NO_MODELS_WARNED
    ollama = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    if not _list_available_models(ollama):
        if not _NO_MODELS_WARNED:
            preferred = (MODEL or "llama3.1:8b").strip() or "llama3.1:8b"
            print(
                "[LLM] No Ollama models are installed. "
                f"Run: ollama pull {preferred}"
            )
            _NO_MODELS_WARNED = True
        return None
    for attempt in range(LLM_MAX_RETRIES + 1):
        try:
            model_name = _resolve_model(ollama)
            response = ollama.chat.completions.create(model=model_name, messages=messages)
            return (response.choices[0].message.content or "").strip()
        except (APIConnectionError, APITimeoutError) as exc:
            print(
                f"[LLM] Connection/timeout error on attempt {attempt + 1}/{LLM_MAX_RETRIES + 1}: "
                f"{exc.__class__.__name__}: {exc}"
            )
            if attempt >= LLM_MAX_RETRIES:
                break
            time.sleep(0.6 * (attempt + 1))
        except APIError as exc:
            print(
                f"[LLM] API error on attempt {attempt + 1}/{LLM_MAX_RETRIES + 1}: "
                f"{exc.__class__.__name__}: {exc}"
            )
            if "not found" in str(exc).lower():
                _SELECTED_MODEL = None
                _resolve_model(ollama, force_refresh=True)
            if attempt >= LLM_MAX_RETRIES:
                break
            time.sleep(0.6 * (attempt + 1))
        except Exception as exc:
            # Keep broad protection so the conversation loop remains alive.
            print(
                f"[LLM] Unexpected error on attempt {attempt + 1}/{LLM_MAX_RETRIES + 1}: "
                f"{exc.__class__.__name__}: {exc}"
            )
            break
    return None

# Base identity shared by all tones. This keeps personality stable while allowing style shifts per emotion.
SYSTEM_PROMPT_BASE = """
You are a warm, supportive, non-clinical mental-health assistant who sounds like a caring friend.

Global response structure (always apply):
1) Start with emotional acknowledgment.
2) Add a reflection or supportive statement.
3) Offer one helpful idea or perspective.
4) Ask at most ONE question, and only if it improves understanding in a meaningful way.

Global guardrails:
- Never interrogate or repeatedly probe without adding value.
- Avoid repetitive clarifying questions.
- If enough context exists, prioritize support and direction over more questioning.
- Output plain text only.
""".strip()

# Dedicated crisis prompt for self-harm/suicide-related messages.
# Important: this is supportive-safety language (care + connection + real-world support),
# not refusal-safety language (policy references, "can't help", or robotic denial).
SYSTEM_PROMPT_CRISIS = """
You are a caring mental-health support assistant responding to a person in acute emotional pain.

Tone:
- Calm, human, gentle, and serious.
- Keep wording warm and direct.
- Avoid sounding clinical, scripted, or overly verbose.

Response goals:
1) Begin with emotional acknowledgment of pain and overwhelm.
2) Ask directly if they are safe right now.
3) Encourage reaching out to a trusted person immediately.
4) Offer to book an urgent appointment with our mental health experts.
5) Keep the response focused on safety and immediate professional support.

Safety constraints:
- Do not provide instructions, methods, planning details, or optimization for self-harm.
- Do not suggest therapeutic techniques (for example, breathing exercises or CBT steps).
- Do not provide medical, legal, or clinical advice or diagnosis.
- Do not use refusal-style wording.
- Do not mention policy, rules, or compliance language.
- Never use phrases such as:
  "I cannot help with that"
  "I am not able to provide"
  "This violates policy"
  "I can't assist"

Style guardrails:
- Keep the person emotionally centered, not the assistant.
- Prefer short-to-medium paragraphs.
- Plain text only.
""".strip()


def build_structured_crisis_response(user_text: str) -> str:
    """Return a structured, non-LLM crisis response with safety checks and resources."""
    _ = (user_text or "").strip()

    return (
        "I hear how much this is hurting, and I'm really glad you told me.\n\n"
        "Are you safe right now?\n\n"
        "If you can, please reach out to someone you trust right away and let them know what you're going through. "
        "You don't have to handle this alone.\n\n"
        "I can book an urgent appointment for you with our mental health experts right now.\n\n"
        "Would you like me to book that appointment for you?"
    )


def _normalize_emotion(emotion: str) -> str:
    """Map raw classifier labels into supported tone buckets.

    This prevents brittle prompt behavior when models return nearby labels such as
    "fear", "stress", or "neutral".
    """
    label = (emotion or "").strip().lower()

    aliases = {
        "happy": "joy",
        "positive": "joy",
        "neutral": "calm",
        "fear": "anxiety",
        "stressed": "anxiety",
        "stress": "anxiety",
        "frustrated": "anger",
        "frustration": "anger",
        "emergency": "crisis",
    }
    normalized = aliases.get(label, label)

    if normalized in {"joy", "calm", "sadness", "anxiety", "anger", "crisis"}:
        return normalized
    return "calm"


def get_tone_rules(emotion: str) -> str:
    """Return tone instructions for the requested emotional state.

    The model receives one focused style block so the wording feels adaptive instead of generic.
    """
    tone = _normalize_emotion(emotion)

    rules = {
        "joy": (
            "Tone: light, relaxed, and friendly.\n"
            "- Use casual, warm phrasing.\n"
            "- Do not do deep emotional probing.\n"
            "- Encourage positive reflection and continuation of what is helping."
        ),
        "calm": (
            "Tone: natural and balanced.\n"
            "- Keep a conversational style with gentle friendliness.\n"
            "- Engage without over-intensity.\n"
            "- Offer small, practical support when useful."
        ),
        "sadness": (
            "Tone: warm, gentle, and emotionally validating.\n"
            "- Avoid slang and avoid humor.\n"
            "- Acknowledge emotional weight directly and kindly.\n"
            "- Emphasize steady supportive presence.\n"
            "- Ask only one soft question at most."
        ),
        "anxiety": (
            "Tone: calm and grounding.\n"
            "- Use shorter, reassuring sentences and slower pacing language.\n"
            "- Reduce cognitive load with one clear idea at a time.\n"
            "- Suggest a simple grounding step (for example: breath count, 5-4-3-2-1, unclenching shoulders)."
        ),
        "anger": (
            "Tone: non-judgmental and steady.\n"
            "- Validate the feeling without encouraging aggression.\n"
            "- Focus on understanding what happened and what control is possible now.\n"
            "- Channel energy toward safe, constructive next steps."
        ),
        "crisis": (
            "Tone: serious, calm, and direct.\n"
            "- No casual language.\n"
            "- Prioritize immediate safety and connection to trusted support.\n"
            "- Encourage contacting a trusted person or local emergency/crisis support.\n"
            "- Be human and supportive, not robotic."
        ),
    }
    return rules[tone]


def build_system_prompt(emotion: str) -> str:
    """Compose the final system prompt from base identity + emotion-specific tone rules."""
    return (
        f"{SYSTEM_PROMPT_BASE}\n\n"
        "Tone control instructions for this turn:\n"
        f"{get_tone_rules(emotion)}"
    )


def build_crisis_prompt(user_text: str, history: list[str]) -> str:
    """Build a single crisis-focused prompt string for LLMs that use text prompts.

    Why this exists:
    - Supportive safety keeps emotional connection first, then guides toward immediate safety.
    - Refusal safety often sounds cold and can reduce trust in high-risk moments.
    """
    cleaned_user_text = (user_text or "").strip()
    cleaned_history = [h.strip() for h in (history or []) if h and h.strip()]
    history_block = "\n".join(f"- {line}" for line in cleaned_history[-8:]) or "- (no prior history)"

    return (
        f"{SYSTEM_PROMPT_CRISIS}\n\n"
        "Conversation context (most recent items):\n"
        f"{history_block}\n\n"
        "Latest user message:\n"
        f"{cleaned_user_text}\n\n"
        "Write the next assistant reply now."
    )


# Common question starters used to catch question-like text even when "?" is missing.
_QUESTION_STARTERS = (
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "who",
    "can",
    "could",
    "would",
    "should",
    "is",
    "are",
    "do",
    "did",
)


def _parse_context_item(item: str) -> tuple[str, str]:
    """Normalize one history item into a chat role and content.

    We accept lightweight prefixes like "user:" and "assistant:" so callers can pass
    simple strings while still preserving role information.
    """
    text = (item or "").strip()
    lowered = text.lower()

    if lowered.startswith("user:"):
        return "user", text.split(":", 1)[1].strip()
    if lowered.startswith("assistant:"):
        return "assistant", text.split(":", 1)[1].strip()

    # Default unknown history items to user text to avoid over-counting assistant questions.
    return "user", text


def _looks_like_question(text: str) -> bool:
    """Detect question-like replies to enforce anti-loop behavior.

    We check for a trailing '?' and question starters because some model outputs ask
    questions without punctuation.
    """
    cleaned = (text or "").strip().lower()
    if not cleaned:
        return False
    if cleaned.endswith("?"):
        return True

    first_word = cleaned.split(" ", 1)[0]
    return first_word in _QUESTION_STARTERS


def _recent_assistant_questions(context: list[str], n: int = QUESTION_LOOP_THRESHOLD) -> bool:
    """Return True when the last N assistant replies are all questions.

    This blocks the common failure mode where the assistant keeps probing but does not help.
    """
    assistant_messages: list[str] = []
    for item in context or []:
        role, content = _parse_context_item(item)
        if role == "assistant" and content:
            assistant_messages.append(content)

    if len(assistant_messages) < n:
        return False

    recent = assistant_messages[-n:]
    return all(_looks_like_question(msg) for msg in recent)


_EMOTIONAL_KEYWORDS = {
    "stress",
    "stressed",
    "sad",
    "anxious",
    "anxiety",
    "depressed",
    "depression",
    "angry",
    "upset",
    "panic",
    "overwhelmed",
    "lonely",
    "hurt",
}


def _has_explicit_emotion_keywords(text: str) -> bool:
    cleaned = (text or "").lower()
    return any(k in cleaned for k in _EMOTIONAL_KEYWORDS)


def _transcribe_audio(audio_signal: np.ndarray) -> str:
    segments, _ = whisper_model.transcribe(audio_signal, vad_filter=True, beam_size=1)
    return " ".join(seg.text for seg in segments).strip()


def _extract_average_pitch(audio_signal: np.ndarray, sample_rate: int) -> float:
    if librosa is None or audio_signal is None or sample_rate <= 0:
        return 0.0
    y = np.asarray(audio_signal, dtype=np.float32).flatten()
    if y.size == 0:
        return 0.0
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sample_rate)
    valid = f0[np.isfinite(f0) & (f0 > 0)]
    return float(np.mean(valid)) if valid.size else 0.0


def _capture_user_turn(prompt_text: str | None = None) -> tuple[str, np.ndarray, int]:
    if prompt_text:
        print(prompt_text)
        try:
            speak_blocking(prompt_text)
        except Exception:
            pass
    audio_signal = record_chunk()
    transcript = _transcribe_audio(audio_signal)
    return transcript, audio_signal, SAMPLE_RATE


def listen() -> tuple[str, np.ndarray]:
    """Record one audio turn and return (transcript, raw_audio_signal)."""
    audio_signal = record_chunk()
    transcript = _transcribe_audio(audio_signal)
    return transcript, audio_signal


def extract_average_pitch(audio_signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Public pitch helper used by session flows."""
    if librosa is None or audio_signal is None or sample_rate <= 0:
        return 0.0
    y = np.asarray(audio_signal, dtype=np.float32).flatten()
    if y.size == 0:
        return 0.0
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    f0 = librosa.yin(y, fmin=50, fmax=300, sr=sample_rate)
    valid = f0[np.isfinite(f0) & (f0 > 0)]
    return float(np.mean(valid)) if valid.size else 0.0


def generate_llama_response(prompt: str, system_prompt: str | None = None) -> str:
    """Generate text using local Llama via Ollama-compatible OpenAI API."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": (prompt or "").strip()})

    reply = _safe_chat_completion(messages)
    if reply:
        return reply
    return _fallback_support_reply(prompt)


def _extract_name(text: str) -> str:
    cleaned = (text or "").strip()
    lowered = cleaned.lower()
    prefixes = ("my name is ", "i am ", "i'm ", "im ")
    for p in prefixes:
        if lowered.startswith(p):
            return cleaned[len(p) :].strip().split(" ")[0].strip(".,!?")
    return cleaned.split(" ")[0].strip(".,!?")


def generate_response(
    user_text: str,
    context: list[str],
    audio_signal=None,
    sample_rate: int = 0,
    user_profile: dict | None = None,
) -> str:
    """Generate one supportive assistant reply using system policy + history.

    Args:
        user_text: The current user message.
        context: Prior turns as strings, ideally prefixed with "user:" / "assistant:".

    Returns:
        Plain-text assistant response.
    """
    if not user_text or not user_text.strip():
        return "I hear you. If you want, share what feels most difficult right now, and we can take one small step together."

    if is_greeting(user_text) and not _has_explicit_emotion_keywords(user_text):
        return "Hi there. I’m here to support you. How are you feeling today?"

    # Handle pending follow-ups deterministically before normal pipeline.
    if STATE.pending_action == "breathing":
        if is_affirmative(user_text):
            STATE.clear()
            return run_breathing_sequence()
        if is_negative(user_text):
            STATE.clear()
            return decline_exercise_response()

    if is_out_of_scope(user_text):
        return redirect_message()

    messages: list[dict[str, str]] = []

    if detect_text_emotion and format_emotion_context:
        emotion_result = detect_text_emotion(user_text)
        emotion_context = format_emotion_context(emotion_result)
        emotion_label = str(emotion_result.get("label", "")).lower()
        emotion_confidence = float(emotion_result.get("score", 0.0))
    else:
        emotion_context = "Detected text emotion: unknown (0.00). Top scores: none."
        emotion_label = ""
        emotion_confidence = 0.0

    # Risk detection runs before LLM generation so we can log/escalate safely
    # without changing downstream response behavior.
    # Rule-based detection is used for transparency and low-latency signals.
    risk_score = detect_risk(user_text)
    state = apply_rules(risk_score)
    print(f"Risk Score: {risk_score:.2f}, State: {state}")

    if state == "CRISIS":
        return build_structured_crisis_response(user_text)

    # Retrieve coping strategies before LLM generation.
    retrieved_context = retrieve(user_text)
    current_pitch = _extract_average_pitch(audio_signal, sample_rate)
    profile = user_profile or {}
    baseline = float(profile.get("pitch_baseline") or 0.0)
    if baseline > 0 and current_pitch > baseline * 1.15:
        mood_hint = "higher than usual"
    elif baseline > 0 and current_pitch < baseline * 0.85:
        mood_hint = "lower than usual"
    else:
        mood_hint = "within normal range"

    voice_indicators = extract_voice_indicators(audio_signal, sample_rate)
    energy_level = voice_indicators["energy_level"]
    zcr_level = voice_indicators["zcr_level"]

    if detect_voice_emotion and format_voice_emotion_context:
        voice_emotion_result = detect_voice_emotion(audio_signal, sample_rate, pitch_baseline=baseline)
        voice_emotion_context = format_voice_emotion_context(voice_emotion_result)
        voice_emotion_label = str(voice_emotion_result.get("label", "")).lower()
        voice_emotion_confidence = float(voice_emotion_result.get("score", 0.0))
    else:
        voice_emotion_context = "Detected voice emotion: unknown (0.00). Top scores: none."
        voice_emotion_label = ""
        voice_emotion_confidence = 0.0

    # Merge text and voice emotion signals conservatively:
    # - Prefer text by default (it is semantically richer).
    # - If text is weak/unknown and voice is confident, use voice for mode/tone.
    fused_emotion_label = emotion_label
    fused_emotion_confidence = emotion_confidence
    if (not fused_emotion_label or fused_emotion_confidence < 0.55) and voice_emotion_confidence >= 0.55:
        fused_emotion_label = voice_emotion_label
        fused_emotion_confidence = voice_emotion_confidence
    elif voice_emotion_label and voice_emotion_label == fused_emotion_label:
        fused_emotion_confidence = max(
            fused_emotion_confidence,
            min(0.95, (fused_emotion_confidence + voice_emotion_confidence) / 2.0 + 0.10),
        )

    if energy_level == "high" and zcr_level == "high":
        directive = "User may sound agitated. Use calming, grounding language."
    elif energy_level == "low" and zcr_level == "low":
        directive = "User may sound low energy. Use gentle, supportive tone."
    else:
        directive = ""

    rag_system_prompt = f"""
{directive}

You are a supportive mental health assistant.

User Profile:
- Name: {profile.get("name", "Unknown")}
- Stress tendency: {profile.get("stress_flag", False)}
- Relaxation methods: {profile.get("relaxation_methods", "unknown")}
- Pitch baseline: {baseline}
- Current pitch comparison: {mood_hint}

Voice Indicators:
- Energy level: {energy_level}
- ZCR level: {zcr_level}

Interpretation rules:
- Use indicators only as tone guidance.
- Do NOT assume diagnosis.
- Do NOT hallucinate emotional state.
- Only respond to explicit user content.

Instruction:
- Personalize using stored name.
- Use pitch comparison only as tone guidance.
- Do NOT diagnose mood from pitch alone.

Context:
{retrieved_context}

User text:
{user_text.strip()}
""".strip()

    decision = decide_action(fused_emotion_label, user_text, fused_emotion_confidence)

    if decision["mode"] == "crisis":
        return build_structured_crisis_response(user_text)

    # Crisis mode uses a dedicated system prompt to keep safety language caring and non-robotic.
    if decision["mode"] == "crisis":
        messages.append({"role": "system", "content": SYSTEM_PROMPT_CRISIS})
    else:
        effective_emotion = _normalize_emotion(fused_emotion_label)
        messages.append({"role": "system", "content": build_system_prompt(effective_emotion)})

    messages.append({"role": "system", "content": rag_system_prompt})
    if profile.get("stress_flag") and any(k in user_text.lower() for k in ("stress", "anxious", "overwhelmed")):
        messages.append(
            {
                "role": "system",
                "content": (
                    "The user has a known stress tendency. "
                    f"When useful, suggest their known relaxation methods: {profile.get('relaxation_methods', '')}."
                ),
            }
        )

    # Guardrail: when recent assistant turns were questions, force this turn to be reflection + suggestion only.
    if _recent_assistant_questions(context):
        messages.append(
            {
                "role": "system",
                "content": (
                    "Recent assistant turns were question-heavy. "
                    "For this turn, do not ask any question. "
                    "Give acknowledgment, reflection, and at least one actionable suggestion."
                ),
            }
        )

    # Soft guidance from classifiers: useful for tone tuning, but not authoritative.
    messages.append(
        {
            "role": "system",
            "content": (
                "Internal signal from RoBERTa emotion classifier. "
                "Use as a soft hint, not as ground truth. "
                f"{emotion_context}"
            ),
        }
    )
    messages.append(
        {
            "role": "system",
            "content": (
                "Internal signal from voice emotion heuristic. "
                "Use as a soft hint, not as ground truth. "
                f"{voice_emotion_context}"
            ),
        }
    )
    messages.append(
        {
            "role": "system",
            "content": (
                f"Selected response mode: {decision['mode']}. "
                "Follow the mode-specific instructions below.\n\n"
                f"{decision['prompt']}"
            ),
        }
    )

    for item in context or []:
        role, content = _parse_context_item(item)
        if content:
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_text.strip()})

    reply = _safe_chat_completion(messages)
    if not reply:
        return _fallback_support_reply(user_text)

    # If the assistant offers a breathing exercise, store pending state.
    if detect_breathing_offer(reply):
        STATE.set_pending("breathing")

    return reply


def main() -> None:
    user_text, audio_signal, sample_rate = _capture_user_turn()
    if not user_text:
        print("No speech detected. Exiting.")
        raise SystemExit(0)

    # Example call with empty history. Caller can pass real prior turns when available.
    reply = generate_response(
        user_text=user_text,
        context=[],
        audio_signal=audio_signal,
        sample_rate=sample_rate,
        user_profile=None,
    )
    print(reply)

    try:
        speak(reply)
    except Exception:
        pass


if __name__ == "__main__":
    main()

