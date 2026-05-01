"""
EPIDRA — Gemini Service
========================
Encapsulates all interaction with the Google Gemini API
using the official `google-generativeai` SDK.

Model : gemini-1.5-flash
Purpose: Answer general disease / public-health questions
         with short, accurate, and safe responses.
"""

import os
import time
import hashlib
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

# ── Load environment ────────────────────────────────────────────
load_dotenv()  # reads backend/.env

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    print("  ⚠️  GEMINI_API_KEY not found — Gemini service will be disabled.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("  ✅ Gemini API key loaded successfully.")


# ── Model initialisation ───────────────────────────────────────
_model: Optional[genai.GenerativeModel] = None

SYSTEM_PROMPT = (
    "You are a public health assistant for EPIDRA "
    "(Epidemic Risk Intelligence for Disease Response and Analysis).\n\n"
    "RULES:\n"
    "- Provide accurate, simple, safe information about water-borne and vector-borne diseases.\n"
    "- Focus on: cholera, dengue, typhoid, malaria, chikungunya, leptospirosis, and similar diseases.\n"
    "- Do NOT exaggerate risks or create panic.\n"
    "- Do NOT provide medical diagnosis or prescribe medication.\n"
    "- Keep responses concise (max 4-5 sentences).\n"
    "- Always recommend consulting a healthcare professional for symptoms.\n"
    "- If asked about something unrelated to public health or diseases, politely redirect.\n"
    "- Do NOT make up statistics or data.\n"
)

GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.3,
    max_output_tokens=300,
    top_p=0.8,
)

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]


def _get_model() -> genai.GenerativeModel:
    """Lazy-init the GenerativeModel singleton."""
    global _model
    if _model is None:
        _model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT,
            generation_config=GENERATION_CONFIG,
            safety_settings=SAFETY_SETTINGS,
        )
    return _model


# ── Response cache (in-memory) ──────────────────────────────────
_cache: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 3600  # 1 hour


def _cache_key(query: str, lang: str) -> str:
    return hashlib.md5(f"{query}:{lang}".encode()).hexdigest()


# ── Public API ──────────────────────────────────────────────────

def get_gemini_response(query: str, lang: str = "en") -> Optional[str]:
    """
    Send *query* to Gemini and return a short, safe text answer.

    Parameters
    ----------
    query : str
        The user's question (e.g. "What is cholera?").
    lang  : str
        Language code — "en", "hi", or "as".

    Returns
    -------
    str | None
        Gemini's text reply, or None if the service is
        unavailable / an error occurred.
    """
    if not GEMINI_API_KEY:
        return None

    # 1. Check cache
    key = _cache_key(query, lang)
    if key in _cache:
        cached_text, cached_at = _cache[key]
        if time.time() - cached_at < _CACHE_TTL:
            return cached_text

    # 2. Build language instruction
    lang_instruction = ""
    if lang == "hi":
        lang_instruction = "\nIMPORTANT: Respond in Hindi (Devanagari script)."
    elif lang == "as":
        lang_instruction = "\nIMPORTANT: Respond in Assamese (Bengali/Assamese script)."

    prompt = f"{lang_instruction}\n\nUser question: {query}"

    # 3. Call Gemini
    try:
        model = _get_model()
        response = model.generate_content(prompt)

        if response and response.text:
            text = response.text.strip()
            _cache[key] = (text, time.time())
            return text

    except Exception as exc:
        print(f"  [Gemini Service] Error: {exc}")

    return None


# ── Convenience wrappers used by the chatbot router ─────────────

def is_gemini_available() -> bool:
    """Return True when the Gemini API key is configured."""
    return bool(GEMINI_API_KEY)
