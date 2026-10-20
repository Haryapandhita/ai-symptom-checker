"""
llm_assistant.py
================
OPTIONAL large-language-model (LLM) "second opinion" using Google Gemini.

The core app works fully offline with the NLP engine in symptom_engine.py.
This module adds an extra, clearly labelled explanation written by an LLM,
which is better at understanding unusual phrasing and combinations of
symptoms than keyword matching.

Design decisions (safety first):
  * The LLM never overrides the rule-based triage. Red-flag rules in
    symptom_engine.py always run and always win; the LLM text is shown
    separately as supporting information.
  * The prompt tells the model to be cautious, to list possibilities rather
    than a single diagnosis, and to never claim certainty.
  * The API key is read from an environment variable or Streamlit secrets -
    it is NEVER written in the code, so it cannot leak through GitHub.

Setup:
    1. Get a free API key from Google AI Studio (aistudio.google.com).
    2. Set it before running the app:
           macOS/Linux:  export GEMINI_API_KEY="your-key"
           Windows:      set GEMINI_API_KEY=your-key
    3. Optionally choose a model:  export GEMINI_MODEL="gemini-2.5-flash"
"""

import os

DEFAULT_MODEL = "gemini-2.5-flash"

# Prompt engineering: a clear role, strict output structure and safety rules.
PROMPT_TEMPLATE = """You are a cautious medical triage assistant used in a student
educational project. You are NOT a doctor and you must not give a definitive
diagnosis.

A person describes their symptoms below. A rule-based system has already
suggested the triage level "{rule_triage}".

Respond in English using exactly this structure (Markdown):
**Possible conditions:** 2-3 possibilities, most likely first, one line each.
**Suggested triage level:** one of Emergency / Urgent Care / Primary Care / Self-Care, with a one-sentence reason.
**Questions a clinician might ask:** 2-3 short questions.
**Self-care tips (if appropriate):** 1-3 short bullet points, or "Not appropriate - seek care."

Rules:
- If any symptom could be life-threatening, choose Emergency.
- If you disagree with the rule-based level, say so and explain briefly.
- Keep the whole answer under 180 words.
- Do not ask for or repeat personal identifying information.

Symptoms: \"\"\"{symptoms}\"\"\"
"""


def get_api_key():
    """Look for the key in an environment variable, then in Streamlit secrets."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:  # no secrets file configured - that's fine
        return None


def is_available() -> bool:
    """True when an API key is configured and the SDK is installed."""
    if not get_api_key():
        return False
    try:
        from google import genai  # noqa: F401
        return True
    except ImportError:
        return False


def get_llm_assessment(symptoms: str, rule_triage: str) -> str:
    """Ask Gemini for a cautious second opinion.

    Returns Markdown text. On any problem (no key, network error, model name
    changed...) it returns a friendly error message instead of crashing the
    app.
    """
    api_key = get_api_key()
    if not api_key:
        return "LLM mode is off: no GEMINI_API_KEY is configured."

    try:
        from google import genai
    except ImportError:
        return "LLM mode needs the 'google-genai' package: pip install google-genai"

    model_name = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    prompt = PROMPT_TEMPLATE.format(symptoms=symptoms[:1000], rule_triage=rule_triage)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        text = (response.text or "").strip()
        return text or "The model returned an empty answer. Please try again."
    except Exception as error:  # show the reason, but keep the app running
        return (f"Could not get an answer from Gemini ({type(error).__name__}). "
                f"Check your API key, internet connection, or try another model "
                f"via the GEMINI_MODEL setting. Details: {error}")
