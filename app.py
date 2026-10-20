"""
app.py
======
Streamlit web interface for the AI Symptom Checker & Triage Tool.

Run locally with:
    streamlit run app.py

The interface:
  1. Shows a medical disclaimer up front.
  2. Lets the user type symptoms (or pick an example).
  3. Calls symptom_engine.analyze_symptoms() and shows:
       - a colour-coded triage recommendation and the reasons behind it
       - red-flag warnings (if any)
       - up to three possible conditions with confidence and matched symptoms
       - a "behind the scenes" view of the NLP pipeline (for transparency)
  4. Optionally asks Google Gemini for an LLM second opinion.

Privacy: nothing the user types is saved, logged or sent anywhere, except to
Gemini when the user explicitly switches on the optional LLM mode.
"""

import streamlit as st

import llm_assistant
from knowledge_base import TRIAGE_LEVELS
from symptom_engine import analyze_symptoms

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺", layout="centered")

DISCLAIMER = (
    "**Disclaimer:** This tool is a student project for educational purposes "
    "only. It is **not** a medical device and is **not** a substitute for "
    "professional medical advice, diagnosis, or treatment. If you think you may "
    "have a medical emergency, call 911 (US) or your local emergency number "
    "immediately."
)

# Example inputs that users can load with one click.
EXAMPLES = {
    "Cold-like": "Runny nose, sneezing and a scratchy throat for two days, no fever",
    "Flu-like": "Fever, chills, body aches and I'm exhausted",
    "Urinary": "It burns when I pee and I need to pee a lot",
    "Chest pain": "Crushing chest pain going down my left arm and I'm sweaty",
}


def load_example(text: str):
    """Callback for the example buttons: put the text into the input box."""
    st.session_state["symptoms"] = text


# ---------------------------------------------------------------------------
# Sidebar: about, how it works, settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "An AI-driven symptom checker that uses **natural language processing "
        "(NLP)** to read free-text symptoms, rank possible conditions, and "
        "suggest a **triage level**: Emergency, Urgent Care, Primary Care or "
        "Self-Care."
    )
    st.subheader("How it works")
    st.markdown(
        "1. Everyday words are mapped to clinical terms\n"
        "2. Negated symptoms (\"no fever\") are removed\n"
        "3. Text is tokenized, stop words removed, words stemmed\n"
        "4. TF-IDF + cosine similarity ranks 24 conditions\n"
        "5. Safety rules decide the triage level - red-flag "
        "warning signs **always** trigger Emergency"
    )
    st.subheader("Settings")
    llm_ready = llm_assistant.is_available()
    use_llm = st.toggle(
        "LLM second opinion (Gemini)",
        value=False,
        disabled=not llm_ready,
        help="Requires a GEMINI_API_KEY. Your symptom text is sent to Google "
             "only when this is switched on.",
    )
    if not llm_ready:
        st.caption("LLM mode is off - set GEMINI_API_KEY to enable it (see README).")
    st.subheader("Privacy")
    st.caption("Nothing you type is stored or logged by this app.")


# ---------------------------------------------------------------------------
# Main area: input
# ---------------------------------------------------------------------------
st.title("🩺 AI Symptom Checker & Triage")
st.warning(DISCLAIMER)

st.write("**Try an example:**")
example_columns = st.columns(len(EXAMPLES))
for column, (label, text) in zip(example_columns, EXAMPLES.items()):
    column.button(label, on_click=load_example, args=(text,), width="stretch")

symptoms = st.text_area(
    "Describe your symptoms in your own words",
    key="symptoms",
    height=120,
    max_chars=1000,
    placeholder="e.g. I've had a sore throat and fever since yesterday, but no cough",
)

analyze_clicked = st.button("Analyze symptoms", type="primary", width="stretch")

# ---------------------------------------------------------------------------
# Main area: results
# ---------------------------------------------------------------------------
if analyze_clicked:
    if not symptoms.strip():
        st.info("Please describe at least one symptom.")
        st.stop()

    result = analyze_symptoms(symptoms)
    level = TRIAGE_LEVELS[result.triage]

    # --- Triage card ---------------------------------------------------------
    st.subheader("Suggested triage level")
    st.markdown(
        f"""
        <div style="border-left: 8px solid {level['color']}; padding: 12px 16px;
                    border-radius: 6px; background: rgba(127,127,127,0.08);">
            <div style="font-size: 1.4rem; font-weight: 700; color: {level['color']};">
                {level['icon']} {result.triage}
            </div>
            <div style="margin-top: 6px;">{level['action']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Red flags -------------------------------------------------------------
    for message in result.red_flags:
        st.error(f"⚠️ **Warning sign detected:** {message}")

    # --- Why this level? --------------------------------------------------
    with st.expander("Why this triage level?", expanded=True):
        for reason in result.reasons:
            st.write(f"- {reason}")

    # --- Possible conditions ----------------------------------------------
    st.subheader("Possible conditions")
    if not result.conditions:
        st.write(
            "No condition in this tool's knowledge base clearly matched. "
            "Please speak to a healthcare provider."
        )
    for rank, cond in enumerate(result.conditions, start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {cond.name}** · confidence: **{cond.confidence}**")
            st.progress(min(cond.score, 1.0))
            st.caption("Matched symptoms: " + ", ".join(cond.matched_symptoms))
            st.write(cond.advice)

    # --- Transparency: what the NLP pipeline did -----------------------------
    with st.expander("Behind the scenes: NLP pipeline"):
        st.write("**Normalized text:**", result.normalized_text)
        st.write("**Ignored (negated) symptoms:**",
                 ", ".join(result.negated_phrases) or "none")
        st.write("**Processed tokens (stemmed):**", ", ".join(result.tokens) or "none")
        st.write("**Words not in the knowledge base:**",
                 ", ".join(result.unrecognized_terms) or "none")
        st.write("**Severity words detected:**", "yes" if result.severity_detected else "no")

    # --- Optional LLM second opinion ------------------------------------------
    if use_llm:
        st.subheader("LLM second opinion (Gemini)")
        st.caption("Generated by a large language model. It may be wrong. "
                   "The triage level above is decided by the safety rules, not by the LLM.")
        with st.spinner("Asking Gemini..."):
            st.markdown(llm_assistant.get_llm_assessment(symptoms, result.triage))

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Built for Module 03 - AI in Healthcare. Knowledge base is hand-curated, "
    "not clinically validated, and covers common adult presentations only."
)
