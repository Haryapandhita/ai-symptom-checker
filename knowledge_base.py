"""
knowledge_base.py
=================
The medical "knowledge base" for the AI Symptom Checker.

This file contains three kinds of data:

1. TRIAGE_LEVELS   - the four care levels the app can recommend, ordered from
                     least to most urgent, with display info.
2. CONDITIONS      - a small, hand-curated list of common conditions. Each one
                     has the symptoms typically associated with it, a default
                     triage level, and general self-care / next-step advice.
3. SYNONYMS        - everyday phrases people use ("throwing up", "can't
                     breathe") mapped to the clinical term used in CONDITIONS
                     ("vomiting", "shortness of breath").
4. RED_FLAGS       - "never miss" warning signs. If any of these appear, the
                     app ALWAYS recommends emergency care, no matter what the
                     matching algorithm says. This is a deliberate safety rule.

IMPORTANT (data-centric note):
    This dataset was written by hand for an educational project. It is NOT
    clinically validated, is not exhaustive, and reflects general adult
    presentations only (children, pregnancy, older adults and people with
    chronic illness can present very differently). A real product would need
    a dataset built and reviewed by clinicians and validated on real cases.
"""

# ---------------------------------------------------------------------------
# 1. Triage levels
# ---------------------------------------------------------------------------
# The "rank" lets the code compare levels: a higher rank = more urgent.
# This is used to escalate (never downgrade) when safety rules fire.
TRIAGE_LEVELS = {
    "Self-Care": {
        "rank": 0,
        "color": "#2e7d32",  # green
        "icon": "🟢",
        "action": "Likely manageable at home. Rest, fluids, and over-the-counter "
                  "remedies may help. See a doctor if symptoms get worse or "
                  "do not improve within a few days.",
    },
    "Primary Care": {
        "rank": 1,
        "color": "#1565c0",  # blue
        "icon": "🔵",
        "action": "Book an appointment with your regular doctor or a telehealth "
                  "service within the next few days.",
    },
    "Urgent Care": {
        "rank": 2,
        "color": "#ef6c00",  # orange
        "icon": "🟠",
        "action": "Get seen today: visit an urgent care clinic or contact your "
                  "doctor for a same-day appointment.",
    },
    "Emergency": {
        "rank": 3,
        "color": "#c62828",  # red
        "icon": "🔴",
        "action": "Call emergency services (911 in the US) or go to the nearest "
                  "emergency department now.",
    },
}

# Helper list sorted from least to most urgent, e.g. for escalating one step.
TRIAGE_ORDER = sorted(TRIAGE_LEVELS, key=lambda level: TRIAGE_LEVELS[level]["rank"])


# ---------------------------------------------------------------------------
# 2. Conditions
# ---------------------------------------------------------------------------
# Each condition is a dictionary:
#   name      - display name
#   symptoms  - list of symptom phrases (use the clinical wording that the
#               SYNONYMS table below normalizes everyday language into)
#   key_symptoms - (Emergency conditions only) hallmark signs. An Emergency
#               condition is only allowed to drive an "Emergency" result if at
#               least one hallmark sign is present; otherwise the engine caps
#               the result at Urgent Care. This prevents, e.g., "headache and
#               fever" alone from being labelled meningitis-level emergency.
#   triage    - default triage level when this condition is the best match
#   advice    - short, general next-step advice
CONDITIONS = [
    {
        "name": "Common Cold",
        "symptoms": ["runny nose", "sneezing", "sore throat", "cough",
                     "nasal congestion", "mild fever", "mild headache"],
        "triage": "Self-Care",
        "advice": "Rest, drink fluids, and consider saline nasal spray or "
                  "throat lozenges. Most colds clear in 7-10 days.",
    },
    {
        "name": "Influenza (Flu)",
        "symptoms": ["fever", "chills", "body aches", "fatigue", "cough",
                     "headache", "sore throat", "muscle pain"],
        "triage": "Primary Care",
        "advice": "Rest and fluids. A doctor may prescribe antivirals if seen "
                  "within 48 hours of symptoms starting, especially for "
                  "high-risk groups.",
    },
    {
        "name": "COVID-19",
        "symptoms": ["fever", "cough", "loss of taste", "loss of smell",
                     "fatigue", "shortness of breath", "sore throat",
                     "body aches"],
        "triage": "Primary Care",
        "advice": "Consider taking a COVID-19 test and limit contact with "
                  "others. Seek care if breathing becomes difficult.",
    },
    {
        "name": "Strep Throat",
        "symptoms": ["sore throat", "fever", "swollen lymph nodes",
                     "difficulty swallowing", "white patches on tonsils",
                     "red tonsils"],
        "triage": "Primary Care",
        "advice": "See a doctor for a rapid strep test; bacterial infections "
                  "may need antibiotics.",
    },
    {
        "name": "Seasonal Allergies",
        "symptoms": ["sneezing", "itchy eyes", "watery eyes", "runny nose",
                     "nasal congestion", "itchy throat"],
        "triage": "Self-Care",
        "advice": "Over-the-counter antihistamines and avoiding triggers "
                  "(pollen, dust) usually help.",
    },
    {
        "name": "Sinusitis",
        "symptoms": ["facial pain", "facial pressure", "nasal congestion",
                     "thick nasal discharge", "headache", "reduced smell"],
        "triage": "Self-Care",
        "advice": "Saline rinses, steam and fluids. See a doctor if it lasts "
                  "more than 10 days or you develop a high fever.",
    },
    {
        "name": "Migraine",
        "symptoms": ["throbbing headache", "headache", "nausea",
                     "sensitivity to light", "sensitivity to sound",
                     "visual aura", "vomiting"],
        "triage": "Primary Care",
        "advice": "Rest in a dark, quiet room. A doctor can help with "
                  "treatment and prevention if migraines are frequent.",
    },
    {
        "name": "Tension Headache",
        "symptoms": ["headache", "head pressure", "neck pain", "stress",
                     "tight band around head"],
        "triage": "Self-Care",
        "advice": "Rest, hydration, and over-the-counter pain relief. "
                  "Managing stress and posture can help prevent them.",
    },
    {
        "name": "Gastroenteritis (Stomach Bug)",
        "symptoms": ["diarrhea", "vomiting", "nausea", "abdominal cramps",
                     "abdominal pain", "mild fever", "loss of appetite"],
        "triage": "Self-Care",
        "advice": "Small sips of fluids or oral rehydration solution. Seek care "
                  "if you cannot keep fluids down or see blood.",
    },
    {
        "name": "Dehydration",
        "symptoms": ["thirst", "dry mouth", "dark urine", "dizziness",
                     "fatigue", "reduced urination"],
        "triage": "Self-Care",
        "advice": "Drink water or an oral rehydration solution. Seek care if "
                  "you feel confused or faint.",
    },
    {
        "name": "Urinary Tract Infection (UTI)",
        "symptoms": ["burning urination", "painful urination",
                     "frequent urination", "urgent urination", "cloudy urine",
                     "pelvic pain", "lower abdominal pain"],
        "triage": "Primary Care",
        "advice": "See a doctor; UTIs usually need antibiotics. Drink plenty "
                  "of water.",
    },
    {
        "name": "Kidney Stone",
        "symptoms": ["flank pain", "back pain",
                     "blood in urine", "pain radiating to groin", "nausea",
                     "vomiting"],
        "triage": "Urgent Care",
        "advice": "Get assessed today. Go to the emergency department if pain "
                  "is unbearable or you have a fever.",
    },
    {
        "name": "Appendicitis",
        "symptoms": ["lower right abdominal pain", "abdominal pain",
                     "nausea", "vomiting", "fever", "loss of appetite",
                     "pain worse when moving"],
        "key_symptoms": ["lower right abdominal pain", "pain worse when moving"],
        "triage": "Emergency",
        "advice": "Appendicitis can be serious. Go to the emergency "
                  "department for evaluation.",
    },
    {
        "name": "Ear Infection",
        "symptoms": ["ear pain", "fever", "hearing loss", "ear drainage",
                     "ear fullness"],
        "triage": "Primary Care",
        "advice": "A doctor can examine the ear and decide whether treatment "
                  "is needed.",
    },
    {
        "name": "Conjunctivitis (Pink Eye)",
        "symptoms": ["red eye", "itchy eyes", "eye discharge",
                     "crusty eyelids", "watery eyes"],
        "triage": "Primary Care",
        "advice": "Avoid touching your eyes and wash hands often. See a doctor "
                  "if there is pain or vision changes.",
    },
    {
        "name": "Asthma Flare-up",
        "symptoms": ["wheezing", "shortness of breath", "chest tightness",
                     "cough", "trouble sleeping"],
        "triage": "Urgent Care",
        "advice": "Use your rescue inhaler if you have one. Seek emergency "
                  "care if it does not help or breathing is very hard.",
    },
    {
        "name": "Pneumonia",
        "symptoms": ["cough with phlegm", "cough", "fever", "chills",
                     "shortness of breath", "chest pain when breathing",
                     "fatigue"],
        "triage": "Urgent Care",
        "advice": "Get seen today; pneumonia may need antibiotics and a chest "
                  "examination.",
    },
    {
        "name": "Ankle or Joint Sprain",
        "symptoms": ["joint pain", "swelling", "bruising", "injury",
                     "difficulty walking", "limited movement"],
        "triage": "Self-Care",
        "advice": "Rest, ice, compression and elevation (RICE). See a doctor "
                  "if you cannot bear weight.",
    },
    {
        "name": "Contact Dermatitis / Skin Rash",
        "symptoms": ["itchy rash", "rash", "skin redness", "blisters",
                     "dry skin", "itching"],
        "triage": "Self-Care",
        "advice": "Avoid the trigger, use gentle moisturizers or "
                  "over-the-counter hydrocortisone. See a doctor if it spreads.",
    },
    {
        "name": "Panic / Anxiety Attack",
        "symptoms": ["racing heart", "palpitations", "sweating", "trembling",
                     "fear", "anxiety", "shortness of breath", "dizziness"],
        "triage": "Primary Care",
        "advice": "Slow breathing can help in the moment. A doctor or mental "
                  "health professional can help with ongoing anxiety. Chest "
                  "pain should always be checked urgently.",
    },
    {
        "name": "Heart Attack (Possible)",
        "symptoms": ["chest pain", "chest pressure", "pain in left arm",
                     "jaw pain", "shortness of breath", "sweating", "nausea"],
        "key_symptoms": ["chest pain", "chest pressure", "pain in left arm",
                         "jaw pain"],
        "triage": "Emergency",
        "advice": "Call emergency services immediately. Do not drive yourself.",
    },
    {
        "name": "Stroke (Possible)",
        "symptoms": ["face drooping", "arm weakness", "slurred speech",
                     "confusion", "numbness on one side", "sudden headache",
                     "vision loss"],
        "key_symptoms": ["face drooping", "arm weakness", "slurred speech",
                         "numbness on one side", "vision loss"],
        "triage": "Emergency",
        "advice": "Call emergency services immediately. Note the time "
                  "symptoms started. Remember BE FAST: Balance, Eyes, Face, "
                  "Arms, Speech, Time.",
    },
    {
        "name": "Meningitis (Possible)",
        "symptoms": ["fever", "stiff neck", "headache",
                     "sensitivity to light", "confusion", "rash", "vomiting"],
        "key_symptoms": ["stiff neck", "confusion"],
        "triage": "Emergency",
        "advice": "Meningitis is a medical emergency. Go to the emergency "
                  "department immediately.",
    },
    {
        "name": "Severe Allergic Reaction (Anaphylaxis)",
        "symptoms": ["throat swelling", "difficulty breathing", "hives",
                     "swollen lips", "swollen face", "dizziness"],
        "key_symptoms": ["throat swelling", "difficulty breathing",
                         "swollen lips", "swollen face"],
        "triage": "Emergency",
        "advice": "Use an epinephrine auto-injector if available and call "
                  "emergency services.",
    },
]


# ---------------------------------------------------------------------------
# 3. Synonyms (everyday language -> clinical wording)
# ---------------------------------------------------------------------------
# People rarely type "emesis"; they type "throwing up". This table lets the
# NLP pipeline translate common phrasing into the terms used above.
# Longer phrases are replaced first (see symptom_engine.normalize_text), so
# "stomach ache" is handled before "ache".
SYNONYMS = {
    # breathing
    "can't breathe": "difficulty breathing",
    "cant breathe": "difficulty breathing",
    "cannot breathe": "difficulty breathing",
    "trouble breathing": "difficulty breathing",
    "hard to breathe": "difficulty breathing",
    "struggling to breathe": "difficulty breathing",
    "short of breath": "shortness of breath",
    "out of breath": "shortness of breath",
    "breathless": "shortness of breath",
    "winded": "shortness of breath",
    # stomach / digestion
    "throwing up": "vomiting",
    "threw up": "vomiting",
    "puking": "vomiting",
    "vomit": "vomiting",
    "feel sick to my stomach": "nausea",
    "nauseous": "nausea",
    "queasy": "nausea",
    "stomach ache": "abdominal pain",
    "stomachache": "abdominal pain",
    "tummy ache": "abdominal pain",
    "belly pain": "abdominal pain",
    "stomach pain": "abdominal pain",
    "stomach cramps": "abdominal cramps",
    "runs": "diarrhea",
    "loose stools": "diarrhea",
    "diarrhoea": "diarrhea",
    "not hungry": "loss of appetite",
    "no appetite": "loss of appetite",
    # temperature
    "high temperature": "fever",
    "temperature": "fever",
    "feverish": "fever",
    "running a fever": "fever",
    "shivering": "chills",
    # general
    "tired": "fatigue",
    "exhausted": "fatigue",
    "worn out": "fatigue",
    "no energy": "fatigue",
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "light headed": "dizziness",
    "achy": "body aches",
    "aching all over": "body aches",
    "body ache": "body aches",
    "muscle aches": "muscle pain",
    "sweaty": "sweating",
    "heart racing": "racing heart",
    "heart pounding": "palpitations",
    "panicking": "anxiety",
    "anxious": "anxiety",
    "shaking": "trembling",
    # head / face
    "migraine": "throbbing headache",
    "pounding headache": "throbbing headache",
    "head hurts": "headache",
    "head ache": "headache",
    "stuffy nose": "nasal congestion",
    "blocked nose": "nasal congestion",
    "congested": "nasal congestion",
    "congestion": "nasal congestion",
    "can't smell": "loss of smell",
    "cant smell": "loss of smell",
    "can't taste": "loss of taste",
    "cant taste": "loss of taste",
    "light hurts my eyes": "sensitivity to light",
    "light sensitivity": "sensitivity to light",
    "pink eye": "red eye",
    "earache": "ear pain",
    "ear ache": "ear pain",
    "scratchy throat": "sore throat",
    "throat hurts": "sore throat",
    "hurts to swallow": "difficulty swallowing",
    "painful swallowing": "difficulty swallowing",
    "swollen glands": "swollen lymph nodes",
    # urinary
    "burns when i pee": "burning urination",
    "burning when i pee": "burning urination",
    "burning when peeing": "burning urination",
    "hurts to pee": "painful urination",
    "peeing a lot": "frequent urination",
    "pee a lot": "frequent urination",
    "blood in my pee": "blood in urine",
    "pee": "urine",
    # chest / heart
    "chest hurts": "chest pain",
    "chest tight": "chest tightness",
    "tight chest": "chest tightness",
    "pressure in my chest": "chest pressure",
    # skin
    "itchy skin": "itching",
    "bumps": "hives",
    "welts": "hives",
    # musculoskeletal
    "twisted my ankle": "injury joint pain",
    "rolled my ankle": "injury joint pain",
    "sprained my ankle": "injury joint pain",
    "swollen ankle": "swelling joint pain",
    "swollen knee": "swelling joint pain",
    # identity entries: already-clinical phrases that contain a shorter
    # synonym key. Because the longest phrase wins, these stop e.g.
    # "nasal congestion" from becoming "nasal nasal congestion".
    "nasal congestion": "nasal congestion",
}


# ---------------------------------------------------------------------------
# 4. Red flags (safety override)
# ---------------------------------------------------------------------------
# Each red flag has:
#   any_of   - list of phrases; the flag fires if ANY phrase is present
#              (after normalization and negation removal).
#   all_of   - optional list of phrases that must ALL be present (used for
#              combinations like fever + stiff neck).
#   message  - what to tell the user.
# Matching is done on the normalized text, so synonyms already apply.
RED_FLAGS = [
    {
        "any_of": ["chest pain", "chest pressure"],
        "message": "Chest pain or pressure can be a sign of a heart attack.",
    },
    {
        "any_of": ["difficulty breathing", "turning blue", "blue lips"],
        "message": "Severe difficulty breathing needs emergency assessment.",
    },
    {
        "any_of": ["face drooping", "slurred speech", "numbness on one side",
                   "weakness on one side", "sudden confusion"],
        "message": "These can be warning signs of a stroke. Note the time "
                   "symptoms began.",
    },
    {
        "any_of": ["worst headache", "sudden severe headache",
                   "thunderclap headache"],
        "message": "A sudden, extremely severe headache can signal bleeding "
                   "in the brain.",
    },
    {
        "all_of": ["fever", "stiff neck"],
        "message": "Fever with a stiff neck can be a sign of meningitis.",
    },
    {
        "any_of": ["throat swelling", "swollen tongue", "swollen lips"],
        "message": "Swelling of the throat, tongue or lips may be a severe "
                   "allergic reaction.",
    },
    {
        "any_of": ["coughing up blood", "vomiting blood", "severe bleeding",
                   "bleeding heavily", "won't stop bleeding"],
        "message": "Significant bleeding needs emergency care.",
    },
    {
        "any_of": ["fainted", "passed out", "unconscious", "seizure",
                   "unresponsive"],
        "message": "Loss of consciousness or a seizure needs emergency "
                   "assessment.",
    },
    {
        "any_of": ["suicidal", "kill myself", "end my life", "want to die",
                   "hurt myself", "self harm"],
        "message": "You deserve support right now. In the US, call or text "
                   "988 (Suicide & Crisis Lifeline). Outside the US, contact "
                   "your local emergency number or crisis line.",
    },
]

# Words that suggest a symptom is more intense than usual. If present, the
# engine escalates the triage level by one step (capped at Urgent Care; only a
# red flag or an Emergency-level match can produce "Emergency").
SEVERITY_WORDS = ["severe", "unbearable", "excruciating", "worst",
                  "extreme", "intense", "agonizing"]
