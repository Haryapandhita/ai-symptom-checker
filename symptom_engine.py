"""
symptom_engine.py
=================
The "AI brain" of the Symptom Checker. It turns free-text symptoms into a
ranked list of possible conditions and a triage recommendation.

How it works (the NLP + matching pipeline)
------------------------------------------
    raw text
      │  1. clean_text()        lowercase, tidy punctuation
      ▼
      │  2. normalize_text()    everyday phrases -> clinical terms
      ▼                         ("throwing up" -> "vomiting")
      │  3. remove_negations()  drop symptoms the user says they DON'T have
      ▼                         ("headache but no fever" -> "headache")
      │  4. tokenize + stop-word removal + stemming (NLTK Porter stemmer)
      ▼
      │  5. TF-IDF vectors + cosine similarity (scikit-learn)
      ▼     compares the user's text against every condition profile
      │  6. symptom coverage    which listed symptoms were actually matched
      ▼
      │  7. triage rules        red-flag safety overrides, severity words,
      ▼                         and caution rules for serious conditions
    AnalysisResult

Why TF-IDF?
    TF-IDF (term frequency - inverse document frequency) gives *rare*,
    distinctive symptoms (e.g. "stiff neck", "loss of smell") more weight than
    very common ones (e.g. "fever", which appears in many conditions). That
    makes the ranking more discriminating than simply counting matching words,
    while still being fully explainable - important in healthcare.
"""

import re
from dataclasses import dataclass, field

from nltk.stem import PorterStemmer          # rule-based stemmer, no download needed
from nltk.tokenize import RegexpTokenizer     # regex tokenizer, no download needed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import (
    CONDITIONS,
    RED_FLAGS,
    SEVERITY_WORDS,
    SYNONYMS,
    TRIAGE_LEVELS,
    TRIAGE_ORDER,
)

# ---------------------------------------------------------------------------
# Configuration constants (tuned by hand on the example cases in tests/)
# ---------------------------------------------------------------------------
MAX_INPUT_CHARS = 1000      # protect against extremely long inputs
MIN_SCORE = 0.12            # below this a condition is not shown at all
TOP_K = 3                   # how many possible conditions to display
COSINE_WEIGHT = 0.7         # final score = 0.7 * similarity + 0.3 * coverage
COVERAGE_WEIGHT = 0.3
HALLMARK_PENALTY = 0.5      # score multiplier for serious conditions lacking a hallmark sign

# A custom stop-word list. We do NOT use a generic list here, because generic
# English lists remove words that matter medically (e.g. "side" in "numbness
# on one side", "back" in "back pain").
STOP_WORDS = {
    "i", "im", "ive", "me", "my", "myself", "we", "our", "you", "your", "he",
    "she", "it", "its", "they", "them", "a", "an", "the", "and", "or", "of",
    "to", "in", "on", "at", "for", "with", "from", "by", "as", "is", "am",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "feel", "feeling", "felt", "get", "got",
    "getting", "really", "very", "bit", "little", "lot", "also", "some",
    "since", "like", "just", "this", "that", "these", "those", "there",
    "today", "yesterday", "now", "kind", "sort", "keep", "think", "maybe",
    "so", "too", "quite", "pretty", "much", "all", "day", "days", "week",
    "weeks", "hour", "hours", "morning", "night", "s", "t", "m", "ve", "d",
}
# Severity words ("severe", "unbearable"...) are handled by a separate triage
# rule, so they are excluded from symptom matching. Otherwise "severe" alone
# would pull the ranking toward any condition that happens to list it.
STOP_WORDS |= set(SEVERITY_WORDS)

# Words that signal the user does NOT have a symptom. Everything after one of
# these, up to the end of the clause, is treated as negated.
NEGATION_PATTERN = re.compile(
    r"\b(no|not|without|denies|deny|never|don't have|do not have|"
    r"dont have|haven't had|havent had|no longer|free of)\b"
)

# Clause boundaries: punctuation and contrast words such as "but".
CLAUSE_SPLIT_PATTERN = re.compile(r"[.;,!?\n]|\b(?:but|however|although|though)\b")

_tokenizer = RegexpTokenizer(r"[a-z]+")
_stemmer = PorterStemmer()

# Build ONE regex for all synonyms, longest phrase first, so that
# "stomach ache" is matched before a shorter key could match part of it.
_SYNONYM_KEYS = sorted(SYNONYMS, key=len, reverse=True)
_SYNONYM_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in _SYNONYM_KEYS) + r")\b"
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ConditionMatch:
    """One possible condition and why it was suggested."""
    name: str
    score: float                     # 0..1 combined score
    confidence: str                  # "High" / "Moderate" / "Low"
    triage: str                      # default triage level for this condition
    advice: str
    matched_symptoms: list = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Everything the user interface needs to display."""
    triage: str                                       # final triage level
    reasons: list = field(default_factory=list)       # why that level was chosen
    red_flags: list = field(default_factory=list)     # safety messages that fired
    conditions: list = field(default_factory=list)    # list[ConditionMatch]
    # Transparency / "show your work" fields for the NLP pipeline view:
    normalized_text: str = ""
    negated_phrases: list = field(default_factory=list)
    tokens: list = field(default_factory=list)
    unrecognized_terms: list = field(default_factory=list)
    severity_detected: bool = False


# ---------------------------------------------------------------------------
# Step 1-3: text cleaning, normalization and negation handling
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, unify apostrophes and collapse extra whitespace."""
    text = text[:MAX_INPUT_CHARS].lower()
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    """Replace everyday expressions with the clinical wording used in the
    knowledge base (single pass, longest phrase wins)."""
    return _SYNONYM_PATTERN.sub(lambda m: SYNONYMS[m.group(1)], text)


def remove_negations(text: str):
    """Split text into clauses and drop the part of each clause that follows a
    negation word.

    Example:
        "headache and nausea, no fever but some chills"
        -> kept:    ["headache and nausea", "some chills"]
        -> negated: ["fever"]

    Returns (kept_clauses, negated_phrases).
    """
    kept, negated = [], []
    for clause in CLAUSE_SPLIT_PATTERN.split(text):
        clause = (clause or "").strip()
        if not clause:
            continue
        match = NEGATION_PATTERN.search(clause)
        if match:
            before = clause[:match.start()].strip()
            after = clause[match.end():].strip()
            if before:
                kept.append(before)
            if after:
                negated.append(after)
        else:
            kept.append(clause)
    return kept, negated


# ---------------------------------------------------------------------------
# Step 4: tokenization, stop-word removal, stemming
# ---------------------------------------------------------------------------
def preprocess_tokens(text: str) -> list:
    """Classic NLP preprocessing: tokenize -> remove stop words -> stem.

    Stemming maps word variants to one root so that "aches", "aching" and
    "ache" all match each other.
    """
    tokens = _tokenizer.tokenize(text.lower())
    return [_stemmer.stem(tok) for tok in tokens if tok not in STOP_WORDS]


def symptom_analyzer(document: str) -> list:
    """Custom analyzer for the TF-IDF vectorizer.

    A document is a set of segments separated by "|" (one symptom phrase or
    one user clause per segment). We produce unigrams plus bigrams *within*
    each segment, so a bigram like "chest pain" counts as one feature but we
    never create fake bigrams across two unrelated symptoms.
    """
    features = []
    for segment in document.split("|"):
        tokens = preprocess_tokens(segment)
        features.extend(tokens)
        features.extend(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    return features


# ---------------------------------------------------------------------------
# Step 5: build the TF-IDF model once, when the module is imported
# ---------------------------------------------------------------------------
# Each condition becomes one "document": its symptom phrases joined by "|".
_condition_documents = [" | ".join(c["symptoms"]) for c in CONDITIONS]

_vectorizer = TfidfVectorizer(analyzer=symptom_analyzer, sublinear_tf=True)
_condition_matrix = _vectorizer.fit_transform(_condition_documents)

# Every stemmed word the knowledge base knows about (used to report
# words the system did not recognize).
_KNOWN_STEMS = {stem for doc in _condition_documents for stem in preprocess_tokens(doc)}

# Hallmark ("key") symptoms of Emergency-level conditions, by condition name.
_KEY_SYMPTOMS = {c["name"]: c.get("key_symptoms", []) for c in CONDITIONS}

# Pre-stem every symptom phrase for the coverage check in step 6.
_STEMMED_SYMPTOMS = [
    [(phrase, set(preprocess_tokens(phrase))) for phrase in c["symptoms"]]
    for c in CONDITIONS
]


# ---------------------------------------------------------------------------
# Helpers for triage
# ---------------------------------------------------------------------------
def _contains_phrase(text: str, phrase: str) -> bool:
    """Whole-word phrase search, e.g. 'chest pain' but not 'chest painting'."""
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def _more_urgent(level_a: str, level_b: str) -> str:
    """Return whichever of two triage levels is more urgent."""
    if TRIAGE_LEVELS[level_a]["rank"] >= TRIAGE_LEVELS[level_b]["rank"]:
        return level_a
    return level_b


def _confidence_label(score: float, matched_count: int) -> str:
    """Translate a numeric score into plain language. A single matched
    symptom is never 'High' confidence - one symptom fits many conditions."""
    if matched_count < 2:
        return "Low"
    if score >= 0.45:
        return "High"
    if score >= 0.25:
        return "Moderate"
    return "Low"


def check_red_flags(text: str) -> list:
    """Return the safety message for every red-flag rule that fires."""
    messages = []
    for flag in RED_FLAGS:
        any_hit = any(_contains_phrase(text, p) for p in flag.get("any_of", []))
        all_list = flag.get("all_of", [])
        all_hit = bool(all_list) and all(_contains_phrase(text, p) for p in all_list)
        if any_hit or all_hit:
            messages.append(flag["message"])
    return messages


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def analyze_symptoms(raw_text: str) -> AnalysisResult:
    """Run the full pipeline on the user's symptom description.

    Raises ValueError if the input is empty.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Please describe at least one symptom.")

    # --- Steps 1-3: clean, normalize, handle negations --------------------
    text = normalize_text(clean_text(raw_text))
    kept_clauses, negated = remove_negations(text)
    kept_text = " | ".join(kept_clauses)          # "|" keeps clause boundaries
    user_stems = set(preprocess_tokens(kept_text))

    result = AnalysisResult(
        triage="Primary Care",
        normalized_text=text,
        negated_phrases=negated,
        tokens=preprocess_tokens(kept_text),
    )
    result.unrecognized_terms = sorted(user_stems - _KNOWN_STEMS)

    # --- Step 5: TF-IDF cosine similarity against every condition ---------
    user_vector = _vectorizer.transform([kept_text])
    similarities = cosine_similarity(user_vector, _condition_matrix)[0]

    # --- Step 6: symptom coverage + combined score -------------------------
    candidates = []
    for idx, condition in enumerate(CONDITIONS):
        matched = [phrase for phrase, stems in _STEMMED_SYMPTOMS[idx]
                   if stems and stems <= user_stems]
        coverage = len(matched) / len(condition["symptoms"])
        score = COSINE_WEIGHT * similarities[idx] + COVERAGE_WEIGHT * coverage
        # Emergency conditions with NO hallmark sign present are down-weighted,
        # so common symptoms (e.g. "fever") rank common illnesses first.
        hallmarks = _KEY_SYMPTOMS.get(condition["name"], [])
        if hallmarks and not any(s in hallmarks for s in matched):
            score *= HALLMARK_PENALTY
        if score >= MIN_SCORE and matched:
            candidates.append(ConditionMatch(
                name=condition["name"],
                score=round(float(score), 3),
                confidence=_confidence_label(score, len(matched)),
                triage=condition["triage"],
                advice=condition["advice"],
                matched_symptoms=matched,
            ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    result.conditions = candidates[:TOP_K]

    # --- Step 7: triage decision -------------------------------------------
    # 7a. Base level: the top condition's default triage.
    if result.conditions:
        top = result.conditions[0]
        triage = top.triage
        result.reasons.append(
            f"The best match, '{top.name}', usually calls for: {top.triage}.")
        # An Emergency condition only produces "Emergency" when at least one
        # of its hallmark signs is present AND at least two symptoms match.
        # Otherwise the evidence is too thin: cap at Urgent Care instead.
        if triage == "Emergency":
            hallmarks = _KEY_SYMPTOMS.get(top.name, [])
            has_hallmark = any(s in hallmarks for s in top.matched_symptoms)
            if not has_hallmark or len(top.matched_symptoms) < 2:
                triage = "Urgent Care"
                result.reasons.append(
                    "No hallmark sign of that serious condition was found, so "
                    "the level was set to Urgent Care rather than Emergency.")
        # One matching symptom fits many conditions, so it is not enough
        # evidence to recommend more than a regular doctor's visit.
        # (Severity words and red flags below can still raise the level.)
        if len(top.matched_symptoms) < 2 and \
                TRIAGE_LEVELS[triage]["rank"] > TRIAGE_LEVELS["Primary Care"]["rank"]:
            triage = "Primary Care"
            result.reasons.append(
                "Only one symptom matched, which is too little to judge "
                "urgency, so a regular doctor's visit is suggested.")
    else:
        triage = "Primary Care"
        result.reasons.append(
            "The symptoms did not clearly match any condition in this tool's "
            "knowledge base, so a doctor's opinion is recommended.")

    # 7b. Caution rule: if a serious condition is in the shortlist with
    # several matching symptoms, do not let the level stay below Urgent Care.
    for cond in result.conditions[1:]:
        if cond.triage == "Emergency" and len(cond.matched_symptoms) >= 3:
            if TRIAGE_LEVELS[triage]["rank"] < TRIAGE_LEVELS["Urgent Care"]["rank"]:
                triage = "Urgent Care"
                result.reasons.append(
                    f"'{cond.name}' cannot be ruled out, so prompt "
                    f"assessment is advised.")

    # 7c. Severity words ("severe", "unbearable"...) escalate one step,
    # but never beyond Urgent Care on their own.
    if any(_contains_phrase(kept_text, w) for w in SEVERITY_WORDS):
        result.severity_detected = True
        if TRIAGE_LEVELS[triage]["rank"] < TRIAGE_LEVELS["Urgent Care"]["rank"]:
            next_level = TRIAGE_ORDER[TRIAGE_LEVELS[triage]["rank"] + 1]
            triage = _more_urgent(triage, next_level)
            result.reasons.append(
                "Severity words were used, so the level was raised one step.")

    # 7d. RED FLAGS ALWAYS WIN. This safety override runs last so nothing
    # else can downgrade it.
    result.red_flags = check_red_flags(kept_text)
    if result.red_flags:
        triage = "Emergency"
        result.reasons.append("A red-flag warning sign was detected.")

    result.triage = triage
    return result


# Allow quick testing from the command line: python symptom_engine.py
if __name__ == "__main__":
    examples = [
        "I have a severe headache and a high fever.",
        "Runny nose, sneezing and a scratchy throat, no fever",
        "Crushing chest pain going down my left arm and I'm sweaty",
        "burns when I pee and I need to pee a lot",
    ]
    for example in examples:
        r = analyze_symptoms(example)
        print(f"\n> {example}\n  Triage: {r.triage}")
        for c in r.conditions:
            print(f"  - {c.name} ({c.confidence}, {c.score}) matched={c.matched_symptoms}")
        for flag in r.red_flags:
            print(f"  ! {flag}")
