"""
tests/test_engine.py
====================
Automated tests for the symptom engine. Run with:  pytest -q

The cases below act as a tiny "validation set": each one is a realistic way
a person might describe their symptoms, paired with the condition and triage
level we expect. They cover:
  * everyday language / synonyms
  * negation ("no fever")
  * red-flag safety overrides
  * the rule that keeps thin evidence from producing "Emergency"
"""

import os
import sys

import pytest

# Make the project root importable when running "pytest" from any folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symptom_engine import analyze_symptoms, remove_negations, normalize_text  # noqa: E402


# (symptom text, expected top condition, expected triage level)
CASES = [
    ("Runny nose, sneezing and a scratchy throat, no fever",
     "Common Cold", "Self-Care"),
    ("Itchy watery eyes and sneezing every spring", "Seasonal Allergies", "Self-Care"),
    ("Fever, chills, body aches and I'm exhausted", "Influenza (Flu)", "Primary Care"),
    ("I lost my sense of smell, can't taste anything, and have a cough",
     "COVID-19", "Primary Care"),
    ("It burns when I pee and I'm peeing a lot", "Urinary Tract Infection (UTI)",
     "Primary Care"),
    ("Throwing up and diarrhea since last night with stomach cramps",
     "Gastroenteritis (Stomach Bug)", "Self-Care"),
    ("Pounding headache, sensitive to light and nauseous", "Migraine", "Primary Care"),
    ("Wheezing and tight chest, short of breath", "Asthma Flare-up", "Urgent Care"),
    ("Twisted my ankle, it's swollen and bruised", "Ankle or Joint Sprain", "Self-Care"),
    ("Crushing chest pain going down my left arm and I'm sweaty",
     "Heart Attack (Possible)", "Emergency"),
    ("My face is drooping and I have slurred speech", "Stroke (Possible)", "Emergency"),
    ("High fever, stiff neck and I'm confused", "Meningitis (Possible)", "Emergency"),
    ("Sharp lower right abdominal pain, nausea and vomiting",
     "Appendicitis", "Emergency"),
]


@pytest.mark.parametrize("text,expected_condition,expected_triage", CASES)
def test_example_cases(text, expected_condition, expected_triage):
    result = analyze_symptoms(text)
    assert result.conditions, f"No condition matched for: {text}"
    assert result.conditions[0].name == expected_condition
    assert result.triage == expected_triage


def test_negation_removes_symptom():
    kept, negated = remove_negations("headache and nausea, no fever but some chills")
    assert "fever" in " ".join(negated)
    assert "fever" not in " ".join(kept)
    assert "chills" in " ".join(kept)


def test_synonyms_are_normalized():
    assert normalize_text("i keep throwing up") == "i keep vomiting"
    # identity entry prevents "nasal nasal congestion"
    assert normalize_text("nasal congestion") == "nasal congestion"


def test_headache_and_fever_is_not_an_emergency():
    """Thin evidence for a serious condition must not trigger 'Emergency'."""
    result = analyze_symptoms("I have a severe headache and a high fever.")
    assert result.triage != "Emergency"


def test_red_flag_overrides_everything():
    result = analyze_symptoms("I have a mild cold but now I can't breathe")
    assert result.triage == "Emergency"
    assert result.red_flags


def test_negated_red_flag_does_not_fire():
    result = analyze_symptoms("Cough and runny nose, no chest pain")
    assert result.triage != "Emergency"
    assert not result.red_flags


def test_crisis_language_gets_emergency_and_helpline():
    result = analyze_symptoms("I feel hopeless and want to end my life")
    assert result.triage == "Emergency"
    assert any("988" in msg for msg in result.red_flags)


def test_severity_word_escalates_one_step():
    mild = analyze_symptoms("runny nose and sneezing")
    severe = analyze_symptoms("severe runny nose and sneezing")
    assert mild.triage == "Self-Care"
    assert severe.triage == "Primary Care"


def test_unknown_input_falls_back_to_primary_care():
    result = analyze_symptoms("my toe feels purple and weird")
    assert result.triage == "Primary Care"


def test_empty_input_raises():
    with pytest.raises(ValueError):
        analyze_symptoms("   ")
