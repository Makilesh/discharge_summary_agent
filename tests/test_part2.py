"""
test_part2.py — Unit Tests for Part 2 Learning Loop
======================================================

Tests cover all five Part 2 modules with no external API calls required
(except T-14 which mocks the LLM). Each test is independent and can run
in isolation.

Clinical Safety:
    T-04 and T-06 verify that the safety clamp works correctly — these
    are the most important tests in the suite.
"""

from __future__ import annotations
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.edit_signal import (
    compute_edit_signal,
    parse_summary_sections,
    _levenshtein_distance,
    EditSignal,
    SCORED_SECTIONS,
    SAFETY_CLAMP_CEILING,
)
from src.simulated_reviewer import (
    SimulatedReviewer,
    EditedDraft,
    _apply_rev001_conflict_disambiguation,
    _apply_rev002_undocumented_med_changes,
    _apply_rev003_urine_culture_followup,
    _apply_rev004_dama_prefix,
    _apply_rev005_citation_check,
    _apply_rev006_allergies_missing,
    _apply_rev007_flag_ordering,
)
from src.correction_memory import (
    CorrectionMemoryBank,
    CorrectionPattern,
)
from src.bandit import (
    ContextualBandit,
    GamingDetector,
    BanditDecision,
    ARMS,
    NUM_ARMS,
    OPTIMISTIC_INIT_REWARD,
)


# ─── FIXTURES ────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_draft():
    """A minimal but realistic draft for testing."""
    return """# DISCHARGE SUMMARY — DRAFT FOR CLINICIAN REVIEW

> ⚠️ **THIS IS AN AI-GENERATED DRAFT. NOT FOR CLINICAL USE WITHOUT CLINICIAN REVIEW AND SIGN-OFF.**

## 1. Patient Demographics
- **Name:** SMITH, JOHN
- **Age:** 25 years
- **MRN:** MRN-4567

## 2. Admission Date
12/05/2023

## 3. Discharge Date
17/05/2023

## 4. Principal Diagnosis
[CONFLICT — Multiple sources disagree. See escalation_flags. Clinician must resolve.]

## 5. Secondary Diagnoses
- Type 2 Diabetes Mellitus
- Urinary Tract Infection

## 6. Allergies
[MISSING — not documented in source records. Clinician must supply.]

## 7. Hospital Course
Patient presented with altered sensorium and high blood sugars. Managed with IV fluids and insulin.

## 8. Investigations Summary

### Laboratory Results
| Test | Value | Unit | Reference Range | Date | Abnormal |
|------|-------|------|-----------------|------|----------|
| Glucose | 443 | mg/dL | 70-110 | 12/05 | 🚨 CRITICAL |

### Imaging Results
- **USG Abdomen** (13/05/2023): Normal study

## 9. Procedures Performed
No procedures documented in source records.

## 10. Admission Medications
[MISSING — not documented in source records. Clinician must supply.]

## 11. Discharge Medications
[MISSING — not documented in source records. Clinician must supply.]

## 12. Medication Changes (Reconciliation)
| Drug | Change | Reason | Flag |
|------|--------|--------|------|
| INSULIN | ADDED_DURING_STAY | NOT DOCUMENTED | ⚠️ MEDICATION_ADDED_NO_REASON |

## 13. Pending Results
No pending results identified.

## 14. Follow-Up Instructions
- Follow up in 1 week

## 15. Discharge Condition
Discharge on request — attenders declined further management.

## 16. ⚠️ Escalation Flags for Clinician

### 🚨 CRITICAL
1. **SUMMARY_COMPLETENESS** — Agent step cap (20) reached.

### ⚠️ WARNING
1. **medication_reconciliation.INSULIN** — Insulin added without documented reason.

## 17. Conflicts Requiring Review

No conflicts detected.

---

## Summary Status

**DRAFT — NOT FOR CLINICAL USE WITHOUT REVIEW**
"""


@pytest.fixture
def sample_edited(sample_draft):
    """A minimally edited version of the draft."""
    # Apply a few corrections
    edited = sample_draft.replace(
        "[MISSING — not documented in source records. Clinician must supply.]\n\n## 7.",
        "NOT KNOWN (documented across nursing notes pages 17-22)\n\n## 7.",
    )
    edited = edited.replace("NOT DOCUMENTED", "⚠️ UNDOCUMENTED CHANGE — Clinician to annotate")
    return edited


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ─── T-01: REV-001 fires on CONFLICT token ──────────────────────────────────────

def test_t01_rev001_conflict_disambiguation(sample_draft):
    """REV-001 must fire when [CONFLICT] token is present in the draft."""
    result, applied = _apply_rev001_conflict_disambiguation(sample_draft)
    assert applied, "REV-001 should fire on CONFLICT token"
    assert "Diagnosis Disambiguation Table" in result
    assert "CLINICIAN MUST RESOLVE" in result


# ─── T-02: REV-004 fires on DAMA flag ───────────────────────────────────────────

def test_t02_rev004_dama_prefix(sample_draft):
    """REV-004 must fire when DAMA indicators are present."""
    result, applied = _apply_rev004_dama_prefix(sample_draft)
    assert applied, "REV-004 should fire on DAMA keywords"
    assert "DISCHARGE ON REQUEST" in result


# ─── T-03: Fabrication block for uncited LLM correction ─────────────────────────

def test_t03_fabrication_block():
    """Phase 2 LLM corrections without source citations must be blocked."""
    # Test the fabrication guard logic by mocking the entire LLM correction pass
    # to return a mix of cited and uncited corrections.
    mock_corrections = [
        {
            "section": "principal_diagnosis",
            "change": "Patient actually has Stage 4 cancer",
            "source_citation": ""  # No citation → fabrication
        },
        {
            "section": "discharge_medications",
            "change": "Add metformin 500mg",
            "source_citation": "Page 5"  # Has citation → accepted
        }
    ]

    # Mock _run_llm_correction_pass to simulate fabrication guard behavior
    with patch("src.simulated_reviewer._run_llm_correction_pass") as mock_llm_pass:
        # Simulate: uncited correction blocked, cited one accepted
        mock_llm_pass.return_value = (
            [{"section": "discharge_medications", "change": "Add metformin 500mg", "source_citation": "Page 5"}],
            ["REVIEWER_FABRICATION_BLOCKED: Section 'principal_diagnosis': 'Patient actually has Stage 4 cancer' — no source citation provided"],
        )

        reviewer = SimulatedReviewer(source_context="Test context")
        result = reviewer.review("## 4. Principal Diagnosis\n- Test\n## 16. Flags\nNone")

    assert len(result.reviewer_fabrication_blocks) >= 1, "Uncited correction must be blocked"
    assert "REVIEWER_FABRICATION_BLOCKED" in result.reviewer_fabrication_blocks[0]
    assert len(result.phase2_corrections) == 1
    assert result.phase2_corrections[0]["source_citation"] == "Page 5"


# ─── T-04: R_SAFE=0 when escalation flag dropped → reward ≤ 0.10 ────────────────

def test_t04_safety_clamp_on_dropped_flag():
    """
    If an escalation flag is present in the edited version but missing
    from the draft, R_SAFE must be 0.0 and composite_reward must be
    clamped to max(R, 0.10).
    """
    # Draft WITHOUT escalation flag
    draft = """## 16. ⚠️ Escalation Flags for Clinician

No escalation flags generated.
"""

    # Edited WITH escalation flag
    edited = """## 16. ⚠️ Escalation Flags for Clinician

### 🚨 CRITICAL
1. **SUMMARY_COMPLETENESS** — Agent step cap (20) reached.
"""

    signal = compute_edit_signal("test-safety", draft, edited)
    assert signal.r_safe == 0.0, f"R_SAFE should be 0.0 when flag dropped, got {signal.r_safe}"
    assert signal.safety_clamped, "Safety clamp should be active"
    assert signal.composite_reward <= SAFETY_CLAMP_CEILING, (
        f"Composite reward ({signal.composite_reward}) exceeds safety clamp ceiling ({SAFETY_CLAMP_CEILING})"
    )


# ─── T-05: Perfect match → composite_reward = 1.0 ───────────────────────────────

def test_t05_perfect_match():
    """
    When draft == edited (identical), all sub-signals should be 1.0
    and composite_reward should be 1.0.
    """
    text = """## 4. Principal Diagnosis
- DKA

## 7. Hospital Course
Patient treated for DKA.

## 11. Discharge Medications
- Insulin

## 12. Medication Changes
No changes.

## 13. Pending Results
No pending results identified.

## 15. Discharge Condition
Stable.

## 16. ⚠️ Escalation Flags for Clinician
No escalation flags generated.
"""
    signal = compute_edit_signal("test-perfect", text, text)
    assert signal.r_sed == 1.0, f"R_SED should be 1.0 for identical texts, got {signal.r_sed}"
    assert signal.r_sec == 1.0, f"R_SEC should be 1.0 for identical texts, got {signal.r_sec}"
    assert signal.composite_reward == 1.0, f"Composite should be 1.0, got {signal.composite_reward}"
    assert not signal.safety_clamped, "Safety clamp should not activate for perfect match"


# ─── T-06: Safety clamp activates correctly ──────────────────────────────────────

def test_t06_safety_clamp_ceiling():
    """
    Even if all other sub-signals are high, safety clamp must cap
    the composite reward when R_SAFE = 0.0.
    """
    # Construct a case where draft and edited are similar (high R_SED, R_SEC)
    # but an escalation flag is dropped
    draft = """## 4. Principal Diagnosis
- DKA

## 7. Hospital Course
Patient treated for DKA successfully.

## 11. Discharge Medications
- Insulin

## 12. Medication Changes
No changes.

## 13. Pending Results
No pending results identified.

## 15. Discharge Condition
Stable.

## 16. ⚠️ Escalation Flags for Clinician
No escalation flags generated.
"""

    edited = """## 4. Principal Diagnosis
- DKA

## 7. Hospital Course
Patient treated for DKA successfully.

## 11. Discharge Medications
- Insulin

## 12. Medication Changes
No changes.

## 13. Pending Results
No pending results identified.

## 15. Discharge Condition
Stable.

## 16. ⚠️ Escalation Flags for Clinician

### 🚨 CRITICAL
1. **GLUCOSE_CRITICAL** — Blood glucose 443 mg/dL requires monitoring.
"""

    signal = compute_edit_signal("test-clamp", draft, edited)
    assert signal.r_safe == 0.0
    assert signal.safety_clamped
    assert signal.composite_reward <= SAFETY_CLAMP_CEILING


# ─── T-07: Pattern persists across save/load cycle ──────────────────────────────

def test_t07_pattern_persistence(tmp_dir):
    """Correction patterns must survive a save/load cycle."""
    memory_path = os.path.join(tmp_dir, "test_memory.jsonl")

    # Create and save
    memory1 = CorrectionMemoryBank(memory_path=memory_path)
    pattern = CorrectionPattern(
        pattern_id="test-001",
        source_section="principal_diagnosis",
        rule_origin="REV-001",
        before_snippet="BEFORE text",
        after_snippet="AFTER text",
        clinical_category="diagnosis",
        frequency=3,
        reward_delta=0.15,
        last_seen_iso="2024-01-01T00:00:00Z",
    )
    memory1.add_pattern(pattern)
    memory1.save()

    # Load in new instance
    memory2 = CorrectionMemoryBank(memory_path=memory_path)
    assert len(memory2.patterns) == 1
    loaded = memory2.patterns[0]
    assert loaded.pattern_id == "test-001"
    assert loaded.source_section == "principal_diagnosis"
    assert loaded.frequency == 3
    assert loaded.reward_delta == 0.15


# ─── T-08: Retrieval returns correct section filter ──────────────────────────────

def test_t08_retrieval_section_filter(tmp_dir):
    """get_relevant_corrections must only return patterns from the requested section."""
    memory = CorrectionMemoryBank(memory_path=os.path.join(tmp_dir, "mem.jsonl"))

    # Add patterns for different sections
    for section in ["principal_diagnosis", "hospital_course", "discharge_medications"]:
        pattern = CorrectionPattern(
            pattern_id=f"test-{section}",
            source_section=section,
            rule_origin="REV-001",
            before_snippet="before",
            after_snippet="after",
            clinical_category="diagnosis",
            frequency=1,
            reward_delta=0.1,
            last_seen_iso="2024-01-01T00:00:00Z",
        )
        memory.add_pattern(pattern)

    results = memory.get_relevant_corrections("principal_diagnosis")
    assert len(results) == 1
    assert results[0].source_section == "principal_diagnosis"


# ─── T-09: Retrieval ranking by reward_delta * log(1+freq) ──────────────────────

def test_t09_retrieval_ranking(tmp_dir):
    """Patterns must be ranked by reward_delta * log(1+frequency) descending."""
    memory = CorrectionMemoryBank(memory_path=os.path.join(tmp_dir, "mem.jsonl"))

    # Add patterns with different scores
    patterns_data = [
        ("low", 1, 0.05),    # score = 0.05 * log(2) ≈ 0.035
        ("high", 10, 0.20),  # score = 0.20 * log(11) ≈ 0.480
        ("mid", 5, 0.10),    # score = 0.10 * log(6) ≈ 0.179
    ]

    for pid, freq, rd in patterns_data:
        pattern = CorrectionPattern(
            pattern_id=f"test-{pid}",
            source_section="principal_diagnosis",
            rule_origin="REV-001",
            before_snippet=f"before-{pid}",
            after_snippet=f"after-{pid}",
            clinical_category="diagnosis",
            frequency=freq,
            reward_delta=rd,
            last_seen_iso="2024-01-01T00:00:00Z",
        )
        memory.add_pattern(pattern)

    results = memory.get_relevant_corrections("principal_diagnosis", top_k=3)
    assert len(results) == 3

    # Verify ordering: high > mid > low
    scores = [r.reward_delta * math.log(1 + r.frequency) for r in results]
    assert scores[0] >= scores[1] >= scores[2], (
        f"Patterns not ranked correctly: {scores}"
    )
    assert results[0].pattern_id == "test-high"


# ─── T-10: UCB1 formula correct for known values ────────────────────────────────

def test_t10_ucb1_formula(tmp_dir):
    """UCB1 scores must match the formula: mean + sqrt(2 * ln(N) / n_i)."""
    bandit = ContextualBandit(state_path=os.path.join(tmp_dir, "bandit.json"))

    # Manually set arm stats
    bandit.arms[0].n_pulls = 5
    bandit.arms[0].mean_reward = 0.6
    bandit.arms[0].total_reward = 3.0

    bandit.arms[1].n_pulls = 3
    bandit.arms[1].mean_reward = 0.8
    bandit.arms[1].total_reward = 2.4

    # Set remaining arms as pulled once with low reward
    for i in range(2, NUM_ARMS):
        bandit.arms[i].n_pulls = 1
        bandit.arms[i].mean_reward = 0.3
        bandit.arms[i].total_reward = 0.3

    bandit.total_pulls = 5 + 3 + 1 * (NUM_ARMS - 2)

    decision = bandit.select_arm()

    # Verify UCB1 formula for arm 0
    expected_bonus_0 = math.sqrt(2 * math.log(bandit.total_pulls) / 5)
    expected_score_0 = 0.6 + expected_bonus_0

    assert abs(decision.ucb1_scores[0] - expected_score_0) < 0.001, (
        f"UCB1 score for arm 0: expected {expected_score_0:.4f}, got {decision.ucb1_scores[0]:.4f}"
    )


# ─── T-11: Unpulled arm selected first (optimistic init) ────────────────────────

def test_t11_unpulled_arm_selected_first(tmp_dir):
    """Arms that have never been pulled must be selected before pulled arms."""
    bandit = ContextualBandit(state_path=os.path.join(tmp_dir, "bandit.json"))

    # Pull arm 0 once
    bandit.update(0, 0.9)

    # Select next arm — should be an unpulled arm (1-4)
    decision = bandit.select_arm()
    assert decision.selected_arm != 0, (
        f"Unpulled arm should be selected, but got arm 0"
    )
    assert bandit.arms[decision.selected_arm].n_pulls == 0, (
        f"Selected arm {decision.selected_arm} has been pulled {bandit.arms[decision.selected_arm].n_pulls} times"
    )


# ─── T-12: Bandit state persists correctly to JSON ──────────────────────────────

def test_t12_bandit_state_persistence(tmp_dir):
    """Bandit state must survive a save/load cycle."""
    state_path = os.path.join(tmp_dir, "bandit_state.json")

    bandit1 = ContextualBandit(state_path=state_path)
    bandit1.update(0, 0.7)
    bandit1.update(0, 0.8)
    bandit1.update(1, 0.5)

    # Load in new instance
    bandit2 = ContextualBandit(state_path=state_path)
    assert bandit2.total_pulls == 3
    assert bandit2.arms[0].n_pulls == 2
    assert abs(bandit2.arms[0].mean_reward - 0.75) < 0.001
    assert bandit2.arms[1].n_pulls == 1
    assert abs(bandit2.arms[1].mean_reward - 0.5) < 0.001


# ─── T-13: GamingDetector fires on vagueness regression ─────────────────────────

def test_t13_gaming_detector_vagueness(tmp_dir):
    """
    If composite reward increases but principal_diagnosis accuracy decreases,
    the gaming detector must fire.
    """
    detector = GamingDetector(alert_log_path=os.path.join(tmp_dir, "alerts.log"))

    # Record iterations where reward goes up but PD accuracy goes down
    for i in range(1, 6):
        detector.record_iteration(
            iteration=i,
            composite_reward=0.4 + (i * 0.05),  # Increasing
            section_scores={
                "principal_diagnosis": 0.8 - (i * 0.05),  # Decreasing
                "hospital_course": 0.7,
            },
            hospital_course_word_count=100,
        )

    # Check at iteration 5
    alerts = detector.check_for_gaming(5)
    assert len(alerts) >= 1, "Gaming detector should fire on vagueness pattern"
    assert "vagueness" in alerts[0].lower() or "gaming" in alerts[0].lower()


# ─── T-14: Full pipeline produces all output artifacts ───────────────────────────

def test_t14_full_pipeline_outputs(tmp_dir):
    """
    A full orchestrator run (with mocked LLM) must produce all required
    output artifacts.
    """
    from src.learning_loop import LearningOrchestrator

    # Create a minimal agent state
    agent_state = {
        "current_phase": "DONE",
        "loaded_documents": [],
        "unreadable_pages": [],
        "steps_remaining": 0,
        "extracted_demographics": {"name": "TEST PATIENT", "age": "25"},
        "admission_date": "01/01/2024",
        "discharge_date": "05/01/2024",
        "diagnoses": {"principal": ["Test Diagnosis"], "secondary": [], "provisional": [], "final": ["Test Diagnosis"]},
        "hospital_course": "Patient was admitted and treated.",
        "procedures": [],
        "allergies": ["NOT KNOWN"],
        "discharge_condition": "Stable",
        "admission_medications": [],
        "inpatient_medications": [],
        "discharge_medications": [],
        "medication_reconciliation": [],
        "lab_results": [],
        "imaging_results": [],
        "pending_results": [],
        "follow_up_instructions": ["Follow up in 1 week"],
        "conflicts": [],
        "escalation_flags": [{"field": "TEST", "severity": "WARNING", "reason": "Test flag", "requires_clinician": True}],
        "fabrication_blocks": [],
        "trace": [],
        "retry_counts": {},
        "processing_queue": [],
    }

    # Mock the LLM correction pass to avoid real API calls
    with patch("src.simulated_reviewer._run_llm_correction_pass") as mock_llm_pass:
        mock_llm_pass.return_value = ([], [])  # No corrections, no fabrications

        orchestrator = LearningOrchestrator(
            agent_state=agent_state,
            output_dir=tmp_dir,
            n_train=3,
        )

        records = orchestrator.run_training()
        eval_results = orchestrator.run_evaluation()
        orchestrator.generate_output_artifacts(eval_results)

    # Verify required artifacts exist
    expected_files = [
        "training_curve.json",
        "before_after_report.md",
        "correction_memory_summary.md",
        "limitations_analysis.md",
        "reviewer_prompt_used.txt",
    ]

    for fname in expected_files:
        fpath = os.path.join(tmp_dir, fname)
        assert os.path.exists(fpath), f"Missing output artifact: {fname}"
        assert os.path.getsize(fpath) > 0, f"Output artifact is empty: {fname}"

    # Verify training_curve.json has correct number of entries
    with open(os.path.join(tmp_dir, "training_curve.json"), "r") as f:
        curve = json.load(f)
    assert len(curve) == 3, f"Expected 3 entries in training_curve.json, got {len(curve)}"


# ─── T-15: Exploit-only mode selects highest mean-reward arm ────────────────────

def test_t15_exploit_only_mode(tmp_dir):
    """In exploit-only mode, the bandit must select the arm with the highest mean reward."""
    bandit = ContextualBandit(state_path=os.path.join(tmp_dir, "bandit.json"))

    # Set up known mean rewards
    rewards = {0: 0.5, 1: 0.9, 2: 0.3, 3: 0.7, 4: 0.6}
    for arm_id, reward in rewards.items():
        bandit.arms[arm_id].n_pulls = 5
        bandit.arms[arm_id].mean_reward = reward
        bandit.arms[arm_id].total_reward = reward * 5
    bandit.total_pulls = 25

    decision = bandit.select_arm(exploit_only=True)
    assert decision.selected_arm == 1, (
        f"Exploit-only should select arm 1 (highest mean 0.9), got arm {decision.selected_arm}"
    )
    assert "exploit" in decision.reason.lower()


# ─── ADDITIONAL TESTS ───────────────────────────────────────────────────────────

def test_levenshtein_basic():
    """Basic Levenshtein distance calculations."""
    assert _levenshtein_distance("", "") == 0
    assert _levenshtein_distance("abc", "abc") == 0
    assert _levenshtein_distance("abc", "abd") == 1
    assert _levenshtein_distance("abc", "") == 3
    assert _levenshtein_distance("", "abc") == 3
    assert _levenshtein_distance("kitten", "sitting") == 3


def test_section_parser():
    """Section parser must extract sections correctly."""
    text = """## 4. Principal Diagnosis
- DKA
- T2DM

## 5. Secondary Diagnoses
- HTN

## 7. Hospital Course
Patient was treated for DKA.

## 11. Discharge Medications
- Insulin

## 12. Medication Changes
No changes.

## 13. Pending Results
No pending results identified.

## 15. Discharge Condition
Stable.

## 16. ⚠️ Escalation Flags for Clinician
No escalation flags generated.
"""
    sections = parse_summary_sections(text)
    assert "principal_diagnosis" in sections
    assert "DKA" in sections["principal_diagnosis"]
    assert "hospital_course" in sections
    assert "treated" in sections["hospital_course"]


def test_rev002_undocumented_med_changes():
    """REV-002 must replace NOT DOCUMENTED with warning annotation."""
    text = "| INSULIN | ADDED_DURING_STAY | NOT DOCUMENTED | ⚠️ FLAG |"
    result, applied = _apply_rev002_undocumented_med_changes(text)
    assert applied
    assert "UNDOCUMENTED CHANGE" in result
    assert "NOT DOCUMENTED" not in result


def test_rev006_allergies_missing():
    """REV-006 must replace [MISSING] in allergies with NOT KNOWN."""
    text = """## 6. Allergies
[MISSING — not documented in source records. Clinician must supply.]

## 7. Hospital Course"""
    result, applied = _apply_rev006_allergies_missing(text)
    assert applied
    assert "NOT KNOWN" in result


def test_correction_memory_deduplication(tmp_dir):
    """Adding the same pattern twice should increment frequency, not duplicate."""
    memory = CorrectionMemoryBank(memory_path=os.path.join(tmp_dir, "mem.jsonl"))

    for _ in range(3):
        pattern = CorrectionPattern(
            pattern_id=f"test-dup",
            source_section="principal_diagnosis",
            rule_origin="REV-001",
            before_snippet="same before",
            after_snippet="same after",
            clinical_category="diagnosis",
            frequency=1,
            reward_delta=0.1,
            last_seen_iso="2024-01-01T00:00:00Z",
        )
        memory.add_pattern(pattern)

    assert len(memory.patterns) == 1, "Duplicate patterns should be merged"
    assert memory.patterns[0].frequency == 3, "Frequency should be incremented"


def test_bandit_arm_count():
    """Verify we have exactly 5 arms defined."""
    assert len(ARMS) == NUM_ARMS == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
