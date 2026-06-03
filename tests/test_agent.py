"""
test_agent.py — Unit Tests for the Discharge Summary Agent
============================================================

Tests cover:
    - CR-1: Treatment without diagnosis detection
    - CR-2: Diagnosis mismatch detection
    - CR-3: Lab evidence vs clinical claim
    - CR-4: Culture-treatment mismatch
    - CR-5: DAMA detection
    - Medication reconciliation flags
    - Hard cap behavior
    - Empty page handling
    - State validation

Clinical Safety:
    These tests validate that the agent NEVER silently resolves conflicts,
    NEVER fabricates data, and ALWAYS flags safety concerns.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.state import create_initial_state
from src.cross_reference import (
    check_cr1_treatment_diagnosis_alignment,
    check_cr2_diagnosis_conflicts,
    check_cr3_lab_evidence,
    check_cr4_culture_treatment,
    check_cr5_discharge_condition,
    cross_reference_audit,
    escalate_to_clinician,
)
from src.tools import reconcile_medications
from src.trace import validate_state_completeness, generate_trace_summary, emit_trace
from src.config import MAX_ITERATIONS


# ─── FIXTURES ────────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_state() -> dict:
    """Create a clean initial state for testing."""
    return create_initial_state()


@pytest.fixture
def state_with_insulin_no_dm() -> dict:
    """State where insulin is given but no DM/DKA diagnosis exists."""
    state = create_initial_state()
    state["inpatient_medications"] = [
        {
            "name": "INJ LANTUS",
            "dose": "10U",
            "route": "SC",
            "frequency": "OD",
            "start_date": "28/2/26",
            "stop_date": None,
            "status": "INPATIENT_ONLY",
            "change_reason": None,
            "change_reason_documented": False,
            "source_pages": [43, 44],
        },
        {
            "name": "INJ H.ACTRAPID",
            "dose": "sliding scale",
            "route": "SC",
            "frequency": "TID",
            "start_date": "28/2/26",
            "stop_date": None,
            "status": "INPATIENT_ONLY",
            "change_reason": None,
            "change_reason_documented": False,
            "source_pages": [43, 44],
        },
    ]
    state["diagnoses"] = {
        "principal": [],
        "secondary": [],
        "provisional": ["TAFE"],
        "final": ["Synovitis", "Cholelithiasis without cholecystitis"],
    }
    state["discharge_medications"] = []
    return state


@pytest.fixture
def state_with_diagnosis_conflicts() -> dict:
    """State with multiple conflicting diagnoses across document types."""
    state = create_initial_state()
    state["diagnoses"] = {
        "principal": [],
        "secondary": [],
        "provisional": ["DKA", "TAFE + Uncontrolled T2DM"],
        "final": ["Synovitis", "Cholelithiasis without cholecystitis"],
    }
    state["loaded_documents"] = [
        {
            "page_num": 1,
            "source_type": "ER_OBSERVATION_CHART",
            "raw_text": "Provisional Dx: DKA",
            "confidence": 0.8,
            "extracted_data": {"er_diagnosis": ["DKA"]},
        },
        {
            "page_num": 46,
            "source_type": "ADMISSION_RECORD",
            "raw_text": "Provisional: TAFE + Uncontrolled T2DM. Final: Synovitis",
            "confidence": 0.9,
            "extracted_data": {
                "provisional_diagnosis": ["TAFE", "Uncontrolled T2DM"],
                "final_diagnosis": ["Synovitis", "Cholelithiasis without cholecystitis"],
            },
        },
        {
            "page_num": 10,
            "source_type": "ICU_CHART",
            "raw_text": "DKA + Uncontrolled T2DM",
            "confidence": 0.7,
            "extracted_data": {"icu_diagnoses": ["DKA", "Uncontrolled T2DM"]},
        },
        {
            "page_num": 30,
            "source_type": "CONSULTATION_SHEET",
            "raw_text": "AFI, DKA, Uncontrolled T2DM, B/L Pyelonephritis",
            "confidence": 0.8,
            "extracted_data": {
                "consultation_diagnosis": [
                    "AFI", "DKA", "Uncontrolled T2DM", "B/L Pyelonephritis"
                ],
            },
        },
    ]
    return state


@pytest.fixture
def state_with_culture_mismatch() -> dict:
    """State where culture is negative but IV antibiotics were given."""
    state = create_initial_state()
    state["lab_results"] = [
        {
            "test_name": "Urine Culture",
            "result_value": "NO SIGNIFICANT BACTERIURIA, Colony count < 10,000 CFU/ML",
            "unit": "",
            "reference_range": "",
            "date": "27/2/26",
            "abnormal_flag": False,
            "source_page": 41,
        },
    ]
    state["inpatient_medications"] = [
        {
            "name": "INJ MEROPENEM",
            "dose": "1g",
            "route": "IV",
            "frequency": "TID",
            "start_date": "28/2/26",
            "stop_date": None,
            "status": "INPATIENT_ONLY",
            "change_reason": None,
            "change_reason_documented": False,
        },
    ]
    return state


@pytest.fixture
def state_with_critical_labs() -> dict:
    """State with critically abnormal lab values."""
    state = create_initial_state()
    state["lab_results"] = [
        {
            "test_name": "Sodium",
            "result_value": "114",
            "unit": "mmol/L",
            "reference_range": "136-146",
            "date": "27/2/26",
            "abnormal_flag": True,
            "critically_abnormal": True,
            "source_page": 38,
        },
        {
            "test_name": "Blood Glucose (GRBS)",
            "result_value": "443",
            "unit": "mg/dL",
            "reference_range": "70-140",
            "date": "28/2/26",
            "abnormal_flag": True,
            "critically_abnormal": True,
            "source_page": 50,
        },
    ]
    state["discharge_condition"] = "Hemodynamically stable, improved"
    return state


@pytest.fixture
def state_with_dama() -> dict:
    """State with Discharge Against Medical Advice indicators."""
    state = create_initial_state()
    state["discharge_condition"] = "Stable"
    state["loaded_documents"] = [
        {
            "page_num": 30,
            "source_type": "CONSULTATION_SHEET",
            "raw_text": "Discharge on Request (Evening). Attenders not willing to continue.",
            "confidence": 0.8,
            "extracted_data": {
                "discharge_plan": "Discharge on Request",
                "recommendations": ["Follow up in 1 week"],
            },
        },
        {
            "page_num": 8,
            "source_type": "NURSING_NOTES",
            "raw_text": "Patient seen by Dr, advised discharge on request of attenders.",
            "confidence": 0.7,
            "extracted_data": {},
        },
    ]
    return state


# ─── TEST: CR-1 — Treatment Without Diagnosis ───────────────────────────────────

class TestCR1:
    """Tests for CR-1: Treatment-Diagnosis Alignment."""

    def test_insulin_without_dm_diagnosis(self, state_with_insulin_no_dm: dict):
        """
        CONFLICT-002: Lantus + Actrapid given but final diagnosis omits DM/DKA.
        Must detect and escalate.
        """
        conflicts = check_cr1_treatment_diagnosis_alignment(state_with_insulin_no_dm)

        assert len(conflicts) > 0, "CR-1 should detect insulin without DM diagnosis"

        treatment_conflicts = [
            c for c in conflicts
            if c["type"] == "TREATMENT_WITHOUT_DIAGNOSIS"
        ]
        assert len(treatment_conflicts) > 0, "Should flag TREATMENT_WITHOUT_DIAGNOSIS"

        # Verify escalation was triggered
        flags = state_with_insulin_no_dm.get("escalation_flags", [])
        assert len(flags) > 0, "Should have escalation flags"
        assert any(f["severity"] == "CRITICAL" for f in flags), "Should be CRITICAL severity"

    def test_insulin_with_dm_diagnosis_no_conflict(self, empty_state: dict):
        """If DM is in the diagnosis, insulin should NOT trigger CR-1."""
        state = empty_state
        state["inpatient_medications"] = [
            {"name": "INJ LANTUS", "dose": "10U", "route": "SC",
             "frequency": "OD", "status": "INPATIENT_ONLY",
             "change_reason": None, "change_reason_documented": False},
        ]
        state["diagnoses"] = {
            "principal": ["DKA"],
            "secondary": ["Uncontrolled T2DM"],
            "provisional": [],
            "final": ["DKA", "Uncontrolled T2DM"],
        }
        state["discharge_medications"] = []

        conflicts = check_cr1_treatment_diagnosis_alignment(state)

        treatment_conflicts = [
            c for c in conflicts
            if c["type"] == "TREATMENT_WITHOUT_DIAGNOSIS"
        ]
        assert len(treatment_conflicts) == 0, "Should NOT flag when DM is diagnosed"


# ─── TEST: CR-2 — Diagnosis Mismatch ────────────────────────────────────────────

class TestCR2:
    """Tests for CR-2: Inter-Document Diagnosis Conflict."""

    def test_multi_source_diagnosis_conflict(self, state_with_diagnosis_conflicts: dict):
        """
        CONFLICT-001: ER says DKA, admission says TAFE, ICU says DKA+T2DM,
        consultation says AFI+DKA+T2DM+Pyelonephritis, final says Synovitis.
        Must detect ALL conflicts.
        """
        conflicts = check_cr2_diagnosis_conflicts(state_with_diagnosis_conflicts)

        assert len(conflicts) > 0, "CR-2 should detect diagnosis conflicts"

        diag_conflicts = [c for c in conflicts if c["type"] == "DIAGNOSIS_MISMATCH"]
        assert len(diag_conflicts) > 0, "Should flag DIAGNOSIS_MISMATCH"

        # Verify escalation was triggered
        flags = state_with_diagnosis_conflicts.get("escalation_flags", [])
        assert len(flags) > 0, "Should have escalation flags"

    def test_consistent_diagnoses_no_conflict(self, empty_state: dict):
        """If all sources agree, no CR-2 conflict should fire."""
        state = empty_state
        state["diagnoses"] = {
            "principal": ["DKA"],
            "secondary": [],
            "provisional": ["DKA"],
            "final": ["DKA"],
        }
        state["loaded_documents"] = [
            {
                "page_num": 1, "source_type": "ER_OBSERVATION_CHART",
                "raw_text": "DKA", "confidence": 0.9,
                "extracted_data": {"er_diagnosis": ["DKA"]},
            },
        ]

        conflicts = check_cr2_diagnosis_conflicts(state)
        diag_conflicts = [c for c in conflicts if c["type"] == "DIAGNOSIS_MISMATCH"]
        # Consistent diagnoses should have no or minimal conflicts
        assert len(diag_conflicts) == 0, "Should NOT flag when diagnoses are consistent"

    def test_chief_complaint_none_does_not_crash(self, empty_state: dict):
        """Malformed extracted chief complaint entries must not abort CR-2."""
        state = empty_state
        state["loaded_documents"] = [
            {
                "page_num": 1,
                "source_type": "ER_OBSERVATION_CHART",
                "raw_text": "",
                "confidence": 0.8,
                "extracted_data": {"chief_complaints": [None]},
            },
            {
                "page_num": 2,
                "source_type": "ADMISSION_RECORD",
                "raw_text": "",
                "confidence": 0.8,
                "extracted_data": {"chief_complaints": ["Fever"]},
            },
        ]

        conflicts = check_cr2_diagnosis_conflicts(state)
        assert isinstance(conflicts, list)


# ─── TEST: CR-3 — Lab Evidence vs Clinical Claim ────────────────────────────────

class TestCR3:
    """Tests for CR-3: Lab Evidence vs Clinical Claim."""

    def test_critical_sodium_detected(self, state_with_critical_labs: dict):
        """
        CONFLICT-004: Sodium 114 mmol/L is critically low (ref 136-146).
        Must be detected and escalated.
        """
        conflicts = check_cr3_lab_evidence(state_with_critical_labs)

        critical_labs = [c for c in conflicts if c["type"] == "CRITICAL_LAB_VALUE"]
        assert len(critical_labs) > 0, "Should detect critically abnormal sodium"

        # Check that the specific sodium finding is present
        sodium_conflict = [
            c for c in critical_labs
            if "sodium" in c.get("description", "").lower()
        ]
        assert len(sodium_conflict) > 0, "Should specifically flag sodium 114"

    def test_discharge_improved_vs_abnormal_labs(self, state_with_critical_labs: dict):
        """
        If discharge says "improved" but labs are still critically abnormal,
        must flag the discrepancy.
        """
        conflicts = check_cr3_lab_evidence(state_with_critical_labs)

        dc_conflicts = [c for c in conflicts if c["type"] == "DISCHARGE_CLAIM_VS_LABS"]
        assert len(dc_conflicts) > 0, "Should flag discharge claim vs abnormal labs"


# ─── TEST: CR-4 — Culture-Treatment Mismatch ────────────────────────────────────

class TestCR4:
    """Tests for CR-4: Culture Result vs Treatment."""

    def test_negative_culture_with_antibiotics(self, state_with_culture_mismatch: dict):
        """
        CONFLICT-003: Urine culture negative but IV Meropenem given.
        Must detect and flag for clinician review.
        """
        conflicts = check_cr4_culture_treatment(state_with_culture_mismatch)

        assert len(conflicts) > 0, "CR-4 should detect culture-treatment mismatch"
        assert conflicts[0]["type"] == "CULTURE_TREATMENT_MISMATCH"
        assert "meropenem" in conflicts[0]["description"].lower()


# ─── TEST: CR-5 — DAMA Detection ────────────────────────────────────────────────

class TestCR5:
    """Tests for CR-5: Discharge Condition Claim."""

    def test_dama_detection(self, state_with_dama: dict):
        """
        CONFLICT-005: Multiple documents indicate discharge on request.
        Must detect and flag DAMA status.
        """
        conflicts = check_cr5_discharge_condition(state_with_dama)

        dama_conflicts = [
            c for c in conflicts
            if c["type"] == "DISCHARGE_AGAINST_ADVICE_IMPLICATIONS"
        ]
        assert len(dama_conflicts) > 0, "Should detect DAMA indicators"
        assert "on request" in dama_conflicts[0]["description"].lower()


# ─── TEST: Medication Reconciliation ─────────────────────────────────────────────

class TestMedicationReconciliation:
    """Tests for medication reconciliation logic."""

    def test_stopped_medication_flagged(self):
        """A medication active during admission but missing from discharge must be flagged."""
        admission = []
        inpatient = [
            {"name": "INJ MEROPENEM", "dose": "1g", "route": "IV",
             "frequency": "TID", "status": "INPATIENT_ONLY",
             "change_reason": None, "change_reason_documented": False},
        ]
        discharge = []

        result = reconcile_medications(admission, inpatient, discharge)

        recon = result["reconciliation"]
        flags = result["flags"]

        stopped = [r for r in recon if r.get("change_type") == "STOPPED_AT_DISCHARGE"]
        assert len(stopped) > 0, "Should detect stopped medication"

        assert len(flags) > 0, "Should flag stopped medication without reason"
        assert flags[0]["severity"] == "WARNING"

    def test_new_at_discharge_flagged(self):
        """A medication appearing only in discharge list must be flagged."""
        admission = []
        inpatient = []
        discharge = [
            {"name": "TAB METFORMIN", "dose": "500mg", "route": "PO",
             "frequency": "BD", "status": "DISCHARGE",
             "change_reason": None, "change_reason_documented": False},
        ]

        result = reconcile_medications(admission, inpatient, discharge)
        flags = result["flags"]

        new_flags = [f for f in flags if "NEW_MEDICATION" in f.get("reason", "")]
        assert len(new_flags) > 0 or any(
            r.get("change_type") == "NEW_AT_DISCHARGE"
            for r in result["reconciliation"]
        ), "Should detect new medication at discharge"

    def test_continued_medication_no_flag(self):
        """A medication continued from admission to discharge should not be flagged."""
        admission = [
            {"name": "TAB ASPIRIN", "dose": "75mg", "route": "PO",
             "frequency": "OD", "status": "ADMISSION",
             "change_reason": None, "change_reason_documented": True},
        ]
        inpatient = []
        discharge = [
            {"name": "TAB ASPIRIN", "dose": "75mg", "route": "PO",
             "frequency": "OD", "status": "DISCHARGE",
             "change_reason": None, "change_reason_documented": True},
        ]

        result = reconcile_medications(admission, inpatient, discharge)
        flags = result["flags"]

        aspirin_flags = [f for f in flags if "ASPIRIN" in f.get("field", "").upper()]
        assert len(aspirin_flags) == 0, "Continued medication should not be flagged"


# ─── TEST: Hard Cap Behavior ────────────────────────────────────────────────────

class TestHardCap:
    """Tests for the 20-step hard cap enforcement."""

    def test_steps_remaining_zero_triggers_escalation(self, empty_state: dict):
        """When steps_remaining hits 0, the agent must trigger HARD_CAP_HIT."""
        state = empty_state
        state["steps_remaining"] = 0

        # Import the routing function
        from src.graph import route_after_verify

        result = route_after_verify(state)
        assert result == "hard_cap_escalate", "Should route to hard_cap_escalate when steps=0"

    def test_steps_remaining_negative_triggers_escalation(self, empty_state: dict):
        """Negative steps_remaining should also trigger escalation."""
        state = empty_state
        state["steps_remaining"] = -1

        from src.graph import route_after_verify

        result = route_after_verify(state)
        assert result == "hard_cap_escalate", "Should route to hard_cap_escalate when steps<0"


# ─── TEST: State Validation ─────────────────────────────────────────────────────

class TestStateValidation:
    """Tests for validate_state_completeness."""

    def test_empty_state_reports_all_missing(self, empty_state: dict):
        """A fresh state should have most fields missing."""
        completed, missing, flagged = validate_state_completeness(empty_state)
        assert len(missing) > 0, "Empty state should report missing fields"
        assert "demographics" in missing
        assert "admission_date" in missing

    def test_populated_state_reports_complete(self, empty_state: dict):
        """A fully populated state should report all fields complete."""
        state = empty_state
        state["extracted_demographics"] = {"name": "Test", "age": "50", "gender": "M"}
        state["admission_date"] = "26/02/2026"
        state["discharge_date"] = "02/03/2026"
        state["diagnoses"] = {"principal": ["DKA"], "secondary": [], "provisional": [], "final": ["DKA"]}
        state["hospital_course"] = "Patient was admitted..."
        state["admission_medications"] = [{"name": "Test", "dose": "1mg"}]
        state["inpatient_medications"] = [{"name": "Test", "dose": "1mg"}]
        state["discharge_medications"] = [{"name": "Test", "dose": "1mg"}]
        state["lab_results"] = [{"test_name": "CBC", "result_value": "Normal"}]
        state["imaging_results"] = [{"modality": "USG", "impression": "Normal"}]
        state["allergies"] = ["NOT KNOWN"]
        state["follow_up_instructions"] = ["Review in 1 week"]
        state["discharge_condition"] = "Stable"

        completed, missing, flagged = validate_state_completeness(state)
        assert len(completed) > 5, "Populated state should report many complete fields"


# ─── TEST: Trace Emission ───────────────────────────────────────────────────────

class TestTrace:
    """Tests for trace emission and summary generation."""

    def test_emit_trace_appends_entry(self, empty_state: dict):
        """emit_trace should append an entry to state['trace']."""
        emit_trace(
            state=empty_state,
            step_number=1,
            phase="TEST",
            reasoning="Test reasoning",
            action="TEST_ACTION",
        )
        assert len(empty_state["trace"]) == 1
        assert empty_state["trace"][0]["phase"] == "TEST"

    def test_trace_summary_has_required_fields(self, empty_state: dict):
        """generate_trace_summary should produce a valid summary."""
        summary = generate_trace_summary(empty_state)

        assert summary["step"] == "FINAL"
        assert "total_steps_used" in summary
        assert "safety_verdict" in summary
        assert summary["safety_verdict"] == "DO_NOT_FINALIZE_WITHOUT_CLINICIAN_SIGN_OFF"


# ─── TEST: Escalate To Clinician ─────────────────────────────────────────────────

class TestEscalation:
    """Tests for the escalation mechanism."""

    def test_escalation_appends_flag(self, empty_state: dict):
        """escalate_to_clinician should append a flag to the state."""
        escalate_to_clinician(
            state=empty_state,
            field="diagnoses",
            severity="CRITICAL",
            reason="Test escalation",
            source_evidence=["Evidence A"],
            source_pages=[1],
        )

        flags = empty_state.get("escalation_flags", [])
        assert len(flags) == 1
        assert flags[0]["severity"] == "CRITICAL"
        assert flags[0]["field"] == "diagnoses"
        assert flags[0]["requires_clinician"] is True

    def test_escalation_never_clears_existing(self, empty_state: dict):
        """Multiple escalations should accumulate, never overwrite."""
        for i in range(3):
            escalate_to_clinician(
                state=empty_state,
                field=f"field_{i}",
                severity="WARNING",
                reason=f"Reason {i}",
                source_evidence=[],
                source_pages=[],
            )

        flags = empty_state.get("escalation_flags", [])
        assert len(flags) == 3, "All escalations should accumulate"


# ─── TEST: Empty/Unreadable Page Handling ────────────────────────────────────────

class TestPageHandling:
    """Tests for handling blank or unreadable pages."""

    def test_empty_page_text_marked_unreadable(self, empty_state: dict):
        """Pages with < 30 chars of text should be considered unreadable."""
        from src.config import MIN_TEXT_LENGTH

        short_text = "abc"  # < MIN_TEXT_LENGTH
        assert len(short_text) < MIN_TEXT_LENGTH, "Test text should be shorter than threshold"

        # The graph logic checks this during extraction
        # Just verify the threshold constant is reasonable
        assert MIN_TEXT_LENGTH == 30


# ─── TEST: Graph Planning Handoff & Batch Extraction ────────────────────────────

class TestGraphExtraction:
    """Regression tests for REASON → CALL_TOOL planning state."""

    def test_reason_targets_survive_into_call_tool(self, monkeypatch):
        """
        LangGraph must preserve the planned document type and pages; otherwise
        CALL_TOOL silently extracts zero pages.
        """
        import os
        from src.graph import build_agent_graph

        monkeypatch.setattr(
            "src.graph.extract_all_page_images",
            lambda pdf_path: {1: b"fake-png"},
        )
        monkeypatch.setattr(
            "src.graph.classify_document_pages",
            lambda page_images: [
                {"page_num": 1, "doc_type": "TYPED_DISCHARGE_SUMMARY", "confidence": 0.9}
            ],
        )
        monkeypatch.setattr(
            "src.graph.extract_page_text",
            lambda img_b64, page_num: {
                "text": "Typed discharge summary text longer than thirty characters.",
                "confidence": 1.0,
                "page_num": page_num,
            },
        )
        monkeypatch.setattr(
            "src.graph.extract_typed_summary",
            lambda text, page_num: {
                "demographics": {"name": "TEST PATIENT"},
                "admission_date": "01/01/2024",
                "discharge_date": "02/01/2024",
                "diagnoses": {"principal": ["Test"], "secondary": [], "provisional": [], "final": ["Test"]},
                "follow_up": ["Review"],
                "condition_at_discharge": "Stable",
                "allergies": ["NOT KNOWN"],
            },
        )

        state = create_initial_state()
        state["_pdf_path"] = "dummy.pdf"
        agent = build_agent_graph()

        try:
            call_tool_output = None
            for event in agent.stream(state, {"recursion_limit": 10}):
                if "call_tool" in event:
                    call_tool_output = event["call_tool"]
                    break
        finally:
            os.environ.pop("OCR_CACHE_NAMESPACE", None)

        assert call_tool_output is not None
        assert call_tool_output["loaded_documents"][0]["page_num"] == 1
        assert call_tool_output["loaded_documents"][0]["source_type"] == "TYPED_DISCHARGE_SUMMARY"
        assert call_tool_output["extracted_demographics"]["name"] == "TEST PATIENT"

    def test_unknown_pages_are_queued_for_late_extraction(self):
        """UNKNOWN pages should not be silently dropped from the processing queue."""
        from src.graph import _build_processing_queue

        queue = _build_processing_queue([
            {"page_num": 1, "source_type": "UNKNOWN"},
            {"page_num": 2, "source_type": "DRUG_CHART"},
        ])

        assert queue[0] == {"doc_type": "DRUG_CHART", "pages": [2]}
        assert queue[-1] == {"doc_type": "UNKNOWN", "pages": [1]}

    def test_classifications_are_deduped_and_sanitized(self):
        """Repeated pages and invalid doc labels should not waste extraction budget."""
        from src.graph import _dedupe_and_sanitize_classifications

        cleaned = _dedupe_and_sanitize_classifications([
            {"page_num": 5, "doc_type": "NURSES_NOTES", "confidence": 0.5},
            {"page_num": 5, "doc_type": "DRUG_CHART", "confidence": 0.9},
            {"page_num": 6, "doc_type": "NOT_A_TYPE", "confidence": 0.7},
        ])

        assert cleaned == [
            {"page_num": 5, "doc_type": "DRUG_CHART", "confidence": 0.9},
            {"page_num": 6, "doc_type": "UNKNOWN", "confidence": 0.7},
        ]

    def test_classification_cache_roundtrip(self, monkeypatch, tmp_path):
        """Cached classifications should reload when page coverage matches."""
        import src.graph as graph

        monkeypatch.setenv("OCR_CACHE_DIR", str(tmp_path))
        classifications = [
            {"page_num": 1, "doc_type": "NURSES_NOTES", "confidence": 0.8},
            {"page_num": 2, "doc_type": "DRUG_CHART", "confidence": 0.9},
        ]

        graph._save_classification_cache("pdf_hash", classifications)
        cached = graph._load_classification_cache("pdf_hash", [1, 2])

        assert cached == [
            {"page_num": 1, "doc_type": "NURSING_NOTES", "confidence": 0.8},
            {"page_num": 2, "doc_type": "DRUG_CHART", "confidence": 0.9},
        ]

    def test_unknown_pages_respect_batch_size(self):
        """Fallback UNKNOWN extraction batches must remain bounded."""
        from src.config import BATCH_SIZE
        from src.graph import _build_processing_queue

        docs = [
            {"page_num": i, "source_type": "UNKNOWN"}
            for i in range(1, BATCH_SIZE + 3)
        ]
        queue = _build_processing_queue(docs)

        assert queue[0]["doc_type"] == "UNKNOWN"
        assert len(queue[0]["pages"]) == BATCH_SIZE
        assert queue[1]["pages"] == [BATCH_SIZE + 1, BATCH_SIZE + 2]

    def test_unreadable_pages_are_not_duplicated(self, monkeypatch, empty_state: dict):
        """A page already marked unreadable during classification should not be duplicated."""
        import src.graph as graph

        monkeypatch.setattr(
            graph,
            "extract_page_text",
            lambda img_b64, page_num: {"text": "", "confidence": 0.0, "page_num": page_num},
        )

        empty_state.update({
            "_target_doc_type": "UNKNOWN",
            "_target_pages": [1],
            "page_images": {1: "img1"},
            "unreadable_pages": [1],
        })

        updates = graph._do_extraction(
            empty_state,
            step=1,
            updates={"steps_remaining": 10, "current_phase": "OBSERVE"},
        )

        assert updates.get("unreadable_pages", []) == []

    def test_drug_chart_uses_batch_extractor(self, monkeypatch, empty_state: dict):
        """Drug chart pages should be extracted with one batch call."""
        import src.graph as graph

        calls = []

        def fake_batch(page_texts, page_images=None):
            calls.append((page_texts, page_images))
            return {
                "medications": [
                    {
                        "name": "INJ MEROPENEM",
                        "dose": "1g",
                        "route": "IV",
                        "frequency": "TID",
                        "status": "INPATIENT_ONLY",
                        "change_reason": None,
                        "change_reason_documented": False,
                    }
                ]
            }

        monkeypatch.setattr(
            graph,
            "extract_page_text",
            lambda img_b64, page_num: {
                "text": f"Drug chart page {page_num} with medication rows clearly visible.",
                "confidence": 0.9,
                "page_num": page_num,
            },
        )
        monkeypatch.setattr(graph, "extract_drug_chart_batch", fake_batch)

        empty_state.update({
            "_target_doc_type": "DRUG_CHART",
            "_target_pages": [42, 43, 44],
            "page_images": {42: "img42", 43: "img43", 44: "img44"},
        })

        updates = graph._do_extraction(
            empty_state,
            step=1,
            updates={"steps_remaining": 10, "current_phase": "OBSERVE"},
        )

        assert len(calls) == 1
        assert [page for page, _ in calls[0][0]] == [42, 43, 44]
        assert updates["inpatient_medications"][0]["name"] == "INJ MEROPENEM"

    def test_typed_summary_medications_route_to_discharge(self, empty_state: dict):
        """Medication advice on typed discharge summary belongs in discharge meds."""
        import src.graph as graph

        extracted = {
            "medications": [
                {"name": "TAB RACIPER", "dose": "40mg", "route": "PO", "frequency": "OD"}
            ]
        }
        updates = {}
        new_meds = []

        graph._route_extracted_data(
            empty_state,
            extracted,
            "TYPED_DISCHARGE_SUMMARY",
            2,
            updates,
            [],
            [],
            [],
            new_meds,
        )

        assert new_meds[0]["name"] == "TAB RACIPER"
        assert new_meds[0]["status"] == "DISCHARGE"
        assert new_meds[0]["source_pages"] == [2]


# ─── TEST: Ollama Backup & OCR Caching ──────────────────────────────────────────

class TestOllamaBackup:
    """Tests for Ollama backup and OCR text caching."""

    def test_text_only_backup_refuses_image_input(self, monkeypatch):
        """A non-vision fallback must not hallucinate OCR for image inputs."""
        import src.tools as tools

        class FailingGemini:
            def invoke(self, messages):
                raise RuntimeError("quota exhausted")

        monkeypatch.setattr(tools, "GOOGLE_API_KEY", "real-looking-key")
        monkeypatch.setattr(tools, "_llm", None)
        monkeypatch.setattr(tools, "get_llm", lambda: FailingGemini())

        with pytest.raises(RuntimeError, match="text-only"):
            tools._call_vision_llm("classify this page", ["fake-image-b64"])

    def test_classify_page_from_text_fallback(self):
        """Test classifying a page from its text using Ollama."""
        # Simple test text representing nursing notes
        test_text = "Patient observed by nurse at bedside. Temp 98.6F, BP 120/80. Administered paracetamol."
        
        # Check if Ollama is available
        import http.client
        ollama_available = False
        try:
            conn = http.client.HTTPConnection("localhost", 11434, timeout=2)
            conn.request("GET", "/")
            res = conn.getresponse()
            if res.status == 200:
                ollama_available = True
        except Exception:
            pass
            
        if not ollama_available:
            pytest.skip("Ollama is not running locally. Skipping test.")
            
        from src.tools import classify_page_from_text
        doc_type = classify_page_from_text(test_text)
        from src.config import DOC_TYPES
        assert doc_type in DOC_TYPES or doc_type == "UNKNOWN"

    def test_ocr_caching(self):
        """Test that OCR text is cached locally and loaded correctly."""
        import os
        from src.tools import extract_page_text

        os.environ.pop("OCR_CACHE_NAMESPACE", None)
        
        cache_file = "patient2_page_999.txt"
        if os.path.exists(cache_file):
            os.remove(cache_file)
            
        try:
            # Pre-populate the cache file
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write("=== PAGE 999 ===\n\nCached clinical content that is longer than 30 characters.")
                
            # Calling extract_page_text should load it from cache immediately without calling LLM
            res = extract_page_text("dummy_image_data", 999)
            assert res["confidence"] == 1.0
            assert "Cached clinical content" in res["text"]
            assert res["page_num"] == 999
        finally:
            if os.path.exists(cache_file):
                os.remove(cache_file)

    def test_ocr_cache_namespace_avoids_legacy_file(self, monkeypatch, tmp_path):
        """Namespaced cache paths avoid stale patient2_page_N.txt reuse."""
        from src.tools import get_ocr_cache_file

        monkeypatch.setenv("OCR_CACHE_NAMESPACE", "patient_pdf_hash")
        monkeypatch.setenv("OCR_CACHE_DIR", str(tmp_path))

        cache_file = get_ocr_cache_file(7)
        assert cache_file == tmp_path / "patient_pdf_hash" / "page_7.txt"


# ─── RUN ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
