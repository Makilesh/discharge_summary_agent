"""
state.py — Agent State Schema & Type Definitions
=================================================

Defines all typed data structures for the Discharge Summary Agent.
Every field is explicitly typed to enforce clinical data integrity.

Clinical Safety Constraint:
    - All Optional fields default to None, never to plausible guesses.
    - Lists default to empty, never to synthetic data.
    - The state is the single source of truth for the agent's knowledge.
"""

from __future__ import annotations
from typing import TypedDict, Literal, Optional, Annotated
import operator


# ─── CLINICAL FLAG ──────────────────────────────────────────────────────────────

class ClinicalFlag(TypedDict):
    """A flag requiring clinician attention. Never auto-resolved by the agent."""
    field: str
    severity: Literal["WARNING", "CRITICAL"]
    reason: str
    source_page: Optional[int]
    requires_clinician: bool


# ─── MEDICATION ENTRY ───────────────────────────────────────────────────────────

class MedicationEntry(TypedDict):
    """A single medication record extracted from any source document.
    
    Clinical Safety: If change_reason_documented is False, the medication
    reconciliation step MUST flag this for clinician review — a medication
    change without documented rationale is a safety concern.
    """
    name: str
    dose: Optional[str]
    route: Optional[str]
    frequency: Optional[str]
    start_date: Optional[str]
    stop_date: Optional[str]
    status: Literal["ADMISSION", "INPATIENT_ONLY", "DISCHARGE", "UNKNOWN"]
    change_reason: Optional[str]
    change_reason_documented: bool  # False triggers reconciliation flag


# ─── AGENT STATE ────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    The complete mutable state of the Discharge Summary Agent.
    
    This is the single source of truth. Every extraction, conflict, and
    escalation lives here. The state is passed through every node in the
    LangGraph state machine.
    
    Clinical Safety Invariant:
        - No field may be populated with data not sourced from documents.
        - Conflicts are never silently resolved; they accumulate.
        - escalation_flags is append-only within a run.
    """

    # ─── DOCUMENT INVENTORY ──────────────────────────────────────────────────
    _pdf_path: str
    loaded_documents: Annotated[list[dict], operator.add]
    # Each: {page_num: int, source_type: str, raw_text: str, confidence: float}
    unreadable_pages: Annotated[list[int], operator.add]
    # Pages where OCR/extraction returned < 30 chars
    page_images: dict  # {page_num: base64_encoded_image_string}

    # ─── ITERATION CONTROL ───────────────────────────────────────────────────
    steps_remaining: int                # Initialize to MAX_ITERATIONS (20)
    current_phase: str                  # REASON | PLAN | CALL_TOOL | OBSERVE | VERIFY | COMPILE
    retry_counts: dict                  # {"tool_name": retry_count}
    processing_queue: list[dict]        # Pages grouped by type, in priority order

    # ─── EXTRACTED CLINICAL FIELDS ───────────────────────────────────────────
    extracted_demographics: dict
    # name, age, gender, mrn, ip_no, blood_group, weight
    admission_date: Optional[str]
    discharge_date: Optional[str]
    diagnoses: dict
    # {"principal": [], "secondary": [], "provisional": [], "final": []}
    hospital_course: str                # Synthesized narrative, never invented
    procedures: Annotated[list[dict], operator.add]
    # Each: {name: str, date: str, notes: str}
    allergies: list[str]
    # "NOT KNOWN" is a valid documented value
    discharge_condition: Optional[str]

    # ─── MEDICATION RECONCILIATION ───────────────────────────────────────────
    admission_medications: Annotated[list[MedicationEntry], operator.add]
    inpatient_medications: Annotated[list[MedicationEntry], operator.add]
    discharge_medications: Annotated[list[MedicationEntry], operator.add]
    medication_reconciliation: Annotated[list[dict], operator.add]
    # Each: {drug, change_type, documented_reason, flag}

    # ─── INVESTIGATIONS ──────────────────────────────────────────────────────
    lab_results: Annotated[list[dict], operator.add]
    # Each: {test, value, unit, ref_range, date, abnormal_flag}
    imaging_results: Annotated[list[dict], operator.add]
    # Each: {modality, date, impression}
    pending_results: Annotated[list[str], operator.add]
    # e.g. ["Blood C/S sent 27/2/26 — result not found in documents"]

    # ─── FOLLOW-UP ───────────────────────────────────────────────────────────
    follow_up_instructions: Annotated[list[str], operator.add]

    # ─── SAFETY & ESCALATION ─────────────────────────────────────────────────
    conflicts: Annotated[list[dict], operator.add]
    # Each: {type, sources, description, resolution: "ESCALATED"}
    escalation_flags: Annotated[list[ClinicalFlag], operator.add]
    fabrication_blocks: Annotated[list[str], operator.add]
    # Any field where the agent was tempted to guess

    # ─── OBSERVABILITY ───────────────────────────────────────────────────────
    trace: Annotated[list[dict], operator.add]
    # Full step-by-step JSON trace — see Section 4
    final_summary: Optional[str]
    # The compiled output


# ─── STATE FACTORY ──────────────────────────────────────────────────────────────

def create_initial_state() -> dict:
    """
    Create a fresh initial state with all fields properly initialized.
    
    Returns a plain dict (not TypedDict) so LangGraph can merge updates.
    All lists are empty, all Optionals are None. No data is invented.
    """
    from .config import MAX_ITERATIONS

    return {
        # Document inventory
        "_pdf_path": "",
        "loaded_documents": [],
        "unreadable_pages": [],
        "page_images": {},

        # Iteration control
        "steps_remaining": MAX_ITERATIONS,
        "current_phase": "INITIALIZE",
        "retry_counts": {},
        "processing_queue": [],

        # Clinical fields
        "extracted_demographics": {},
        "admission_date": None,
        "discharge_date": None,
        "diagnoses": {
            "principal": [],
            "secondary": [],
            "provisional": [],
            "final": [],
        },
        "hospital_course": "",
        "procedures": [],
        "allergies": [],
        "discharge_condition": None,

        # Medications
        "admission_medications": [],
        "inpatient_medications": [],
        "discharge_medications": [],
        "medication_reconciliation": [],

        # Investigations
        "lab_results": [],
        "imaging_results": [],
        "pending_results": [],

        # Follow-up
        "follow_up_instructions": [],

        # Safety
        "conflicts": [],
        "escalation_flags": [],
        "fabrication_blocks": [],

        # Observability
        "trace": [],
        "final_summary": None,
    }
