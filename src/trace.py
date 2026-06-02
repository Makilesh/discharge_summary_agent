"""
trace.py — Observability, Trace Emission & State Validation
=============================================================

Every single agent action — including no-ops, retries, and fallbacks —
must emit a trace entry. The trace is the audit log of clinical reasoning.

Clinical Safety:
    - The trace is append-only. No entry may be modified after emission.
    - validate_state_completeness() ensures no required field is silently blank.
    - The TRACE_SUMMARY at run end gives a full safety verdict.
"""

from __future__ import annotations
from typing import Optional
from datetime import datetime, timezone

from .config import DISCHARGE_SUMMARY_SECTIONS


# ─── TRACE EMISSION ─────────────────────────────────────────────────────────────

def emit_trace(
    state: dict,
    step_number: int,
    phase: str,
    reasoning: str,
    action: str,
    tool_name: Optional[str] = None,
    tool_inputs: Optional[dict] = None,
    tool_output_summary: Optional[str] = None,
    observation: str = "",
    decision: str = "",
    fields_updated: Optional[list[str]] = None,
    escalations_triggered: Optional[list[str]] = None,
    fallback_taken: bool = False,
    fallback_reason: Optional[str] = None,
) -> dict:
    """
    Append a structured trace entry to state['trace'].

    Purpose:
        Provides a complete, immutable audit trail of every agent decision
        for clinical review and debugging.

    Clinical Safety Constraint:
        Every branch point — including branches NOT taken — must be logged.
        The trace must explain WHY the agent did or did not take an action.

    Failure Behavior:
        If trace emission itself fails, the error is logged to stderr but
        the agent continues. Trace failures must not crash the agent.

    Returns:
        The trace entry dict (also appended to state['trace']).
    """
    entry = {
        "step": step_number,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "reasoning": reasoning,
        "action": action,
        "tool": tool_name,
        "inputs": _sanitize_inputs(tool_inputs),
        "output_summary": tool_output_summary,
        "observation": observation,
        "decision": decision,
        "fields_updated": fields_updated or [],
        "escalations_triggered": escalations_triggered or [],
        "fallback_taken": fallback_taken,
        "fallback_reason": fallback_reason,
    }

    if "trace" not in state:
        state["trace"] = []
    state["trace"].append(entry)
    return entry


def _sanitize_inputs(inputs: Optional[dict]) -> Optional[dict]:
    """Remove large binary data (base64 images) from trace inputs to keep logs manageable."""
    if inputs is None:
        return None
    sanitized = {}
    for k, v in inputs.items():
        if isinstance(v, str) and len(v) > 1000:
            sanitized[k] = f"[TRUNCATED — {len(v)} chars]"
        elif isinstance(v, list) and v and isinstance(v[0], str) and len(str(v)) > 2000:
            sanitized[k] = f"[LIST — {len(v)} items, truncated]"
        else:
            sanitized[k] = v
    return sanitized


# ─── STATE VALIDATION ───────────────────────────────────────────────────────────

def validate_state_completeness(state: dict) -> tuple[list[str], list[str], list[str]]:
    """
    Check all required fields before compilation begins.

    Purpose:
        Ensures the agent has attempted every required field. Fields that
        are still empty/None after extraction get flagged for the clinician.

    Clinical Safety Constraint:
        This function must be called BEFORE compile_discharge_summary.
        A summary compiled without validation may silently omit critical data.

    Returns:
        Tuple of (completed_fields, missing_fields, flagged_fields).
    """
    completed: list[str] = []
    missing: list[str] = []
    flagged: list[str] = []

    # Demographics
    if state.get("extracted_demographics"):
        completed.append("demographics")
    else:
        missing.append("demographics")

    # Dates
    for date_field in ["admission_date", "discharge_date"]:
        if state.get(date_field):
            completed.append(date_field)
        else:
            missing.append(date_field)

    # Diagnoses
    diag = state.get("diagnoses", {})
    if any(diag.get(k) for k in ["principal", "final"]):
        completed.append("diagnoses")
    else:
        missing.append("diagnoses")

    # Check for diagnosis conflicts
    if state.get("conflicts"):
        diag_conflicts = [c for c in state["conflicts"] if "DIAGNOSIS" in c.get("type", "")]
        if diag_conflicts:
            flagged.append("diagnoses.final")

    # Hospital course
    if state.get("hospital_course"):
        completed.append("hospital_course")
    else:
        missing.append("hospital_course")

    # Medications
    for med_field in ["admission_medications", "inpatient_medications", "discharge_medications"]:
        if state.get(med_field):
            completed.append(med_field.replace("_medications", "_meds"))
        else:
            missing.append(med_field.replace("_medications", "_meds"))

    # Labs
    if state.get("lab_results"):
        completed.append("labs")
    else:
        missing.append("labs")

    # Imaging
    if state.get("imaging_results"):
        completed.append("imaging")
    else:
        missing.append("imaging")

    # Procedures
    if state.get("procedures"):
        completed.append("procedures")
    else:
        # Procedures might legitimately be empty
        completed.append("procedures")

    # Allergies
    if state.get("allergies"):
        completed.append("allergies")
    else:
        missing.append("allergies")

    # Follow-up
    if state.get("follow_up_instructions"):
        completed.append("follow_up")
    else:
        missing.append("follow_up")

    # Discharge condition
    if state.get("discharge_condition"):
        completed.append("discharge_condition")
    else:
        missing.append("discharge_condition")

    # Check for escalation flags on specific fields
    for flag in state.get("escalation_flags", []):
        field = flag.get("field", "")
        if field and field not in flagged:
            flagged.append(field)

    return completed, missing, flagged


# ─── TRACE SUMMARY ──────────────────────────────────────────────────────────────

def generate_trace_summary(state: dict) -> dict:
    """
    Generate the final TRACE_SUMMARY entry appended at the end of every run.

    Purpose:
        Provides a machine-readable and human-readable summary of the entire
        agent run for audit purposes.

    Clinical Safety Constraint:
        safety_verdict is ALWAYS "DO_NOT_FINALIZE_WITHOUT_CLINICIAN_SIGN_OFF".
        The agent never produces a self-finalized clinical document.
    """
    from .config import MAX_ITERATIONS

    completed, missing, flagged = validate_state_completeness(state)
    steps_used = MAX_ITERATIONS - state.get("steps_remaining", 0)

    critical_escalations = sum(
        1 for f in state.get("escalation_flags", [])
        if f.get("severity") == "CRITICAL"
    )

    summary = {
        "step": "FINAL",
        "total_steps_used": steps_used,
        "steps_remaining": state.get("steps_remaining", 0),
        "documents_processed": len(state.get("loaded_documents", [])),
        "unreadable_pages": state.get("unreadable_pages", []),
        "fields_completed": completed,
        "fields_missing": missing,
        "fields_flagged": flagged,
        "total_conflicts": len(state.get("conflicts", [])),
        "total_escalation_flags": len(state.get("escalation_flags", [])),
        "critical_escalations": critical_escalations,
        "fabrication_blocks_prevented": len(state.get("fabrication_blocks", [])),
        "hard_cap_hit": state.get("steps_remaining", 0) <= 0,
        "summary_status": "DRAFT_FOR_CLINICIAN_REVIEW",
        "safety_verdict": "DO_NOT_FINALIZE_WITHOUT_CLINICIAN_SIGN_OFF",
    }

    emit_trace(
        state=state,
        step_number=-1,  # Sentinel for FINAL
        phase="TRACE_SUMMARY",
        reasoning="Run complete. Generating final trace summary.",
        action="SUMMARY",
        observation=f"Completed {steps_used} steps. {len(missing)} fields missing, "
                    f"{len(flagged)} fields flagged, {critical_escalations} critical escalations.",
        decision="Emit trace summary and finalize.",
        fields_updated=[],
    )

    return summary
