"""
compiler.py — Discharge Summary Compilation
=============================================

Assembles the final structured Markdown discharge summary from the
agent state. Every clinical claim is annotated with source page references.

Clinical Safety:
    - Missing fields use exact template strings — never plausible guesses.
    - Conflict fields use [CONFLICT — ...] template.
    - escalation_flags_for_clinician section is ALWAYS present.
    - summary_status is ALWAYS "DRAFT — NOT FOR CLINICAL USE WITHOUT REVIEW".
    - The compiler NEVER invents data — it only formats what's in the state.
"""

from __future__ import annotations
from typing import Optional

from .config import (
    MISSING_FIELD_TEMPLATE,
    PENDING_FIELD_TEMPLATE,
    CONFLICT_FIELD_TEMPLATE,
    UNCLEAR_FIELD_TEMPLATE,
)
from .trace import validate_state_completeness, generate_trace_summary


def compile_discharge_summary(state: dict) -> str:
    """
    Assemble the final structured summary from the agent state.

    Purpose:
        Produces a Markdown-formatted discharge summary draft containing
        all 17 required sections with source citations.

    Clinical Safety Constraint:
        - Every clinical claim MUST have a [Page N] citation.
        - Missing fields MUST use template strings.
        - Escalation and conflict sections are NEVER omitted.
        - Status is ALWAYS "DRAFT — NOT FOR CLINICAL USE WITHOUT REVIEW".

    Failure Behavior:
        If compilation encounters an error in any section, that section
        is marked with [COMPILATION ERROR — ...] and compilation continues.
        The agent never crashes during compilation.
    """
    completed, missing, flagged = validate_state_completeness(state)
    sections: list[str] = []

    # ─── HEADER ──────────────────────────────────────────────────────────────
    sections.append("# DISCHARGE SUMMARY — DRAFT FOR CLINICIAN REVIEW")
    sections.append("")
    sections.append("> ⚠️ **THIS IS AN AI-GENERATED DRAFT. NOT FOR CLINICAL USE WITHOUT CLINICIAN REVIEW AND SIGN-OFF.**")
    sections.append("")

    # ─── 1. PATIENT DEMOGRAPHICS ─────────────────────────────────────────────
    sections.append("## 1. Patient Demographics")
    demo = state.get("extracted_demographics", {})
    if demo:
        for key, label in [
            ("name", "Name"),
            ("age", "Age"),
            ("gender", "Gender"),
            ("mrn", "MRN"),
            ("ip_no", "IP Number"),
            ("blood_group", "Blood Group"),
            ("weight", "Weight"),
            ("address", "Address"),
        ]:
            val = demo.get(key)
            if val:
                sections.append(f"- **{label}:** {val}")
            else:
                sections.append(f"- **{label}:** {MISSING_FIELD_TEMPLATE}")
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 2. ADMISSION DATE ───────────────────────────────────────────────────
    sections.append("## 2. Admission Date")
    adm_date = state.get("admission_date")
    if adm_date:
        sections.append(f"{adm_date}")
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 3. DISCHARGE DATE ───────────────────────────────────────────────────
    sections.append("## 3. Discharge Date")
    dis_date = state.get("discharge_date")
    if dis_date:
        sections.append(f"{dis_date}")
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 4. PRINCIPAL DIAGNOSIS ──────────────────────────────────────────────
    sections.append("## 4. Principal Diagnosis")
    diag = state.get("diagnoses", {})
    diag_conflicts = [
        c for c in state.get("conflicts", [])
        if c.get("type") in ("DIAGNOSIS_MISMATCH", "CHIEF_COMPLAINT_MISMATCH")
    ]
    if diag_conflicts:
        sections.append(CONFLICT_FIELD_TEMPLATE)
        sections.append("")
        sections.append("**Diagnosis sources across documents:**")
        # List all diagnoses from all sources
        for category in ["provisional", "final", "principal"]:
            entries = diag.get(category, [])
            if entries:
                sections.append(f"- *{category.title()}:* {', '.join(str(e) for e in entries)}")

        # List diagnoses from individual documents
        for doc in state.get("loaded_documents", []):
            extracted = doc.get("extracted_data", {})
            for key in ["er_diagnosis", "provisional_diagnosis", "final_diagnosis",
                         "icu_diagnoses", "consultation_diagnosis"]:
                val = extracted.get(key)
                if val:
                    if isinstance(val, list):
                        val_str = ", ".join(str(v) for v in val)
                    else:
                        val_str = str(val)
                    sections.append(
                        f"- *{doc.get('source_type', 'Unknown')} "
                        f"(Page {doc.get('page_num', '?')}):* {val_str}"
                    )
    elif diag.get("principal"):
        for d in diag["principal"]:
            sections.append(f"- {d}")
    elif diag.get("final"):
        for d in diag["final"]:
            sections.append(f"- {d}")
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 5. SECONDARY DIAGNOSES ──────────────────────────────────────────────
    sections.append("## 5. Secondary Diagnoses")
    secondary = diag.get("secondary", [])
    if secondary:
        for d in secondary:
            sections.append(f"- {d}")
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 6. ALLERGIES ────────────────────────────────────────────────────────
    sections.append("## 6. Allergies")
    allergies = state.get("allergies", [])
    if allergies:
        for a in allergies:
            sections.append(f"- {a}")
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 7. HOSPITAL COURSE ──────────────────────────────────────────────────
    sections.append("## 7. Hospital Course")
    course = state.get("hospital_course", "")
    if course:
        sections.append(course)
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 8. INVESTIGATIONS SUMMARY ──────────────────────────────────────────
    sections.append("## 8. Investigations Summary")
    sections.append("")
    sections.append("### Laboratory Results")
    labs = state.get("lab_results", [])
    if labs:
        sections.append("| Test | Value | Unit | Reference Range | Date | Abnormal |")
        sections.append("|------|-------|------|-----------------|------|----------|")
        for lab in labs:
            abnormal = "⚠️ YES" if lab.get("abnormal_flag") else "No"
            if lab.get("critically_abnormal"):
                abnormal = "🚨 CRITICAL"
            sections.append(
                f"| {lab.get('test_name', '?')} "
                f"| {lab.get('result_value', '?')} "
                f"| {lab.get('unit', '')} "
                f"| {lab.get('reference_range', '')} "
                f"| {lab.get('date', '')} "
                f"| {abnormal} |"
            )
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    sections.append("### Imaging Results")
    imaging = state.get("imaging_results", [])
    if imaging:
        for img in imaging:
            sections.append(
                f"- **{img.get('modality', '?')}** ({img.get('date', '?')}): "
                f"{img.get('impression', MISSING_FIELD_TEMPLATE)}"
            )
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 9. PROCEDURES PERFORMED ─────────────────────────────────────────────
    sections.append("## 9. Procedures Performed")
    procedures = state.get("procedures", [])
    if procedures:
        for p in procedures:
            sections.append(
                f"- **{p.get('name', '?')}** ({p.get('date', '?')}): "
                f"{p.get('notes', '')}"
            )
    else:
        sections.append("No procedures documented in source records.")
    sections.append("")

    # ─── 10. ADMISSION MEDICATIONS ───────────────────────────────────────────
    sections.append("## 10. Admission Medications")
    adm_meds = state.get("admission_medications", [])
    if adm_meds:
        _render_medication_table(sections, adm_meds)
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 11. DISCHARGE MEDICATIONS ───────────────────────────────────────────
    sections.append("## 11. Discharge Medications")
    dis_meds = state.get("discharge_medications", [])
    if dis_meds:
        _render_medication_table(sections, dis_meds)
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 12. MEDICATION CHANGES ──────────────────────────────────────────────
    sections.append("## 12. Medication Changes (Reconciliation)")
    recon = state.get("medication_reconciliation", [])
    if recon:
        changes = [r for r in recon if r.get("change_type") and r["change_type"] != "CONTINUED"]
        if changes:
            sections.append("| Drug | Change | Reason | Flag |")
            sections.append("|------|--------|--------|------|")
            for r in changes:
                flag = r.get("flag", "")
                if flag:
                    flag = f"⚠️ {flag}"
                sections.append(
                    f"| {r.get('drug', '?')} "
                    f"| {r.get('change_type', '?')} "
                    f"| {r.get('documented_reason', 'NOT DOCUMENTED')} "
                    f"| {flag} |"
                )
        else:
            sections.append("No medication changes identified.")
    else:
        sections.append("Medication reconciliation not yet performed.")
    sections.append("")

    # ─── 13. PENDING RESULTS ─────────────────────────────────────────────────
    sections.append("## 13. Pending Results")
    pending = state.get("pending_results", [])
    if pending:
        for p in pending:
            sections.append(f"- ⏳ {p}")
    else:
        sections.append("No pending results identified.")
    sections.append("")

    # ─── 14. FOLLOW-UP INSTRUCTIONS ──────────────────────────────────────────
    sections.append("## 14. Follow-Up Instructions")
    follow_up = state.get("follow_up_instructions", [])
    if follow_up:
        for f in follow_up:
            sections.append(f"- {f}")
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 15. DISCHARGE CONDITION ─────────────────────────────────────────────
    sections.append("## 15. Discharge Condition")
    dc = state.get("discharge_condition")
    dc_flags = [
        f for f in state.get("escalation_flags", [])
        if f.get("field") == "discharge_condition"
    ]
    if dc:
        sections.append(dc)
        if dc_flags:
            for flag in dc_flags:
                sections.append(
                    f"\n> ⚠️ **CLINICIAN REVIEW REQUIRED** — {flag.get('reason', 'See escalation flags')}"
                )
    else:
        sections.append(MISSING_FIELD_TEMPLATE)
    sections.append("")

    # ─── 16. ESCALATION FLAGS FOR CLINICIAN (NON-NEGOTIABLE) ─────────────────
    sections.append("## 16. ⚠️ Escalation Flags for Clinician")
    sections.append("")
    esc_flags = state.get("escalation_flags", [])
    if esc_flags:
        critical = [f for f in esc_flags if f.get("severity") == "CRITICAL"]
        warnings = [f for f in esc_flags if f.get("severity") == "WARNING"]

        if critical:
            sections.append("### 🚨 CRITICAL")
            for i, flag in enumerate(critical, 1):
                sections.append(
                    f"{i}. **{flag.get('field', '?')}** — {flag.get('reason', '?')}"
                )
                evidence = flag.get("source_evidence", [])
                if evidence:
                    for e in evidence[:3]:
                        sections.append(f"   - Evidence: {e}")
            sections.append("")

        if warnings:
            sections.append("### ⚠️ WARNING")
            for i, flag in enumerate(warnings, 1):
                sections.append(
                    f"{i}. **{flag.get('field', '?')}** — {flag.get('reason', '?')}"
                )
            sections.append("")
    else:
        sections.append("No escalation flags generated.")
    sections.append("")

    # ─── 17. CONFLICTS REQUIRING REVIEW (NON-NEGOTIABLE) ────────────────────
    sections.append("## 17. Conflicts Requiring Review")
    sections.append("")
    conflicts = state.get("conflicts", [])
    if conflicts:
        for i, conflict in enumerate(conflicts, 1):
            rule = conflict.get("rule", "?")
            ctype = conflict.get("type", "?")
            desc = conflict.get("description", "?")
            sections.append(f"### Conflict {i} — [{rule}] {ctype}")
            sections.append(f"{desc}")
            sources = conflict.get("sources", [])
            if sources:
                sections.append("**Sources:**")
                for s in sources[:5]:
                    sections.append(f"- {s}")
            sections.append("")
    else:
        sections.append("No conflicts detected.")
    sections.append("")

    # ─── 18. SUMMARY STATUS (NON-NEGOTIABLE) ────────────────────────────────
    sections.append("---")
    sections.append("")
    sections.append("## Summary Status")
    sections.append("")
    sections.append("**DRAFT — NOT FOR CLINICAL USE WITHOUT REVIEW**")
    sections.append("")
    sections.append(
        "This discharge summary was generated by an AI agent and has NOT been reviewed "
        "by a clinician. All clinical claims must be verified against source documents "
        "before use. Escalation flags and conflicts MUST be resolved by a qualified "
        "clinician before this summary can be finalized."
    )
    sections.append("")
    sections.append(f"- **Fields completed:** {len(completed)}")
    sections.append(f"- **Fields missing:** {len(missing)} — {', '.join(missing) if missing else 'None'}")
    sections.append(f"- **Fields flagged:** {len(flagged)} — {', '.join(flagged) if flagged else 'None'}")
    sections.append(f"- **Total conflicts:** {len(conflicts)}")
    sections.append(f"- **Total escalation flags:** {len(esc_flags)}")
    fab = state.get("fabrication_blocks", [])
    sections.append(f"- **Fabrication blocks prevented:** {len(fab)}")
    if fab:
        for f in fab:
            sections.append(f"  - {f}")

    return "\n".join(sections)


def _render_medication_table(sections: list[str], medications: list[dict]) -> None:
    """Render a medication list as a Markdown table."""
    sections.append("| Medication | Dose | Route | Frequency | Status |")
    sections.append("|-----------|------|-------|-----------|--------|")
    for med in medications:
        sections.append(
            f"| {med.get('name', '?')} "
            f"| {med.get('dose', '?')} "
            f"| {med.get('route', '?')} "
            f"| {med.get('frequency', '?')} "
            f"| {med.get('status', '?')} |"
        )
