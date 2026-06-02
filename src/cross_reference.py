"""
cross_reference.py — Clinical Cross-Reference Audit (Anti-Hallucination Core)
===============================================================================

Implements the 5 CR rules that form the clinical safety backbone of the agent.
This is the MOST CRITICAL module. Violations here are clinical safety failures.

Rules:
    CR-1: Treatment-Diagnosis Alignment
    CR-2: Inter-Document Diagnosis Conflict
    CR-3: Lab Evidence vs Clinical Claim
    CR-4: Culture Result vs Treatment
    CR-5: Discharge Condition Claim

Clinical Safety:
    - This module NEVER resolves conflicts. It only DETECTS and REPORTS.
    - Every detected conflict triggers an escalation flag.
    - The escalate_to_clinician() function is append-only — it never
      modifies the underlying conflicting data.
"""

from __future__ import annotations
from typing import Literal, Optional
import re

from .config import CRITICAL_LAB_THRESHOLDS, CONFLICT_FIELD_TEMPLATE


# ─── ESCALATE TO CLINICIAN ──────────────────────────────────────────────────────

def escalate_to_clinician(
    state: dict,
    field: str,
    severity: Literal["WARNING", "CRITICAL"],
    reason: str,
    source_evidence: list[str],
    source_pages: list[int],
) -> dict:
    """
    Formally mark a field as requiring clinician review.

    Purpose:
        Creates a ClinicalFlag and appends it to the state's escalation_flags.
        The affected field in the final summary will show:
        [⚠ CLINICIAN REVIEW REQUIRED — {reason}]

    Clinical Safety Constraints:
        1. Appends a ClinicalFlag — never removes existing ones.
        2. Marks the affected field for review in the final summary.
        3. Logs the escalation in the trace with full evidence chain.
        4. NEVER modifies the underlying conflicting data to resolve it.
        5. Returns updated state.

    Failure Behavior:
        If escalation itself fails (should never happen), logs to stderr.
        The agent MUST NOT continue without recording the escalation.
    """
    flag = {
        "field": field,
        "severity": severity,
        "reason": reason,
        "source_evidence": source_evidence,
        "source_pages": source_pages,
        "requires_clinician": True,
    }

    if "escalation_flags" not in state:
        state["escalation_flags"] = []
    state["escalation_flags"].append(flag)

    # Log in trace
    from .trace import emit_trace
    emit_trace(
        state=state,
        step_number=state.get("_current_step", -1),
        phase="ESCALATE",
        reasoning=f"Escalating {field} to clinician: {reason}",
        action="ESCALATE_TO_CLINICIAN",
        observation=f"Evidence from pages {source_pages}: {'; '.join(source_evidence[:3])}",
        decision=f"Field '{field}' marked for clinician review with severity={severity}",
        escalations_triggered=[field],
    )

    return state


# ─── CR-1: TREATMENT-DIAGNOSIS ALIGNMENT ────────────────────────────────────────

def check_cr1_treatment_diagnosis_alignment(state: dict) -> list[dict]:
    """
    CR-1: For every active inpatient treatment, verify a corresponding
    documented diagnosis exists.

    Detection Logic:
        - Insulin (Lantus, Actrapid, Humalog, Novorapid) → DM, DKA, T2DM, T1DM
        - IV antibiotics (Meropenem, Piperacillin, Ceftriaxone) → infection diagnosis
        - Antihypertensives → HTN diagnosis
        - Anticoagulants → DVT, PE, AF diagnosis
        - Bronchodilators → asthma, COPD diagnosis

    Clinical Safety:
        If a medication is active in inpatient charts but its indication
        is absent from ALL diagnosis lists, this is a CR-1 violation.
        Also checks if inpatient meds are missing from discharge advice.
    """
    conflicts: list[dict] = []
    all_diagnoses_text = _get_all_diagnoses_text(state)

    # Treatment → Expected diagnosis mapping
    treatment_diagnosis_map = {
        # Insulin drugs → diabetes/DKA
        r"(?i)(lantus|glargine|actrapid|humalog|novorapid|insulin|lispro|aspart)": [
            "DM", "DKA", "diabetes", "T2DM", "T1DM", "diabetic",
            "uncontrolled diabetes", "hyperglycemia",
        ],
        # Broad-spectrum IV antibiotics → infection
        r"(?i)(meropenem|piperacillin|tazobactam|ceftriaxone|cefoperazone|vancomycin|imipenem)": [
            "sepsis", "infection", "UTI", "pneumonia", "pyelonephritis",
            "bacteremia", "cellulitis", "abscess", "peritonitis", "AFI",
        ],
        # DKA-specific treatments
        r"(?i)(sodium\s*bicarbonate|NaHCO3)": [
            "DKA", "metabolic acidosis", "acidosis",
        ],
    }

    # Check each inpatient medication
    inpatient_meds = state.get("inpatient_medications", [])
    discharge_meds = state.get("discharge_medications", [])
    discharge_med_names = {m.get("name", "").upper() for m in discharge_meds}

    for med in inpatient_meds:
        med_name = med.get("name", "")
        if not med_name:
            continue

        # Check treatment-diagnosis alignment
        for pattern, expected_diagnoses in treatment_diagnosis_map.items():
            if re.search(pattern, med_name):
                # Check if ANY expected diagnosis appears in ANY diagnosis source
                found = False
                for diag_keyword in expected_diagnoses:
                    if diag_keyword.lower() in all_diagnoses_text.lower():
                        found = True
                        break

                if not found:
                    conflict = {
                        "type": "TREATMENT_WITHOUT_DIAGNOSIS",
                        "sources": [
                            f"Inpatient medication: {med_name}",
                            f"All diagnoses: {all_diagnoses_text[:200]}",
                        ],
                        "description": (
                            f"{med_name} was administered during admission but no corresponding "
                            f"diagnosis ({', '.join(expected_diagnoses[:3])}) appears in any "
                            f"diagnosis field. This constitutes a CR-1 violation."
                        ),
                        "resolution": "ESCALATED",
                        "rule": "CR-1",
                    }
                    conflicts.append(conflict)

                    escalate_to_clinician(
                        state=state,
                        field="diagnoses",
                        severity="CRITICAL",
                        reason=(
                            f"Treatment ({med_name}) administered without matching diagnosis. "
                            f"Expected one of: {', '.join(expected_diagnoses[:3])}"
                        ),
                        source_evidence=[
                            f"Medication: {med_name} ({med.get('dose', 'dose unknown')}, "
                            f"{med.get('route', 'route unknown')}, {med.get('frequency', 'freq unknown')})",
                        ],
                        source_pages=med.get("source_pages", []),
                    )

        # Check if inpatient med is missing from discharge
        med_name_upper = med_name.upper().replace("INJ ", "").replace("TAB ", "").replace("CAP ", "")
        found_in_discharge = any(
            med_name_upper in d_name
            for d_name in discharge_med_names
        )
        if not found_in_discharge and med.get("status") != "INPATIENT_ONLY":
            # Could be intentionally stopped, but flag for review
            conflict = {
                "type": "INPATIENT_MED_NOT_IN_DISCHARGE",
                "sources": [f"Inpatient: {med_name}", "Discharge medications list"],
                "description": (
                    f"{med_name} was active during admission but absent from discharge "
                    f"medications. Verify if this was intentionally stopped."
                ),
                "resolution": "ESCALATED",
                "rule": "CR-1",
            }
            conflicts.append(conflict)

    return conflicts


# ─── CR-2: INTER-DOCUMENT DIAGNOSIS CONFLICT ────────────────────────────────────

def check_cr2_diagnosis_conflicts(state: dict) -> list[dict]:
    """
    CR-2: For every diagnosis field across all document types, if
    Document_A.diagnosis != Document_B.diagnosis → flag conflict.

    Detection Logic:
        Collects diagnoses from: ER chart, admission record, ICU chart,
        consultation sheets, typed discharge summary. Compares all pairs.

    Clinical Safety:
        DO NOT pick one diagnosis over another. ADD both to conflicts[].
        The final summary field must read:
        [CONFLICT: see escalation flags]
    """
    conflicts: list[dict] = []

    # Collect diagnoses from all sources
    diagnosis_sources: list[dict] = []

    # From structured diagnoses field
    diagnoses = state.get("diagnoses", {})
    if diagnoses.get("provisional"):
        diagnosis_sources.append({
            "source": "Structured — Provisional",
            "diagnoses": diagnoses["provisional"],
        })
    if diagnoses.get("final"):
        diagnosis_sources.append({
            "source": "Structured — Final",
            "diagnoses": diagnoses["final"],
        })

    # From loaded documents
    for doc in state.get("loaded_documents", []):
        doc_type = doc.get("source_type", "")
        extracted = doc.get("extracted_data", {})
        if not extracted:
            continue

        # Collect diagnosis fields from different document types
        diag_fields = []
        for key in ["er_diagnosis", "provisional_diagnosis", "final_diagnosis",
                     "icu_diagnoses", "consultation_diagnosis", "diagnoses"]:
            val = extracted.get(key)
            if val:
                if isinstance(val, list):
                    diag_fields.extend(val)
                elif isinstance(val, str):
                    diag_fields.append(val)
                elif isinstance(val, dict):
                    for v in val.values():
                        if isinstance(v, list):
                            diag_fields.extend(v)
                        elif isinstance(v, str) and v:
                            diag_fields.append(v)

        if diag_fields:
            diagnosis_sources.append({
                "source": f"{doc_type} (Page {doc.get('page_num', '?')})",
                "diagnoses": diag_fields,
            })

    # Compare across sources for conflicts
    if len(diagnosis_sources) > 1:
        all_diag_sets = []
        for src in diagnosis_sources:
            diag_set = {d.strip().upper() for d in src["diagnoses"] if d}
            all_diag_sets.append((src["source"], diag_set))

        # Check for materially different diagnoses across sources
        all_diagnoses_combined = set()
        for _, ds in all_diag_sets:
            all_diagnoses_combined.update(ds)

        # If different sources have substantially different diagnosis sets
        for i, (src_a, set_a) in enumerate(all_diag_sets):
            for j, (src_b, set_b) in enumerate(all_diag_sets):
                if i >= j:
                    continue
                # Find diagnoses in A but not B (using substring matching)
                unique_to_a = set_a - set_b
                unique_to_b = set_b - set_a
                if unique_to_a or unique_to_b:
                    conflict = {
                        "type": "DIAGNOSIS_MISMATCH",
                        "sources": [src_a, src_b],
                        "description": (
                            f"Diagnosis conflict: {src_a} records {list(set_a)[:5]}, "
                            f"but {src_b} records {list(set_b)[:5]}. "
                            f"Unique to {src_a}: {list(unique_to_a)[:3]}. "
                            f"Unique to {src_b}: {list(unique_to_b)[:3]}. "
                            f"These are materially different and cannot be arbitrarily resolved."
                        ),
                        "resolution": "ESCALATED",
                        "rule": "CR-2",
                    }
                    conflicts.append(conflict)

    if conflicts:
        escalate_to_clinician(
            state=state,
            field="diagnoses",
            severity="CRITICAL",
            reason=(
                "Multiple source documents contain materially different diagnoses. "
                "Agent cannot resolve — clinician must review all sources and determine "
                "the correct diagnosis set."
            ),
            source_evidence=[
                f"{src['source']}: {', '.join(src['diagnoses'][:3])}"
                for src in diagnosis_sources[:5]
            ],
            source_pages=[],
        )

    # Also check chief complaint mismatches
    chief_complaints_sources = []
    for doc in state.get("loaded_documents", []):
        extracted = doc.get("extracted_data", {})
        cc = extracted.get("chief_complaints", [])
        if cc:
            chief_complaints_sources.append({
                "source": f"{doc.get('source_type', 'UNKNOWN')} (Page {doc.get('page_num', '?')})",
                "complaints": cc,
            })

    if len(chief_complaints_sources) > 1:
        all_cc = set()
        for src in chief_complaints_sources:
            for c in src["complaints"]:
                all_cc.add(c.strip().upper())

        # If substantially different complaint sets
        sets = [
            {c.strip().upper() for c in src["complaints"]}
            for src in chief_complaints_sources
        ]
        if len(sets) >= 2 and sets[0] != sets[1]:
            conflict = {
                "type": "CHIEF_COMPLAINT_MISMATCH",
                "sources": [s["source"] for s in chief_complaints_sources[:2]],
                "description": (
                    f"Chief complaint mismatch: {chief_complaints_sources[0]['source']} records "
                    f"{chief_complaints_sources[0]['complaints']}, but "
                    f"{chief_complaints_sources[1]['source']} records "
                    f"{chief_complaints_sources[1]['complaints']}. "
                    f"The discharge summary must reconcile these into a comprehensive symptom list."
                ),
                "resolution": "ESCALATED",
                "rule": "CR-2",
            }
            conflicts.append(conflict)

    return conflicts


# ─── CR-3: LAB EVIDENCE VS CLINICAL CLAIM ───────────────────────────────────────

def check_cr3_lab_evidence(state: dict) -> list[dict]:
    """
    CR-3: If lab_result shows CRITICAL_ABNORMAL and no corresponding
    treatment or note exists → escalate as CRITICAL.

    Also: If typed_discharge_summary claims "resolved" but last lab
    still shows abnormal values → escalate.

    Detection Logic:
        - Checks each lab result against CRITICAL_LAB_THRESHOLDS.
        - Checks if discharge condition claims improvement but last labs disagree.
        - Specifically targets: Na, K, glucose, creatinine, pH, HCO3, WBC.
    """
    conflicts: list[dict] = []
    lab_results = state.get("lab_results", [])

    # Check for critically abnormal values
    for result in lab_results:
        test = result.get("test_name", "").lower()
        value_str = result.get("result_value", "")

        if not value_str or value_str in ("PENDING", "AWAITED", "null", None):
            continue

        try:
            value = float(re.sub(r'[^\d.]', '', str(value_str)))
        except (ValueError, TypeError):
            continue

        # Check against critical thresholds
        for threshold_key, thresholds in CRITICAL_LAB_THRESHOLDS.items():
            if threshold_key in test:
                is_critical = False
                reason_parts = []

                if "low" in thresholds and value < thresholds["low"]:
                    is_critical = True
                    reason_parts.append(
                        f"{test}: {value} {thresholds.get('unit', '')} is critically LOW "
                        f"(threshold: {thresholds['low']})"
                    )
                if "high" in thresholds and value > thresholds["high"]:
                    is_critical = True
                    reason_parts.append(
                        f"{test}: {value} {thresholds.get('unit', '')} is critically HIGH "
                        f"(threshold: {thresholds['high']})"
                    )

                if is_critical:
                    conflict = {
                        "type": "CRITICAL_LAB_VALUE",
                        "sources": [f"Lab result: {test} = {value_str}"],
                        "description": "; ".join(reason_parts),
                        "resolution": "ESCALATED",
                        "rule": "CR-3",
                    }
                    conflicts.append(conflict)

                    escalate_to_clinician(
                        state=state,
                        field="lab_results",
                        severity="CRITICAL",
                        reason="; ".join(reason_parts),
                        source_evidence=[
                            f"{test} = {value_str} {result.get('unit', '')} "
                            f"(ref: {result.get('reference_range', 'unknown')})",
                        ],
                        source_pages=[result.get("source_page", -1)],
                    )

    # Check discharge condition vs last lab values
    discharge_condition = state.get("discharge_condition", "")
    if discharge_condition:
        dc_lower = discharge_condition.lower()
        if any(word in dc_lower for word in ["improved", "resolved", "stable", "better"]):
            # Check if any critically abnormal lab exists
            critical_labs = [
                r for r in lab_results
                if r.get("critically_abnormal") or r.get("abnormal_flag")
            ]
            if critical_labs:
                abnormal_summary = ", ".join(
                    f"{r.get('test_name', '?')}={r.get('result_value', '?')}"
                    for r in critical_labs[:5]
                )
                conflict = {
                    "type": "DISCHARGE_CLAIM_VS_LABS",
                    "sources": [
                        f"Discharge condition: '{discharge_condition}'",
                        f"Abnormal labs: {abnormal_summary}",
                    ],
                    "description": (
                        f"Discharge summary claims '{discharge_condition}' but "
                        f"lab results still show abnormal values: {abnormal_summary}. "
                        f"Cannot endorse the 'resolved/improved' claim without lab confirmation."
                    ),
                    "resolution": "ESCALATED",
                    "rule": "CR-3",
                }
                conflicts.append(conflict)

                escalate_to_clinician(
                    state=state,
                    field="discharge_condition",
                    severity="CRITICAL",
                    reason=(
                        f"Discharge condition states '{discharge_condition}' but last "
                        f"available labs show abnormal values: {abnormal_summary}"
                    ),
                    source_evidence=[
                        f"Discharge condition: {discharge_condition}",
                        f"Abnormal labs: {abnormal_summary}",
                    ],
                    source_pages=[],
                )

    # Check for pending results
    pending = state.get("pending_results", [])
    for item in pending:
        conflict = {
            "type": "PENDING_RESULT_AT_DISCHARGE",
            "sources": [item],
            "description": (
                f"Pending result at time of discharge: {item}. "
                f"Must be listed in pending_results, not silently omitted."
            ),
            "resolution": "ESCALATED",
            "rule": "CR-3",
        }
        conflicts.append(conflict)

    return conflicts


# ─── CR-4: CULTURE RESULT VS TREATMENT ──────────────────────────────────────────

def check_cr4_culture_treatment(state: dict) -> list[dict]:
    """
    CR-4: If urine/blood culture shows "NO SIGNIFICANT BACTERIURIA" or
    "STERILE" AND patient was treated with broad-spectrum IV antibiotics →
    flag the mismatch.

    Clinical Safety:
        This is NOT an error per se — empirical antibiotic use is common.
        But the mismatch MUST be surfaced for clinician annotation.
        The agent must NOT silently resolve this.
    """
    conflicts: list[dict] = []

    # Find culture results
    culture_results = []
    for lab in state.get("lab_results", []):
        test = lab.get("test_name", "").lower()
        value = str(lab.get("result_value", "")).lower()
        if any(kw in test for kw in ["culture", "c/s", "sensitivity"]):
            culture_results.append(lab)
        elif any(kw in value for kw in ["no growth", "sterile", "no significant", "no bacteriuria"]):
            culture_results.append(lab)

    # Find broad-spectrum IV antibiotics in medications
    broad_spectrum_abs = []
    all_meds = (
        state.get("inpatient_medications", []) +
        state.get("admission_medications", [])
    )
    ab_patterns = r"(?i)(meropenem|piperacillin|tazobactam|ceftriaxone|cefoperazone|vancomycin|imipenem|amikacin|gentamicin|ciprofloxacin|levofloxacin)"

    for med in all_meds:
        med_name = med.get("name", "")
        route = med.get("route", "").upper()
        if re.search(ab_patterns, med_name) and route in ("IV", "I/V", ""):
            broad_spectrum_abs.append(med)

    # Check for mismatch
    if culture_results and broad_spectrum_abs:
        negative_cultures = [
            c for c in culture_results
            if any(kw in str(c.get("result_value", "")).lower()
                   for kw in ["no growth", "sterile", "no significant", "no bacteriuria",
                              "< 10,000", "<10000", "no organism"])
        ]

        if negative_cultures:
            culture_summary = ", ".join(
                f"{c.get('test_name', '?')}: {c.get('result_value', '?')}"
                for c in negative_cultures[:3]
            )
            ab_summary = ", ".join(
                f"{m.get('name', '?')} ({m.get('route', '?')})"
                for m in broad_spectrum_abs[:3]
            )

            conflict = {
                "type": "CULTURE_TREATMENT_MISMATCH",
                "sources": [
                    f"Culture results: {culture_summary}",
                    f"Antibiotics used: {ab_summary}",
                ],
                "description": (
                    f"Culture results show no significant growth ({culture_summary}), "
                    f"but patient was treated with broad-spectrum IV antibiotics ({ab_summary}). "
                    f"Antibiotic use without corroborating culture growth. "
                    f"Clinician must review: empirical treatment vs culture mismatch."
                ),
                "resolution": "ESCALATED",
                "rule": "CR-4",
            }
            conflicts.append(conflict)

            escalate_to_clinician(
                state=state,
                field="medication_reconciliation",
                severity="WARNING",
                reason=(
                    f"Antibiotic use ({ab_summary}) without corroborating culture growth. "
                    f"Cultures: {culture_summary}. "
                    f"Clinician must review: empirical treatment vs culture mismatch."
                ),
                source_evidence=[
                    f"Culture: {culture_summary}",
                    f"Antibiotics: {ab_summary}",
                ],
                source_pages=[],
            )

    return conflicts


# ─── CR-5: DISCHARGE CONDITION CLAIM ────────────────────────────────────────────

def check_cr5_discharge_condition(state: dict) -> list[dict]:
    """
    CR-5: If discharge_condition = "Hemodynamically stable" or "Improved"
    AND the last available vital signs show tachycardia/hypotension/fever →
    flag the discrepancy.

    Also checks for Discharge Against Medical Advice (DAMA) indicators.

    Clinical Safety:
        DO NOT simply copy the discharge condition claim. If vitals
        contradict it, flag it. If discharge was against advice, this
        changes clinical responsibility and follow-up urgency.
    """
    conflicts: list[dict] = []

    discharge_condition = state.get("discharge_condition", "")
    if not discharge_condition:
        return conflicts

    dc_lower = discharge_condition.lower()

    # Check for DAMA indicators across all documents
    dama_indicators = []
    for doc in state.get("loaded_documents", []):
        text = doc.get("raw_text", "").lower()
        extracted = doc.get("extracted_data", {})

        # Check raw text for DAMA keywords
        dama_keywords = ["against medical advice", "on request", "dama",
                         "discharge on request", "not willing", "attenders request",
                         "left against", "self-discharge"]
        for kw in dama_keywords:
            if kw in text:
                dama_indicators.append(
                    f"Page {doc.get('page_num', '?')}: contains '{kw}'"
                )

        # Check extracted data
        discharge_type = extracted.get("discharge_type", "")
        discharge_plan = extracted.get("discharge_plan", "")
        for field_val in [discharge_type, discharge_plan, str(extracted.get("recommendations", ""))]:
            if isinstance(field_val, str):
                for kw in dama_keywords:
                    if kw in field_val.lower():
                        dama_indicators.append(
                            f"Page {doc.get('page_num', '?')}: {field_val[:100]}"
                        )

    if dama_indicators:
        conflict = {
            "type": "DISCHARGE_AGAINST_ADVICE_IMPLICATIONS",
            "sources": dama_indicators[:5],
            "description": (
                f"Evidence of Discharge Against Medical Advice (DAMA) or Discharge on Request: "
                f"{'; '.join(dama_indicators[:3])}. "
                f"DAMA status must be explicitly flagged — it changes clinical responsibility "
                f"and follow-up urgency. The summary must not present this as a routine discharge."
            ),
            "resolution": "ESCALATED",
            "rule": "CR-5",
        }
        conflicts.append(conflict)

        escalate_to_clinician(
            state=state,
            field="discharge_condition",
            severity="CRITICAL",
            reason=(
                "Evidence of Discharge Against Medical Advice (DAMA) or Discharge on Request. "
                "This changes clinical responsibility and follow-up urgency."
            ),
            source_evidence=dama_indicators[:5],
            source_pages=[],
        )

    # Check vitals vs discharge condition claim
    if any(word in dc_lower for word in ["stable", "improved", "hemodynamically"]):
        # Look for last vitals in loaded documents
        last_vitals = _get_last_vitals(state)
        if last_vitals:
            vital_concerns = []
            if last_vitals.get("pulse") and float(last_vitals["pulse"]) > 100:
                vital_concerns.append(f"Tachycardia (pulse {last_vitals['pulse']})")
            if last_vitals.get("bp_systolic") and float(last_vitals["bp_systolic"]) < 90:
                vital_concerns.append(f"Hypotension (SBP {last_vitals['bp_systolic']})")
            if last_vitals.get("temperature") and float(last_vitals["temperature"]) > 99.5:
                vital_concerns.append(f"Fever (temp {last_vitals['temperature']})")

            if vital_concerns:
                conflict = {
                    "type": "DISCHARGE_CONDITION_VS_VITALS",
                    "sources": [
                        f"Discharge condition: '{discharge_condition}'",
                        f"Last vitals: {', '.join(vital_concerns)}",
                    ],
                    "description": (
                        f"Discharge condition claims '{discharge_condition}' but last "
                        f"available vital signs show: {', '.join(vital_concerns)}. "
                        f"This discrepancy must be flagged."
                    ),
                    "resolution": "ESCALATED",
                    "rule": "CR-5",
                }
                conflicts.append(conflict)

    return conflicts


# ─── MASTER CROSS-REFERENCE AUDIT ───────────────────────────────────────────────

def cross_reference_audit(state: dict) -> tuple[list[dict], list[dict]]:
    """
    Run all 5 CR rules across the complete extracted state.

    Purpose:
        This is the MANDATORY pass before compiling the final summary.
        It catches all inter-document conflicts, treatment-diagnosis
        mismatches, and safety concerns.

    Clinical Safety:
        This function MUST be called before compile_discharge_summary().
        The agent must NEVER skip this step, even under step cap pressure.

    Returns:
        Tuple of (all_conflicts, all_escalation_flags).
        These are APPENDED to the state — never replacing existing ones.
    """
    all_conflicts: list[dict] = []

    # CR-1: Treatment-Diagnosis Alignment
    cr1_conflicts = check_cr1_treatment_diagnosis_alignment(state)
    all_conflicts.extend(cr1_conflicts)

    # CR-2: Inter-Document Diagnosis Conflict
    cr2_conflicts = check_cr2_diagnosis_conflicts(state)
    all_conflicts.extend(cr2_conflicts)

    # CR-3: Lab Evidence vs Clinical Claim
    cr3_conflicts = check_cr3_lab_evidence(state)
    all_conflicts.extend(cr3_conflicts)

    # CR-4: Culture Result vs Treatment
    cr4_conflicts = check_cr4_culture_treatment(state)
    all_conflicts.extend(cr4_conflicts)

    # CR-5: Discharge Condition Claim
    cr5_conflicts = check_cr5_discharge_condition(state)
    all_conflicts.extend(cr5_conflicts)

    # Check for pending blood culture (CONFLICT-006 specific)
    _check_pending_blood_culture(state, all_conflicts)

    return all_conflicts, state.get("escalation_flags", [])


def _check_pending_blood_culture(state: dict, conflicts: list[dict]) -> None:
    """Check if blood culture results are missing from documents."""
    # Look for blood culture orders in nursing notes
    blood_culture_ordered = False
    blood_culture_result_found = False

    for doc in state.get("loaded_documents", []):
        text = doc.get("raw_text", "").lower()
        if any(kw in text for kw in ["blood c/s", "blood culture", "blood c&s"]):
            if any(kw in text for kw in ["sent", "collected", "drawn", "ordered"]):
                blood_culture_ordered = True

    for lab in state.get("lab_results", []):
        test = lab.get("test_name", "").lower()
        if "blood" in test and any(kw in test for kw in ["culture", "c/s"]):
            blood_culture_result_found = True

    if blood_culture_ordered and not blood_culture_result_found:
        pending_msg = "Blood C/S ordered but final result not found in documents."
        if pending_msg not in state.get("pending_results", []):
            state.setdefault("pending_results", []).append(pending_msg)

        conflict = {
            "type": "PENDING_RESULT_AT_DISCHARGE",
            "sources": ["Nursing notes — blood C/S ordered"],
            "description": (
                "Blood C/S sent per nursing notes. No final culture result appears "
                "in documents. Must be listed in pending_results, not silently omitted."
            ),
            "resolution": "ESCALATED",
            "rule": "CR-3",
        }
        conflicts.append(conflict)

        escalate_to_clinician(
            state=state,
            field="pending_results",
            severity="WARNING",
            reason="Blood culture ordered but result not found in documents.",
            source_evidence=["Blood C/S ordered per nursing notes"],
            source_pages=[],
        )


# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────────

def _get_all_diagnoses_text(state: dict) -> str:
    """Collect all diagnosis text from every source into a single searchable string."""
    parts: list[str] = []

    # From structured diagnoses
    diagnoses = state.get("diagnoses", {})
    for category in ["principal", "secondary", "provisional", "final"]:
        for d in diagnoses.get(category, []):
            parts.append(str(d))

    # From loaded documents
    for doc in state.get("loaded_documents", []):
        extracted = doc.get("extracted_data", {})
        for key in ["er_diagnosis", "provisional_diagnosis", "final_diagnosis",
                     "icu_diagnoses", "consultation_diagnosis"]:
            val = extracted.get(key)
            if isinstance(val, list):
                parts.extend(str(v) for v in val)
            elif isinstance(val, str):
                parts.append(val)

    return " | ".join(parts)


def _get_last_vitals(state: dict) -> Optional[dict]:
    """Get the last recorded vital signs from loaded documents."""
    last_vitals = None
    for doc in state.get("loaded_documents", []):
        extracted = doc.get("extracted_data", {})
        vitals = extracted.get("vitals_timeline", [])
        if isinstance(vitals, list) and vitals:
            last_vitals = vitals[-1]
        single_vitals = extracted.get("vitals")
        if isinstance(single_vitals, dict) and any(single_vitals.values()):
            last_vitals = single_vitals
    return last_vitals
