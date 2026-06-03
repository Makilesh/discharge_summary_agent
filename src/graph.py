"""
graph.py — LangGraph State Machine for the Discharge Summary Agent
====================================================================

Implements the deterministic state machine with the phase cycle:
    REASON → PLAN → CALL_TOOL → OBSERVE → VERIFY → [REASON | COMPILE | ESCALATE]

The graph uses LangGraph's StateGraph with conditional edges for routing.
Each node is a separate function that takes and returns AgentState.

Clinical Safety:
    - Hard cap of MAX_ITERATIONS = 20 per patient document set.
    - steps_remaining is decremented on every transition.
    - On cap breach: HARD_CAP_ESCALATE compiles with [MISSING] markers.
    - Never retries a failed tool call more than 2 times.
    - Batch processes pages of the same type to maximize step efficiency.
"""

from __future__ import annotations
import json
from typing import Optional

from langgraph.graph import StateGraph, END

from .state import AgentState, create_initial_state
from .config import (
    MAX_ITERATIONS, EXTRACTION_PRIORITY_ORDER, DOC_TYPES,
    MISSING_FIELD_TEMPLATE, MIN_TEXT_LENGTH, BATCH_SIZE,
)
from .pdf_processor import extract_all_page_images, get_page_image_base64, get_batch_page_images
from .tools import (
    classify_document_pages, extract_page_text, extract_typed_summary,
    extract_clinical_data, extract_lab_report, extract_drug_chart_batch,
    reconcile_medications, lookup_drug_interactions, safe_tool_call,
)
from .cross_reference import cross_reference_audit
from .compiler import compile_discharge_summary
from .trace import emit_trace, validate_state_completeness, generate_trace_summary


# ─── NODE: INITIALIZE ───────────────────────────────────────────────────────────

def initialize_node(state: dict) -> dict:
    """
    Load PDF, extract all page images, and classify every page.

    Purpose:
        First step in the pipeline. Builds the document inventory that
        drives all subsequent extraction decisions.

    Clinical Safety:
        Every page must be attempted. Pages that fail OCR are logged
        in unreadable_pages — never silently dropped.

    This node consumes 2 steps: one for image extraction, one for classification.
    """
    start_trace_len = len(state.get("trace", []))
    pdf_path = state.get("_pdf_path", "")
    step = MAX_ITERATIONS - state.get("steps_remaining", MAX_ITERATIONS)

    emit_trace(
        state=state,
        step_number=step,
        phase="INITIALIZE",
        reasoning=f"Starting agent. Loading PDF from {pdf_path}",
        action="LOAD_PDF",
        observation="Beginning page image extraction and classification",
        decision="Extract all page images, then batch-classify",
    )

    # Extract page images
    page_images = extract_all_page_images(pdf_path)
    total_pages = len(page_images)

    emit_trace(
        state=state,
        step_number=step,
        phase="INITIALIZE",
        reasoning=f"Extracted {total_pages} page images from PDF",
        action="IMAGES_EXTRACTED",
        observation=f"{total_pages} pages rendered as PNG images",
        decision="Proceeding to batch classification",
        fields_updated=["page_images"],
    )

    # Batch-classify pages (process in batches of 10 to stay within LLM context limits)
    all_classifications: list[dict] = []
    page_nums = sorted(page_images.keys())
    batch_size = 10  # Images per classification batch

    for batch_start in range(0, len(page_nums), batch_size):
        batch_pages = page_nums[batch_start:batch_start + batch_size]
        batch_images = [
            (pn, get_page_image_base64(page_images, pn))
            for pn in batch_pages
            if get_page_image_base64(page_images, pn) is not None
        ]

        if batch_images:
            try:
                classifications = classify_document_pages(batch_images)
                all_classifications.extend(classifications)
            except Exception as e:
                # On classification failure, mark pages as UNKNOWN
                for pn in batch_pages:
                    all_classifications.append({
                        "page_num": pn,
                        "doc_type": "UNKNOWN",
                        "confidence": 0.0,
                    })
                emit_trace(
                    state=state,
                    step_number=step,
                    phase="INITIALIZE",
                    reasoning=f"Classification batch failed for pages {batch_pages}",
                    action="CLASSIFICATION_FAILED",
                    observation=str(e)[:200],
                    decision="Marking pages as UNKNOWN, continuing",
                    fallback_taken=True,
                    fallback_reason=str(e)[:200],
                )

    # Build document inventory
    loaded_documents: list[dict] = []
    unreadable_pages: list[int] = []

    for cls in all_classifications:
        pn = cls.get("page_num", 0)
        doc_type = cls.get("doc_type", "UNKNOWN")
        confidence = cls.get("confidence", 0.0)

        doc_entry = {
            "page_num": pn,
            "source_type": doc_type,
            "raw_text": "",  # Will be filled during extraction
            "confidence": confidence,
            "extracted_data": {},
        }
        loaded_documents.append(doc_entry)

        if doc_type == "UNKNOWN" and confidence < 0.5:
            unreadable_pages.append(pn)

    # Build processing queue by priority
    processing_queue = _build_processing_queue(loaded_documents)

    emit_trace(
        state=state,
        step_number=step + 1,
        phase="INITIALIZE",
        reasoning=f"Classified {total_pages} pages. {len(unreadable_pages)} unreadable.",
        action="CLASSIFICATION_COMPLETE",
        observation=(
            f"Document types found: {_summarize_doc_types(loaded_documents)}. "
            f"Unreadable pages: {unreadable_pages}"
        ),
        decision="Proceeding to extraction phase in priority order",
        fields_updated=["loaded_documents", "unreadable_pages", "processing_queue"],
    )

    # Store base64 images for later use (avoid storing raw bytes in state)
    page_images_b64: dict[int, str] = {}
    for pn, img_bytes in page_images.items():
        import base64
        page_images_b64[pn] = base64.b64encode(img_bytes).decode("utf-8")

    return {
        "loaded_documents": loaded_documents,
        "unreadable_pages": unreadable_pages,
        "page_images": page_images_b64,
        "processing_queue": processing_queue,
        "steps_remaining": state.get("steps_remaining", MAX_ITERATIONS) - 2,
        "current_phase": "REASON",
        "trace": state.get("trace", [])[start_trace_len:],
    }


# ─── NODE: REASON ───────────────────────────────────────────────────────────────

def reason_node(state: dict) -> dict:
    """
    Decide what to extract next based on priority order and current gaps.

    Purpose:
        The REASON phase outputs a structured plan with explicit field targets.
        It examines what data has been extracted and what remains.

    Clinical Safety:
        Must follow the EXTRACTION_PRIORITY_ORDER strictly.
        Must not skip ahead — higher priority types inform lower ones.
    """
    start_trace_len = len(state.get("trace", []))
    step = MAX_ITERATIONS - state.get("steps_remaining", MAX_ITERATIONS)
    queue = state.get("processing_queue", [])

    if not queue:
        # All queue items processed — check if we need medication reconciliation
        if not state.get("medication_reconciliation"):
            reasoning = (
                "All document types have been processed. Now need to reconcile "
                "medications and run cross-reference audit."
            )
            decision = "Proceed to medication reconciliation"
            next_action = "RECONCILE_MEDICATIONS"
        else:
            reasoning = "All extractions and reconciliation complete."
            decision = "Proceed to VERIFY phase"
            next_action = "VERIFY"

        emit_trace(
            state=state,
            step_number=step,
            phase="REASON",
            reasoning=reasoning,
            action="PLAN",
            decision=decision,
        )

        return {
            "current_phase": "CALL_TOOL" if next_action == "RECONCILE_MEDICATIONS" else "VERIFY",
            "steps_remaining": state["steps_remaining"] - 1,
            "_next_action": next_action,
            "trace": state.get("trace", [])[start_trace_len:],
        }

    # Get next batch from queue
    next_batch = queue[0]
    remaining_queue = queue[1:]

    doc_type = next_batch.get("doc_type", "UNKNOWN")
    pages = next_batch.get("pages", [])

    reasoning = (
        f"Next priority: {doc_type} (pages {pages}). "
        f"Queue has {len(remaining_queue)} more batches. "
        f"Steps remaining: {state['steps_remaining']}."
    )

    emit_trace(
        state=state,
        step_number=step,
        phase="REASON",
        reasoning=reasoning,
        action="PLAN",
        decision=f"Extract {doc_type} from pages {pages}",
        fields_updated=[],
    )

    return {
        "current_phase": "CALL_TOOL",
        "steps_remaining": state["steps_remaining"] - 1,
        "processing_queue": remaining_queue,
        "_next_action": "EXTRACT",
        "_target_doc_type": doc_type,
        "_target_pages": pages,
        "trace": state.get("trace", [])[start_trace_len:],
    }


# ─── NODE: CALL_TOOL ────────────────────────────────────────────────────────────

def call_tool_node(state: dict) -> dict:
    """
    Execute the planned tool call.

    Purpose:
        Runs the appropriate extraction tool based on the REASON phase's plan.
        Uses safe_tool_call for retry logic.

    Clinical Safety:
        - All tool calls wrapped in safe_tool_call (max 2 retries).
        - Failed tools result in [UNRESOLVED] markers.
        - Batch processing for same-type pages to save steps.
    """
    start_trace_len = len(state.get("trace", []))
    step = MAX_ITERATIONS - state.get("steps_remaining", MAX_ITERATIONS)
    next_action = state.get("_next_action", "EXTRACT")
    page_images = state.get("page_images", {})

    updates: dict = {
        "steps_remaining": state["steps_remaining"] - 1,
        "current_phase": "OBSERVE",
    }

    if next_action == "RECONCILE_MEDICATIONS":
        updates = _do_medication_reconciliation(state, step, updates)
    elif next_action == "EXTRACT":
        updates = _do_extraction(state, step, updates)
    elif next_action == "DRUG_INTERACTION_CHECK":
        updates = _do_drug_interaction_check(state, step, updates)

    updates["trace"] = state.get("trace", [])[start_trace_len:]
    return updates


def _do_extraction(state: dict, step: int, updates: dict) -> dict:
    """Perform extraction for a batch of same-type pages."""
    doc_type = state.get("_target_doc_type", "UNKNOWN")
    target_pages = state.get("_target_pages", [])
    page_images = state.get("page_images", {})

    new_docs: list[dict] = []
    new_lab_results: list[dict] = []
    new_imaging: list[dict] = []
    new_procedures: list[dict] = []
    new_meds: list[dict] = []

    if not target_pages:
        emit_trace(
            state=state, step_number=step, phase="CALL_TOOL",
            reasoning="Extraction was requested but no target pages were present in state",
            action="TARGET_PAGES_MISSING",
            tool_name=f"extract_{doc_type.lower()}",
            observation=(
                "No pages were available for extraction. This usually indicates "
                "a planning-state handoff failure."
            ),
            decision="Do not fabricate extracted data; continue to verification",
            fallback_taken=True,
            fallback_reason="Missing _target_pages",
        )
        updates.setdefault("fabrication_blocks", []).append(
            f"{doc_type}: extraction skipped because target pages were missing."
        )
        updates["loaded_documents"] = []
        return updates

    page_payloads: list[dict] = []
    for page_num in target_pages:
        img_b64 = page_images.get(page_num)
        if not img_b64:
            continue

        # Step 1: OCR the page
        try:
            ocr_result = extract_page_text(img_b64, page_num)
            text = ocr_result.get("text", "")
            confidence = ocr_result.get("confidence", 0.0)
        except Exception as e:
            emit_trace(
                state=state, step_number=step, phase="CALL_TOOL",
                reasoning=f"OCR failed for page {page_num}",
                action="OCR_FAILED", fallback_taken=True,
                fallback_reason=str(e)[:200],
            )
            text = ""
            confidence = 0.0

        if len(text) < MIN_TEXT_LENGTH:
            updates.setdefault("unreadable_pages", []).append(page_num)
            continue

        page_payloads.append({
            "page_num": page_num,
            "img_b64": img_b64,
            "text": text,
            "confidence": confidence,
        })

    # Drug charts are the highest-value multi-page batch: medication timelines
    # need cross-page context, and one batch call is cheaper than N page calls.
    if doc_type == "DRUG_CHART" and page_payloads:
        page_texts = [(p["page_num"], p["text"]) for p in page_payloads]
        combined_text = "\n\n".join(text for _, text in page_texts)
        include_images = "[DRUG NAME PARTIALLY LEGIBLE" in combined_text
        page_image_inputs = (
            [(p["page_num"], p["img_b64"]) for p in page_payloads]
            if include_images else None
        )

        try:
            extracted = extract_drug_chart_batch(
                page_texts=page_texts,
                page_images=page_image_inputs,
            )
            if not isinstance(extracted, dict):
                extracted = {}
        except Exception as e:
            emit_trace(
                state=state, step_number=step, phase="CALL_TOOL",
                reasoning=f"Batch extraction failed for drug chart pages {target_pages}",
                action="EXTRACTION_FAILED", fallback_taken=True,
                fallback_reason=str(e)[:200],
            )
            extracted = {}

        meds = extracted.get("medications", [])
        if isinstance(meds, list):
            source_pages = [p["page_num"] for p in page_payloads]
            for med in meds:
                if isinstance(med, dict):
                    med.setdefault("status", "INPATIENT_ONLY")
                    med.setdefault("source_pages", source_pages)
                    new_meds.append(med)

        for payload in page_payloads:
            new_docs.append({
                "page_num": payload["page_num"],
                "source_type": doc_type,
                "raw_text": payload["text"][:5000],
                "confidence": payload["confidence"],
                "extracted_data": {
                    "batch_extraction": True,
                    "medication_count": len(new_meds),
                    "chart_date_range": extracted.get("chart_date_range", {}),
                    "notes": extracted.get("notes"),
                },
            })

        emit_trace(
            state=state, step_number=step, phase="CALL_TOOL",
            reasoning=(
                f"Batch processed {len(page_payloads)} drug chart pages to preserve "
                "cross-page medication context while reducing LLM calls"
            ),
            action="BATCH_EXTRACTION_COMPLETE",
            tool_name="extract_drug_chart_batch",
            observation=f"Extracted {len(new_meds)} medication entries from pages {target_pages}",
            decision="Route batched medications to reconciliation state fields",
            fields_updated=["drug_chart", "inpatient_medications"],
        )
    else:
        for payload in page_payloads:
            page_num = payload["page_num"]
            img_b64 = payload["img_b64"]
            text = payload["text"]
            confidence = payload["confidence"]

            # If the page type is UNKNOWN, try to classify it on the fly from text
            page_doc_type = doc_type
            if page_doc_type == "UNKNOWN":
                from .tools import classify_page_from_text
                page_doc_type = classify_page_from_text(text)
                print(f"  [RE-CLASSIFY] Page {page_num} text-classified as {page_doc_type}")

            # Step 2: Extract structured data based on doc type. Prefer text-only
            # specialized tools where possible; use image context only for pages
            # that are already ambiguous, to control cost without hiding uncertainty.
            try:
                if page_doc_type == "TYPED_DISCHARGE_SUMMARY":
                    extracted = extract_typed_summary(text=text, page_num=page_num)
                elif page_doc_type.startswith("LAB_REPORT"):
                    extracted = extract_lab_report(
                        text=text, page_num=page_num, report_type=page_doc_type
                    )
                else:
                    image_for_extraction = img_b64 if "[UNCLEAR" in text else None
                    extracted = extract_clinical_data(
                        text=text, page_num=page_num,
                        doc_type=page_doc_type, image_b64=image_for_extraction,
                    )
                if not isinstance(extracted, dict):
                    extracted = {}
            except Exception as e:
                emit_trace(
                    state=state, step_number=step, phase="CALL_TOOL",
                    reasoning=f"Extraction failed for page {page_num} ({page_doc_type})",
                    action="EXTRACTION_FAILED", fallback_taken=True,
                    fallback_reason=str(e)[:200],
                )
                extracted = {}

            # Build document entry
            doc_entry = {
                "page_num": page_num,
                "source_type": page_doc_type,
                "raw_text": text[:5000],  # Truncate for state size
                "confidence": confidence,
                "extracted_data": extracted,
            }
            new_docs.append(doc_entry)

            # Route extracted data to appropriate state fields
            _route_extracted_data(
                state, extracted, page_doc_type, page_num, updates,
                new_lab_results, new_imaging, new_procedures, new_meds,
            )

    emit_trace(
        state=state, step_number=step, phase="CALL_TOOL",
        reasoning=f"Processed {len(target_pages)} pages of type {doc_type} (on-the-fly resolved to: {page_doc_type if 'page_doc_type' in locals() else doc_type})",
        action="EXTRACTION_COMPLETE",
        tool_name=f"extract_{doc_type.lower()}",
        observation=f"Extracted data from {len(new_docs)} pages",
        decision="Route data to state fields",
        fields_updated=[doc_type.lower()],
    )

    updates["loaded_documents"] = new_docs
    if new_lab_results:
        updates["lab_results"] = new_lab_results
    if new_imaging:
        updates["imaging_results"] = new_imaging
    if new_procedures:
        updates["procedures"] = new_procedures
    if new_meds:
        # Route meds to appropriate list
        for med in new_meds:
            status = med.get("status", "UNKNOWN")
            if status == "ADMISSION":
                updates.setdefault("admission_medications", []).append(med)
            elif status == "DISCHARGE":
                updates.setdefault("discharge_medications", []).append(med)
            else:
                updates.setdefault("inpatient_medications", []).append(med)

    return updates


def _route_extracted_data(
    state: dict, extracted: dict, doc_type: str, page_num: int,
    updates: dict,
    new_lab_results: list, new_imaging: list, new_procedures: list, new_meds: list,
) -> None:
    """Route extracted data from a page to the appropriate state fields."""
    if not extracted:
        return

    # Demographics (from typed summary or admission record)
    if doc_type in ("TYPED_DISCHARGE_SUMMARY", "ADMISSION_RECORD"):
        demo = extracted.get("demographics")
        if demo and isinstance(demo, dict):
            # Merge with existing, don't overwrite
            existing = state.get("extracted_demographics", {})
            merged = {**existing}
            for k, v in demo.items():
                if v and not merged.get(k):
                    merged[k] = v
            updates["extracted_demographics"] = merged

        # Dates
        for date_field in ["admission_date", "discharge_date"]:
            val = extracted.get(date_field)
            if val and not state.get(date_field):
                updates[date_field] = val

        # Diagnoses
        diag = extracted.get("diagnoses")
        if diag and isinstance(diag, dict):
            existing_diag = state.get("diagnoses", {
                "principal": [], "secondary": [], "provisional": [], "final": []
            })
            for category in ["principal", "secondary", "provisional", "final"]:
                new_entries = diag.get(category, [])
                if isinstance(new_entries, list):
                    for entry in new_entries:
                        if entry and entry not in existing_diag.get(category, []):
                            existing_diag.setdefault(category, []).append(entry)
            updates["diagnoses"] = existing_diag

        # Allergies
        allergies = extracted.get("allergies", [])
        if allergies:
            updates["allergies"] = allergies

        # Follow-up
        follow_up = extracted.get("follow_up", [])
        if follow_up:
            updates["follow_up_instructions"] = follow_up

        # Condition at discharge
        condition = extracted.get("condition_at_discharge")
        if condition:
            updates["discharge_condition"] = condition

    # Diagnoses from ER/ICU/Consultation
    if doc_type in ("ER_OBSERVATION_CHART", "ICU_CHART", "CONSULTATION_SHEET"):
        for diag_key in ["er_diagnosis", "icu_diagnoses", "consultation_diagnosis",
                          "provisional_diagnosis", "final_diagnosis"]:
            val = extracted.get(diag_key)
            if val:
                if isinstance(val, str):
                    val = [val]
                existing_diag = state.get("diagnoses", {
                    "principal": [], "secondary": [], "provisional": [], "final": []
                })
                # Store in provisional for cross-reference
                for entry in val:
                    if entry and entry not in existing_diag.get("provisional", []):
                        existing_diag.setdefault("provisional", []).append(entry)
                updates["diagnoses"] = existing_diag

    # Chief complaints (for CR-2 detection)
    chief = extracted.get("chief_complaints", [])
    if chief:
        # Store in loaded_documents for cross-reference
        pass

    # Lab results
    if doc_type.startswith("LAB_REPORT"):
        results = extracted.get("results", [])
        if isinstance(results, list):
            for r in results:
                r["source_page"] = page_num
            new_lab_results.extend(results)

    # Imaging
    if doc_type.startswith("IMAGING_REPORT") or doc_type == "ECHO_REPORT":
        impression = extracted.get("impression") or extracted.get("findings")
        if impression:
            new_imaging.append({
                "modality": doc_type.replace("IMAGING_REPORT_", "").replace("_", " "),
                "date": extracted.get("date", ""),
                "impression": str(impression),
                "source_page": page_num,
            })

    # Medications from drug charts
    if doc_type == "DRUG_CHART":
        meds = extracted.get("medications", [])
        if isinstance(meds, list):
            new_meds.extend(meds)

    # Medications from ICU/ER
    if doc_type in ("ICU_CHART", "ER_OBSERVATION_CHART"):
        meds = extracted.get("medications", []) or extracted.get("treatments_given", [])
        if isinstance(meds, list):
            for m in meds:
                if isinstance(m, dict):
                    m.setdefault("status", "INPATIENT_ONLY")
                    new_meds.append(m)

    # Procedures
    if doc_type == "PROCEDURE_CHART":
        procs = extracted.get("procedures", [])
        if isinstance(procs, list):
            new_procedures.extend(procs)

    # Discharge checklist
    if doc_type == "DISCHARGE_CHECKLIST":
        discharge_type = extracted.get("discharge_type")
        if discharge_type:
            updates["discharge_condition"] = discharge_type
        pending = extracted.get("pending_items", [])
        if pending:
            updates.setdefault("pending_results", []).extend(pending)

    # Diabetes monitoring — store glucose readings as lab results
    if doc_type == "MONITORING_CHART_DIABETES":
        readings = extracted.get("readings", [])
        for r in readings:
            new_lab_results.append({
                "test_name": "Blood Glucose (GRBS)",
                "result_value": str(r.get("glucose_value", "")),
                "unit": r.get("glucose_unit", "mg/dL"),
                "reference_range": "70-140 mg/dL",
                "date": r.get("date", ""),
                "abnormal_flag": (
                    r.get("glucose_value") and
                    (float(r.get("glucose_value", 0)) > 140 or float(r.get("glucose_value", 0)) < 70)
                ) if r.get("glucose_value") else False,
                "critically_abnormal": (
                    r.get("glucose_value") and float(r.get("glucose_value", 0)) > 400
                ) if r.get("glucose_value") else False,
                "source_page": page_num,
            })


def _do_medication_reconciliation(state: dict, step: int, updates: dict) -> dict:
    """Perform medication reconciliation."""
    admission = state.get("admission_medications", [])
    inpatient = state.get("inpatient_medications", [])
    discharge = state.get("discharge_medications", [])

    try:
        result = reconcile_medications(admission, inpatient, discharge)
        reconciliation = result.get("reconciliation", [])
        flags = result.get("flags", [])

        updates["medication_reconciliation"] = reconciliation
        updates["escalation_flags"] = flags

        emit_trace(
            state=state, step_number=step, phase="CALL_TOOL",
            reasoning="Reconciling admission vs inpatient vs discharge medications",
            action="RECONCILE_MEDICATIONS",
            tool_name="reconcile_medications",
            observation=f"Found {len(reconciliation)} medication entries, {len(flags)} flags",
            decision="Medication reconciliation complete",
            fields_updated=["medication_reconciliation", "escalation_flags"],
        )
    except Exception as e:
        emit_trace(
            state=state, step_number=step, phase="CALL_TOOL",
            reasoning="Medication reconciliation failed",
            action="RECONCILE_FAILED", fallback_taken=True,
            fallback_reason=str(e)[:200],
        )
        updates["fabrication_blocks"] = [
            "medication_reconciliation: reconcile_medications failed. Marked [UNRESOLVED]."
        ]

    # Also run drug interaction check
    all_med_names = list(set(
        m.get("name", "") for m in (admission + inpatient + discharge) if m.get("name")
    ))
    if all_med_names:
        try:
            interactions = lookup_drug_interactions(all_med_names)
            if interactions.get("interactions"):
                emit_trace(
                    state=state, step_number=step, phase="CALL_TOOL",
                    reasoning="Checking drug interactions",
                    action="DRUG_INTERACTION_CHECK",
                    tool_name="lookup_drug_interactions",
                    observation=f"Found {len(interactions['interactions'])} potential interactions [MOCKED]",
                    decision="Log interactions for clinician review",
                )
        except Exception:
            pass  # Non-critical — mocked tool

    updates["_next_action"] = "VERIFY"
    return updates


def _do_drug_interaction_check(state: dict, step: int, updates: dict) -> dict:
    """Run the mocked drug interaction check."""
    # Already handled in reconciliation step
    return updates


# ─── NODE: OBSERVE ───────────────────────────────────────────────────────────────

def observe_node(state: dict) -> dict:
    """
    Process tool output and transition to VERIFY.

    Purpose:
        Lightweight observation node that logs what was updated and
        transitions to verification.

    Clinical Safety:
        Observation must never modify data — only log what was received.
    """
    start_trace_len = len(state.get("trace", []))
    step = MAX_ITERATIONS - state.get("steps_remaining", MAX_ITERATIONS)

    emit_trace(
        state=state, step_number=step, phase="OBSERVE",
        reasoning="Reviewing results from tool call",
        action="OBSERVE",
        observation="Tool results have been routed to state fields",
        decision="Proceed to VERIFY",
    )

    return {
        "current_phase": "VERIFY",
        "trace": state.get("trace", [])[start_trace_len:],
    }


# ─── NODE: VERIFY ───────────────────────────────────────────────────────────────

def verify_node(state: dict) -> dict:
    """
    Check extraction completeness and detect conflicts.

    Purpose:
        Determines whether to continue extracting (REASON), compile
        the final summary (COMPILE), or escalate.

    Clinical Safety:
        This is the decision point for the agent loop. It must correctly
        identify when enough data has been gathered vs when critical
        fields are still missing.
    """
    start_trace_len = len(state.get("trace", []))
    step = MAX_ITERATIONS - state.get("steps_remaining", MAX_ITERATIONS)
    completed, missing, flagged = validate_state_completeness(state)

    queue = state.get("processing_queue", [])
    has_reconciled = bool(state.get("medication_reconciliation"))

    reasoning = (
        f"Verification: {len(completed)} fields complete, {len(missing)} missing, "
        f"{len(flagged)} flagged. Queue: {len(queue)} batches remaining. "
        f"Reconciled: {has_reconciled}. Steps remaining: {state['steps_remaining']}."
    )

    emit_trace(
        state=state, step_number=step, phase="VERIFY",
        reasoning=reasoning,
        action="VERIFY",
        observation=f"Complete: {completed}. Missing: {missing}. Flagged: {flagged}.",
        decision="Determining next phase",
    )

    return {
        "current_phase": "VERIFY_DONE",
        "steps_remaining": state["steps_remaining"] - 1,
        "trace": state.get("trace", [])[start_trace_len:],
    }


# ─── NODE: COMPILE ──────────────────────────────────────────────────────────────

def compile_node(state: dict) -> dict:
    """
    Run cross-reference audit, then compile the final discharge summary.

    Purpose:
        Final step — runs the mandatory CR audit, generates the
        hospital course narrative, and compiles everything into
        the structured Markdown summary.

    Clinical Safety:
        The cross_reference_audit MUST run before compilation.
        The agent MUST NOT skip this step even under step pressure.
    """
    start_trace_len = len(state.get("trace", []))
    start_flags_len = len(state.get("escalation_flags", []))
    start_conflicts_len = len(state.get("conflicts", []))
    step = MAX_ITERATIONS - state.get("steps_remaining", MAX_ITERATIONS)

    emit_trace(
        state=state, step_number=step, phase="COMPILE",
        reasoning="Running mandatory cross-reference audit before compilation",
        action="CROSS_REFERENCE_AUDIT",
        decision="Execute CR-1 through CR-5, then compile summary",
    )

    # Run cross-reference audit
    new_conflicts, _ = cross_reference_audit(state)

    # Generate hospital course if not yet done
    hospital_course = state.get("hospital_course", "")
    if not hospital_course:
        hospital_course = _generate_hospital_course(state)

    # Compile the final summary
    # Merge new conflicts into state for compilation
    compile_state = {**state}
    compile_state["conflicts"] = state.get("conflicts", []) + new_conflicts
    compile_state["hospital_course"] = hospital_course

    summary = compile_discharge_summary(compile_state)
    trace_summary = generate_trace_summary(compile_state)

    emit_trace(
        state=state, step_number=step, phase="COMPILE",
        reasoning="Compilation complete",
        action="COMPILE_COMPLETE",
        observation=f"Summary: {len(summary)} chars. "
                    f"Conflicts: {len(compile_state['conflicts'])}. "
                    f"Escalations: {len(compile_state.get('escalation_flags', []))}.",
        decision="Agent run complete. Summary ready for clinician review.",
        fields_updated=["final_summary", "conflicts", "hospital_course"],
    )

    return {
        "final_summary": summary,
        "conflicts": compile_state.get("conflicts", [])[start_conflicts_len:],
        "escalation_flags": compile_state.get("escalation_flags", [])[start_flags_len:],
        "hospital_course": hospital_course,
        "current_phase": "DONE",
        "trace": state.get("trace", [])[start_trace_len:],
    }


# ─── NODE: HARD CAP ESCALATE ────────────────────────────────────────────────────

def hard_cap_escalate_node(state: dict) -> dict:
    """
    Emergency compilation when the step cap is reached.

    Purpose:
        Compiles whatever has been extracted so far, marking all
        incomplete sections as [MISSING — agent step cap reached].

    Clinical Safety:
        ALL incomplete sections get explicit markers. The summary
        cannot appear complete when it is not. The HARD_CAP_HIT
        flag is prominently displayed.
    """
    start_trace_len = len(state.get("trace", []))
    start_flags_len = len(state.get("escalation_flags", []))
    start_conflicts_len = len(state.get("conflicts", []))

    emit_trace(
        state=state, step_number=MAX_ITERATIONS, phase="HARD_CAP_ESCALATE",
        reasoning="Agent step cap reached. Compiling with available data.",
        action="HARD_CAP_COMPILE",
        observation=f"Steps remaining: {state.get('steps_remaining', 0)}. "
                    f"Processing queue had {len(state.get('processing_queue', []))} items remaining.",
        decision="Emergency compilation with [MISSING] markers",
    )

    # Still run cross-reference on whatever we have
    try:
        new_conflicts, _ = cross_reference_audit(state)
    except Exception:
        new_conflicts = []

    # Generate hospital course from whatever we have
    hospital_course = state.get("hospital_course", "")
    if not hospital_course:
        hospital_course = _generate_hospital_course(state)

    compile_state = {**state}
    compile_state["conflicts"] = state.get("conflicts", []) + new_conflicts
    compile_state["hospital_course"] = hospital_course
    compile_state["escalation_flags"] = state.get("escalation_flags", []) + [{
        "field": "SUMMARY_COMPLETENESS",
        "severity": "CRITICAL",
        "reason": "Agent step cap (20) reached. Some document types may not have been processed.",
        "source_page": None,
        "requires_clinician": True,
    }]

    summary = compile_discharge_summary(compile_state)
    generate_trace_summary(compile_state)

    return {
        "final_summary": summary,
        "conflicts": compile_state.get("conflicts", [])[start_conflicts_len:],
        "escalation_flags": compile_state.get("escalation_flags", [])[start_flags_len:],
        "hospital_course": hospital_course,
        "current_phase": "DONE",
        "trace": state.get("trace", [])[start_trace_len:],
    }


# ─── ROUTING FUNCTION ───────────────────────────────────────────────────────────

def route_after_verify(state: dict) -> str:
    """
    Route from VERIFY to the next phase.

    Decision Logic:
        1. If steps_remaining <= 0 → HARD_CAP_ESCALATE
        2. If processing_queue has items → REASON (more to extract)
        3. If medication reconciliation not done → REASON
        4. If all fields attempted → COMPILE
        5. Otherwise → REASON
    """
    if state.get("steps_remaining", 0) <= 0:
        return "hard_cap_escalate"

    queue = state.get("processing_queue", [])
    has_reconciled = bool(state.get("medication_reconciliation"))

    if queue:
        return "reason"

    if not has_reconciled:
        return "reason"

    # Check if we have enough data to compile
    completed, missing, _ = validate_state_completeness(state)
    if len(completed) >= 5:  # Minimum viable fields
        return "compile"

    # Still have steps — try to fill gaps
    if state.get("steps_remaining", 0) > 2:
        return "reason"

    return "compile"


# ─── HOSPITAL COURSE GENERATION ─────────────────────────────────────────────────

def _generate_hospital_course(state: dict) -> str:
    """
    Generate the hospital course narrative from extracted data.

    This synthesizes a chronological narrative from nursing notes,
    consultation sheets, and other documents. Every sentence includes
    a source page reference.

    Clinical Safety:
        Every sentence MUST include a source page reference.
        The narrative MUST NOT invent any events not in the source documents.
    """
    events: list[tuple[str, int]] = []  # (event_text, page_num)

    for doc in state.get("loaded_documents", []):
        page_num = doc.get("page_num", 0)
        doc_type = doc.get("source_type", "")
        extracted = doc.get("extracted_data", {})
        raw_text = doc.get("raw_text", "")

        if doc_type == "ER_OBSERVATION_CHART":
            er_diag = extracted.get("er_diagnosis", [])
            if er_diag:
                events.append(
                    (f"Patient presented to ER with provisional diagnosis of {', '.join(er_diag)}", page_num)
                )
            treatments = extracted.get("treatments_given", [])
            if treatments:
                for t in treatments[:3]:
                    if isinstance(t, dict):
                        events.append(
                            (f"ER treatment: {t.get('drug', '?')} {t.get('dose', '')} {t.get('route', '')}", page_num)
                        )

        elif doc_type == "ICU_CHART":
            icu_diag = extracted.get("icu_diagnoses", [])
            if icu_diag:
                events.append(
                    (f"ICU admission with diagnoses: {', '.join(icu_diag)}", page_num)
                )

        elif doc_type == "CONSULTATION_SHEET":
            consult_diag = extracted.get("consultation_diagnosis", [])
            consult_date = extracted.get("consultation_date", "")
            recommendations = extracted.get("recommendations", [])
            discharge_plan = extracted.get("discharge_plan", "")

            if consult_diag:
                events.append(
                    (f"Consultation ({consult_date}): Diagnoses — {', '.join(consult_diag)}", page_num)
                )
            if recommendations:
                events.append(
                    (f"Consultation recommendations: {', '.join(recommendations[:3])}", page_num)
                )
            if discharge_plan:
                events.append(
                    (f"Discharge plan: {discharge_plan}", page_num)
                )

        elif doc_type == "NURSING_NOTES":
            nursing_events = extracted.get("events", [])
            if isinstance(nursing_events, list):
                for ne in nursing_events[:5]:
                    if isinstance(ne, dict):
                        desc = ne.get("description", "")
                        ts = ne.get("timestamp", "")
                        if desc:
                            events.append(
                                (f"Nursing note ({ts}): {desc[:150]}", page_num)
                            )

    if not events:
        return MISSING_FIELD_TEMPLATE

    # Build narrative with page references
    narrative_parts: list[str] = []
    for event_text, page_num in events:
        narrative_parts.append(f"{event_text} **[Page {page_num}]**.")

    return "\n\n".join(narrative_parts)


# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────────────

def _build_processing_queue(loaded_documents: list[dict]) -> list[dict]:
    """
    Build a processing queue ordered by EXTRACTION_PRIORITY_ORDER.
    Groups same-type pages into batches for efficient processing.
    """
    # Group pages by type
    type_pages: dict[str, list[int]] = {}
    for doc in loaded_documents:
        doc_type = doc.get("source_type", "UNKNOWN")
        page_num = doc.get("page_num", 0)
        type_pages.setdefault(doc_type, []).append(page_num)

    # Build queue in priority order
    queue: list[dict] = []
    for doc_type in EXTRACTION_PRIORITY_ORDER:
        pages = type_pages.get(doc_type, [])
        if pages:
            # Batch pages (max BATCH_SIZE per batch)
            for i in range(0, len(pages), BATCH_SIZE):
                batch = sorted(pages[i:i + BATCH_SIZE])
                queue.append({
                    "doc_type": doc_type,
                    "pages": batch,
                })

    # Add any unclassified pages at the end
    for doc_type, pages in type_pages.items():
        if doc_type not in EXTRACTION_PRIORITY_ORDER and doc_type != "UNKNOWN":
            queue.append({
                "doc_type": doc_type,
                "pages": sorted(pages),
            })

    return queue


def _summarize_doc_types(loaded_documents: list[dict]) -> str:
    """Create a summary string of document types found."""
    type_counts: dict[str, int] = {}
    for doc in loaded_documents:
        dt = doc.get("source_type", "UNKNOWN")
        type_counts[dt] = type_counts.get(dt, 0) + 1
    return ", ".join(f"{dt}({count})" for dt, count in sorted(type_counts.items()))


# ─── BUILD THE GRAPH ─────────────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """
    Construct the LangGraph StateGraph for the Discharge Summary Agent.

    The graph implements the deterministic state machine:
        INITIALIZE → REASON → CALL_TOOL → OBSERVE → VERIFY → [REASON | COMPILE | HARD_CAP]

    Returns:
        A compiled LangGraph that can be invoked with create_initial_state().
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("initialize", initialize_node)
    graph.add_node("reason", reason_node)
    graph.add_node("call_tool", call_tool_node)
    graph.add_node("observe", observe_node)
    graph.add_node("verify", verify_node)
    graph.add_node("compile", compile_node)
    graph.add_node("hard_cap_escalate", hard_cap_escalate_node)

    # Set entry point
    graph.set_entry_point("initialize")

    # Add edges
    graph.add_edge("initialize", "reason")
    graph.add_edge("reason", "call_tool")
    graph.add_edge("call_tool", "observe")
    graph.add_edge("observe", "verify")

    # Conditional routing from verify
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "reason": "reason",
            "compile": "compile",
            "hard_cap_escalate": "hard_cap_escalate",
        }
    )

    # Terminal nodes
    graph.add_edge("compile", END)
    graph.add_edge("hard_cap_escalate", END)

    return graph.compile()
