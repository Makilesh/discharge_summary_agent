# Discharge Summary Agent — Engineering Walkthrough

> A complete account of what was built, why each decision was made, how the system works end-to-end, the bugs we found and fixed, and the final production-grade results.

---

## Table of Contents

1. [The Problem](#1-the-problem)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Part 1 Deep-Dive — The Agent](#3-part-1-deep-dive--the-agent)
   - [3.1 LangGraph State Machine](#31-langgraph-state-machine)
   - [3.2 Agent State Schema](#32-agent-state-schema)
   - [3.3 PDF Processing & OCR Pipeline](#33-pdf-processing--ocr-pipeline)
   - [3.4 Multi-Model Routing](#34-multi-model-routing)
   - [3.5 Document Classification & Extraction](#35-document-classification--extraction)
   - [3.6 Medication Reconciliation](#36-medication-reconciliation)
   - [3.7 Clinical Safety — The Cross-Reference Audit](#37-clinical-safety--the-cross-reference-audit)
   - [3.8 No-Fabrication Guardrail](#38-no-fabrication-guardrail)
   - [3.9 Final Compilation](#39-final-compilation)
   - [3.10 Observability & Trace System](#310-observability--trace-system)
4. [Part 2 Deep-Dive — Learning from Doctor Edits](#4-part-2-deep-dive--learning-from-doctor-edits)
   - [4.1 The Contextual Bandit (UCB1)](#41-the-contextual-bandit-ucb1)
   - [4.2 Edit Signal Engine](#42-edit-signal-engine)
   - [4.3 Simulated Reviewer](#43-simulated-reviewer)
   - [4.4 Correction Memory Bank](#44-correction-memory-bank)
   - [4.5 Gaming Detector](#45-gaming-detector)
   - [4.6 Learning Loop Orchestrator](#46-learning-loop-orchestrator)
5. [Production Bugs Found & Fixed](#5-production-bugs-found--fixed)
6. [Test Suite](#6-test-suite)
7. [Final Results](#7-final-results)
8. [Project File Map](#8-project-file-map)
9. [Known Limitations & Future Work](#9-known-limitations--future-work)

---

## 1. The Problem

Hospitals generate messy, multi-page paper records for each patient stay — handwritten drug charts, scanned lab reports, typed discharge summaries, nursing notes, ICU charts, imaging reports, and more. When a patient is discharged, a clinician must manually compile a **Discharge Summary** from all of these sources. This is:

- **Time-consuming**: A single summary can take 30–60 minutes.
- **Error-prone**: Conflicting diagnoses across documents, missed lab results, medication dosing errors, and incomplete follow-up instructions are common.
- **Safety-critical**: A wrong or missing medication, an unaddressed critical lab value, or a fabricated clinical fact can directly harm the patient.

**The task**: Build an agentic AI system that:
1. Reads a raw hospital PDF (71 pages for our test patient).
2. Classifies every page into its document type (admission record, drug chart, lab report, etc.).
3. Extracts structured clinical data from each page.
4. Cross-references all extracted data for conflicts and safety violations.
5. Compiles a structured, 17-section discharge summary draft.
6. **Never fabricates** clinical facts — missing data is explicitly marked.
7. **Always escalates** conflicts to a human clinician — never silently resolves them.

The system was built in two parts:
- **Part 1**: The agent that produces the discharge summary.
- **Part 2**: A learning loop that uses doctor edit feedback to improve future summaries.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PART 1: Discharge Summary Agent                      │
│                                                                             │
│  ┌─────────┐   ┌──────┐   ┌──────────┐   ┌───────┐   ┌──────┐   ┌──────┐ │
│  │INITIALIZE│──▶│REASON│──▶│CALL_TOOL │──▶│OBSERVE│──▶│VERIFY│──▶│ ...  │ │
│  │ PDF+OCR  │   │      │   │Extract/  │   │       │   │      │   │      │ │
│  │ Classify │   │ Plan │   │Reconcile │   │ Log   │   │Check │   │      │ │
│  └─────────┘   └──────┘   └──────────┘   └───────┘   └──┬───┘   └──────┘ │
│                                                          │                  │
│                              ┌──────────────┐  ┌────────▼────────┐         │
│                              │  HARD_CAP    │  │   route_after   │         │
│                              │  ESCALATE    │◀─┤    _verify()    │         │
│                              └──────────────┘  │  steps<=0?      │         │
│                                                │  queue empty?   │         │
│                                                │  all done?      │         │
│                                                └────────┬────────┘         │
│                                                         │                  │
│                              ┌──────────────┐           │                  │
│                              │   COMPILE    │◀──────────┘                  │
│                              │ CR Audit     │                              │
│                              │ + Markdown   │                              │
│                              └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                     PART 2: Learning from Doctor Edits                       │
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                    │
│  │  Contextual  │──▶│  Simulated   │──▶│  Edit Signal │                    │
│  │  Bandit      │   │  Reviewer    │   │  Engine      │                    │
│  │ (UCB1, 5     │   │ (7 rules +   │   │ (R_SED +     │                    │
│  │  arms)       │   │  LLM pass)   │   │  R_SEC +     │                    │
│  └──────┬───────┘   └──────────────┘   │  R_PEND +    │                    │
│         │                               │  R_SAFE)     │                    │
│         │           ┌──────────────┐   └──────┬───────┘                    │
│         │           │  Correction  │          │                            │
│         ◀───────────┤  Memory Bank │◀─────────┘                            │
│                     │ (JSONL store) │                                       │
│                     └──────────────┘                                        │
│                                                                             │
│  ┌──────────────┐                                                          │
│  │   Gaming     │  Runs every 5 iterations, detects reward hacking.        │
│  │  Detector    │                                                          │
│  └──────────────┘                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Part 1 Deep-Dive — The Agent

### 3.1 LangGraph State Machine

**File**: [graph.py](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py)

The agent is a **deterministic state machine** built with LangGraph's `StateGraph`. This was a deliberate choice over a free-form ReAct agent because:

- **Reproducibility**: Same PDF → same extraction path every time (temperature=0.0).
- **Auditability**: Every state transition is logged with full reasoning.
- **Safety**: No LLM decides "what to do next." The routing function [route_after_verify()](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py#L1020-L1052) is a deterministic Python function with explicit `if/elif/else` conditions.

The phase cycle is:

```
INITIALIZE → REASON → CALL_TOOL → OBSERVE → VERIFY → [REASON | COMPILE | HARD_CAP_ESCALATE]
```

| Node | What it does |
|------|-------------|
| [initialize_node](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py#L45-L196) | Load PDF, render all pages as PNG images, batch-classify every page into one of 24 document types |
| [reason_node](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py#L201-L320) | Examine the processing queue, decide what to extract next (strict priority order), plan the tool call |
| [call_tool_node](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py#L325-L355) | Execute the planned extraction — OCR the page, extract structured data, route results to state fields |
| [observe_node](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py#L805-L830) | Lightweight logging node — never modifies data, only records what was received |
| [verify_node](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py#L835-L872) | Check extraction completeness. Count completed vs missing fields, remaining queue items |
| [compile_node](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py#L877-L937) | Run the mandatory cross-reference audit (CR-1 through CR-5), generate hospital course narrative, compile final Markdown summary |
| [hard_cap_escalate_node](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py#L942-L1015) | Emergency exit when step cap is reached — compile with `[MISSING]` markers and a CRITICAL escalation flag |

**Key invariant**: The graph is built once in [build_agent_graph()](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py#L1313-L1359) and compiled into a runnable. The entry point is `initialize`, and the terminal nodes are `compile` and `hard_cap_escalate`.

---

### 3.2 Agent State Schema

**File**: [state.py](file:///d:/GEN%20AI/discharge_summary_agent/src/state.py)

The state is a `TypedDict` with 25+ fields organized into 7 categories:

| Category | Fields | Purpose |
|----------|--------|---------|
| **Document Inventory** | `loaded_documents`, `unreadable_pages`, `page_images` | Track every page: its classification, raw text, and base64 image |
| **Iteration Control** | `steps_remaining`, `current_phase`, `processing_queue` | Enforce the 25-step budget and priority-ordered extraction queue |
| **Clinical Fields** | `extracted_demographics`, `diagnoses`, `hospital_course`, `allergies`, etc. | The actual medical data extracted from the PDF |
| **Medications** | `admission_medications`, `inpatient_medications`, `discharge_medications`, `medication_reconciliation` | Three separate medication lists + reconciliation output |
| **Investigations** | `lab_results`, `imaging_results`, `pending_results` | Lab values, imaging impressions, and anything still pending |
| **Safety** | `conflicts`, `escalation_flags`, `fabrication_blocks` | Every detected conflict, every escalation, every prevented fabrication |
| **Observability** | `trace`, `final_summary` | Full audit trail and the compiled output |

> [!IMPORTANT]
> Most list fields use `Annotated[list[...], operator.add]`. This tells LangGraph to **append** updates instead of replacing. This is critical — an extraction node that finds 5 lab results on page 30 should *add* to the existing list, not overwrite results from page 20.

The [create_initial_state()](file:///d:/GEN%20AI/discharge_summary_agent/src/state.py#L131-L193) factory initializes everything to empty/None. No data is ever invented at initialization.

---

### 3.3 PDF Processing & OCR Pipeline

**File**: [pdf_processor.py](file:///d:/GEN%20AI/discharge_summary_agent/src/pdf_processor.py)

Uses **PyMuPDF (fitz)** to render every page of the PDF as a high-resolution PNG image. These images are then base64-encoded and stored in the state for use by the vision LLM.

**OCR** is performed by the `extract_page_text()` function in [tools.py](file:///d:/GEN%20AI/discharge_summary_agent/src/tools.py). It sends each page image to a Gemini Vision model with a clinical OCR prompt and returns the extracted text plus a confidence score.

**Caching**: OCR results are cached to disk (under `output/ocr_cache/{namespace}/page_{N}.txt`). The cache namespace is derived from the PDF path + model configuration, so different models or different PDFs never share cache. This was essential for development — OCR-ing 71 pages costs ~71 API calls, and we needed to iterate on downstream logic without re-running OCR every time.

**Extraction caching**: Similarly, structured extraction results (the JSON output from `extract_clinical_data()`, `extract_lab_report()`, etc.) are cached per-page per-document-type. This means re-running the agent after a code change to the compiler or cross-reference logic doesn't re-extract any data.

---

### 3.4 Multi-Model Routing

**File**: [config.py](file:///d:/GEN%20AI/discharge_summary_agent/src/config.py#L32-L107)

The system runs on **Gemini Free Tier**, which imposes strict per-model rate limits. A single model would bottleneck at 10 RPM (requests per minute). The solution: distribute API calls across 4 models with independent quotas.

| Model | RPM | RPD | Role |
|-------|-----|-----|------|
| `gemini-3.1-flash-lite` | 15 | 1000 | Bulk OCR, page classification, simple doc types (nursing notes, vitals, checklists) |
| `gemini-3.5-flash` | 10 | 1500 | Complex clinical reasoning — lab extraction, drug chart parsing, compilation |
| `gemini-2.5-flash` | 10 | 250 | Overflow fallback when primary is rate-limited |
| `gemini-3-flash-preview` | 10 | 1500 | Secondary backup, equal capabilities to primary |

The routing is defined in `MODEL_MAPPING` — a dict mapping each document type and task to its assigned model. For example:

```python
"OCR": MODEL_LITE,              # High-frequency, low-complexity
"LAB_REPORT_BIOCHEMISTRY": MODEL_PRIMARY,  # Needs clinical reasoning
"NURSING_NOTES": MODEL_LITE,    # Simple extraction
"COMPILATION": MODEL_PRIMARY,   # Most important task
```

**Fallback chain**: When a model returns HTTP 429 (rate limited), the system automatically tries the next model in [MODEL_FALLBACK_CHAIN](file:///d:/GEN%20AI/discharge_summary_agent/src/config.py#L102-L107). For example, if `gemini-3.5-flash` is throttled, it tries `gemini-3-flash-preview`, then `gemini-2.5-flash`.

**RPM throttling**: The [_rpm_sleep()](file:///d:/GEN%20AI/discharge_summary_agent/src/tools.py) function tracks the timestamp of the last call to each model and sleeps if necessary to stay under the per-model RPM limit.

**Local Ollama fallback**: If all Gemini models are exhausted (or the network is down), the system can fall back to local models:
- `qwen2.5vl:7b` for vision tasks (OCR, classification)
- `deepseek-r1:14b` for reasoning tasks (extraction, compilation)

---

### 3.5 Document Classification & Extraction

**Classification** uses the LITE model to batch-classify pages. The prompt asks: "For each page image, identify the document type from this taxonomy: [24 types]." The 24 types cover everything a hospital record might contain:

```
ADMISSION_RECORD, NURSING_NOTES, ER_OBSERVATION_CHART, ICU_CHART, DRUG_CHART,
LAB_REPORT_BIOCHEMISTRY, LAB_REPORT_HAEMATOLOGY, LAB_REPORT_URINE, LAB_REPORT_ABG,
LAB_REPORT_CULTURE, IMAGING_REPORT_USG, IMAGING_REPORT_CT, ECHO_REPORT,
MONITORING_CHART_DIABETES, MONITORING_CHART_VITALS, CONSULTATION_SHEET, ...
```

Classification results are cached, sanitized (alias resolution like `NURSES_NOTES` → `NURSING_NOTES`), and deduplicated (one classification per page, keeping highest confidence).

**Extraction** is priority-ordered — defined in [EXTRACTION_PRIORITY_ORDER](file:///d:/GEN%20AI/discharge_summary_agent/src/config.py#L149-L174). The typed discharge summary is processed first (it's the anchor for demographics and dates), followed by admission record, ER chart, ICU chart, drug chart, etc. This ordering matters because:

1. Higher-priority types inform the interpretation of lower-priority ones.
2. If the step cap is reached, we've already extracted the most important data.
3. Cross-reference depends on having diagnoses from multiple sources before running.

**Batch processing**: Same-type pages are grouped into batches. For example, if pages 15–28 are all drug chart pages, they're extracted in a single batch call to `extract_drug_chart_batch()`, which preserves cross-page medication timeline context. This is much more efficient than 14 separate calls and produces better results because the model sees the full medication timeline.

**Data routing**: After extraction, the [_route_extracted_data()](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py#L572-L738) function distributes results to the correct state fields. Lab results go to `lab_results`, medications go to the appropriate list (`admission_medications`, `inpatient_medications`, or `discharge_medications`), diagnoses are accumulated across sources, etc.

---

### 3.6 Medication Reconciliation

**File**: [tools.py — reconcile_medications()](file:///d:/GEN%20AI/discharge_summary_agent/src/tools.py#L1110-L1220)

Medication reconciliation compares three lists — admission meds, inpatient meds, and discharge meds — and classifies each drug as:

- **CONTINUED**: Present in both admission/inpatient AND discharge.
- **STOPPED**: Present during stay but absent from discharge advice.
- **ADDED**: Present in discharge but not during the stay.
- **DOSE_CHANGED**: Same drug, different dose between admission and discharge.

**The deduplication challenge**: Hospital drug charts are handwritten. OCR produces variant spellings of the same drug:
- `"HAPPY NERVE PLUS"` vs `"HAPPYNERVE PLUS"`
- `"H.ACTRAPID"` vs `"INSULIN H. ACTRAPID"`
- `"SUMOL"` vs `"SUMEL"` (same medication, OCR misread)

We built a 3-layer dedup pipeline:

1. **Normalize**: Strip prefixes (`INJ`, `TAB`, `CAP`, `SYP`), collapse dots, hyphens, and extra whitespace, uppercase everything.
2. **Fuzzy match**: Compute Levenshtein similarity between all pairs. If ≥ 75% similar → merge (keep the longer name as canonical).
3. **Substring containment**: If one name is a substring of another → merge. This catches `"H ACTRAPID"` being absorbed into `"INSULIN H ACTRAPID"`.

**Result**: 20 raw medication entries → 16 unique drugs after dedup.

Every medication flagged as STOPPED without a documented reason triggers a WARNING escalation — the clinician must verify whether the discontinuation was intentional.

---

### 3.7 Clinical Safety — The Cross-Reference Audit

**File**: [cross_reference.py](file:///d:/GEN%20AI/discharge_summary_agent/src/cross_reference.py)

This is the **most critical module** in the system. It runs 5 rules (CR-1 through CR-5) that detect inter-document conflicts and safety violations. The audit is **mandatory** — it runs before every compilation, even under step cap pressure.

#### CR-1: Treatment–Diagnosis Alignment
[check_cr1_treatment_diagnosis_alignment()](file:///d:/GEN%20AI/discharge_summary_agent/src/cross_reference.py#L89-L199)

For every active inpatient treatment, verify a corresponding diagnosis exists. Examples:
- Insulin (Lantus, Actrapid, Humalog) → must have DM, DKA, T2DM, or T1DM in diagnoses
- IV Meropenem → must have infection/sepsis/UTI/pneumonia in diagnoses
- Sodium bicarbonate → must have DKA or metabolic acidosis

Also checks: is every inpatient medication accounted for in the discharge advice? If not, flag it — a med that was active during the stay but missing from discharge advice could be an unintentional omission.

#### CR-2: Inter-Document Diagnosis Conflict
[check_cr2_diagnosis_conflicts()](file:///d:/GEN%20AI/discharge_summary_agent/src/cross_reference.py#L204-L385)

Collects diagnoses from every source (ER chart, admission record, ICU chart, consultation sheets, typed discharge summary) and compares them. If the ER says "DKA" but the final discharge summary says "TAFE" and the ICU says "DKA + T2DM" — that's a CR-2 conflict.

The agent does **not** pick one diagnosis over another. It flags the conflict and presents all sources to the clinician.

Also checks chief complaint mismatches — if the ER chief complaint differs from the admission record chief complaint, that's flagged too.

#### CR-3: Lab Evidence vs Clinical Claim
[check_cr3_lab_evidence()](file:///d:/GEN%20AI/discharge_summary_agent/src/cross_reference.py#L390-L543)

Checks every lab result against [CRITICAL_LAB_THRESHOLDS](file:///d:/GEN%20AI/discharge_summary_agent/src/config.py#L218-L227):

| Analyte | Critical Low | Critical High | Unit |
|---------|-------------|--------------|------|
| Sodium | 120 | 155 | mmol/L |
| Potassium | 2.5 | 6.5 | mmol/L |
| Glucose | 50 | 400 | mg/dL |
| Creatinine | — | 5.0 | mg/dL |
| pH (blood) | 7.25 | 7.55 | — |
| HCO3 | 15 | — | mmol/L |
| WBC | — | 20 | x10³/µL |

Also checks: if the discharge summary claims "improved" or "resolved" but the last labs still show abnormal values → flag the contradiction.

Also checks: pending results at discharge. Items that are genuinely pending investigations (blood culture sent, biopsy results awaited) are escalated. Non-lab items (IV cannula, catheter) are filtered out.

> [!IMPORTANT]
> **Production-grade matching** was critical here. The original implementation used Python substring matching (`if "ph" in test_name`), which caused 21 false CRITICAL alerts because `"ph"` matches `"neutrophils"`, `"lymphocytes"`, `"eosinophils"`, etc. We fixed this with word-boundary regex matching. See [Section 5](#5-production-bugs-found--fixed) for details.

#### CR-4: Culture Result vs Treatment
[check_cr4_culture_treatment()](file:///d:/GEN%20AI/discharge_summary_agent/src/cross_reference.py#L548-L637)

If culture results show "NO SIGNIFICANT BACTERIURIA" or "STERILE" but the patient was given broad-spectrum IV antibiotics (Meropenem, Piperacillin, etc.) → flag the mismatch. This isn't necessarily an error (empirical antibiotic use is common) but the clinician must annotate why.

#### CR-5: Discharge Condition Claim
[check_cr5_discharge_condition()](file:///d:/GEN%20AI/discharge_summary_agent/src/cross_reference.py#L642-L750)

Two checks:
1. **DAMA detection**: Scans all documents for keywords like "against medical advice", "on request", "not willing", "self-discharge". If found → CRITICAL escalation. DAMA status changes clinical responsibility and follow-up urgency.
2. **Vitals contradiction**: If discharge condition says "hemodynamically stable" but last vitals show tachycardia (pulse > 100), hypotension (SBP < 90), or fever (temp > 99.5°F) → flag it.

---

### 3.8 No-Fabrication Guardrail

This is the core safety constraint. The agent uses 4 template strings (defined in [config.py](file:///d:/GEN%20AI/discharge_summary_agent/src/config.py#L199-L213)) that appear **literally** in the output:

```
[MISSING — not documented in source records. Clinician must supply.]
[PENDING — {item} sent {date}. Result not available in source documents.]
[CONFLICT — Multiple sources disagree. See escalation_flags. Clinician must resolve.]
[UNCLEAR — Source text partially legible: '{best_guess}'. Verify against original.]
```

The agent **never** substitutes a plausible value. If the patient's allergies aren't documented anywhere in 71 pages, the output says `[MISSING]`, not "NKDA" (No Known Drug Allergies).

**Fabrication blocks**: When a tool call fails after 2 retries, the affected field is added to `fabrication_blocks[]` with the specific tool and failure reason. This list appears in the summary status section.

---

### 3.9 Final Compilation

**File**: [compiler.py](file:///d:/GEN%20AI/discharge_summary_agent/src/compiler.py)

The compiler assembles a 17-section Markdown document from the state. It's purely a formatting step — it never calls an LLM and never invents data.

The 17 sections are:

1. Patient Demographics
2. Admission Date
3. Discharge Date
4. Principal Diagnosis (with `[CONFLICT]` if CR-2 fired)
5. Secondary Diagnoses
6. Allergies
7. Hospital Course (synthesized narrative with page citations)
8. Investigations Summary (lab table + imaging list)
9. Procedures Performed
10. Admission Medications (table)
11. Discharge Medications (table)
12. Medication Changes (reconciliation table with flags)
13. Pending Results
14. Follow-Up Instructions
15. Discharge Condition (with DAMA warnings if CR-5 fired)
16. ⚠️ Escalation Flags for Clinician (CRITICAL + WARNING sections)
17. Conflicts Requiring Review (full CR rule details)

Every summary ends with:

> **DRAFT — NOT FOR CLINICAL USE WITHOUT REVIEW**

And a status block listing fields completed, fields missing, total conflicts, total escalation flags, and fabrication blocks prevented.

---

### 3.10 Observability & Trace System

**File**: [trace.py](file:///d:/GEN%20AI/discharge_summary_agent/src/trace.py)

Every state transition emits a structured trace entry via [emit_trace()](file:///d:/GEN%20AI/discharge_summary_agent/src/trace.py):

```json
{
  "step_number": 5,
  "phase": "CALL_TOOL",
  "reasoning": "Extracting lab report from page 30",
  "action": "EXTRACTION_COMPLETE",
  "tool_name": "extract_lab_report",
  "observation": "Found 12 lab results",
  "decision": "Route to lab_results state field",
  "fields_updated": ["lab_results"],
  "fallback_taken": false
}
```

The full trace is saved to `output/trace.json` and provides complete audit transparency. For Patient 2, this trace contains **108 entries** — every classification, OCR, extraction, reconciliation, and escalation decision.

---

## 4. Part 2 Deep-Dive — Learning from Doctor Edits

Part 2 answers the question: **Can the agent learn from a doctor's corrections to produce better drafts over time?**

The approach: a **contextual bandit** that selects among 5 different compilation strategies (prompt arms), measures how much the doctor had to edit each draft, and converges on the strategy that minimizes edit burden while preserving clinical safety.

### 4.1 The Contextual Bandit (UCB1)

**File**: [bandit.py](file:///d:/GEN%20AI/discharge_summary_agent/src/bandit.py)

Uses the **Upper Confidence Bound (UCB1)** algorithm. UCB1 balances exploration (trying under-sampled arms) with exploitation (using the best-known arm).

The 5 prompt strategy arms:

| Arm | Name | Strategy |
|-----|------|----------|
| 0 | BASELINE | Standard compiler prompt. The control condition. |
| 1 | SECTION_EXEMPLARS | Injects per-section correction examples from memory. |
| 2 | GLOBAL_PREAMBLE | Prepends all correction history as a global preamble. |
| 3 | CONFLICT_FIRST | Safety-first: checks escalation flags before generating each section. |
| 4 | HYBRID_EXEMPLAR | Combines safety-first + per-section exemplars. |

Every arm includes the `ANTI_VAGUENESS_INSTRUCTION` — a hard constraint that prevents the agent from learning to produce shorter/vaguer output to game the edit distance metric.

The bandit state (arm pull counts, mean rewards, history) is persisted atomically to `bandit_state.json` using tmp-file + `os.replace()` to prevent partial writes.

---

### 4.2 Edit Signal Engine

**File**: [edit_signal.py](file:///d:/GEN%20AI/discharge_summary_agent/src/edit_signal.py)

Computes a scalar reward from comparing a draft against the doctor's edited version. The reward is a weighted composite of 4 sub-signals:

```
R = 0.35 × R_SED + 0.35 × R_SEC + 0.15 × R_PEND + 0.15 × R_SAFE
```

| Sub-signal | Weight | Measures | Range |
|-----------|--------|---------|-------|
| **R_SED** | 0.35 | Normalized edit distance (lower edits → higher reward) | 0.0–1.0 |
| **R_SEC** | 0.35 | Section-level accuracy on 7 critical sections | 0.0–1.0 |
| **R_PEND** | 0.15 | Pending results coverage (did we catch all pending items?) | 0.0–1.0 |
| **R_SAFE** | 0.15 | Safety flag preservation (binary: were all escalation flags kept?) | 0.0 or 1.0 |

> [!CAUTION]
> **Safety Clamp**: If `R_SAFE = 0.0` (any escalation flag was dropped), the composite reward is clamped to `max(R, 0.10)` regardless of other sub-scores. A draft that drops a safety flag can **never** earn a high reward.

The 7 scored sections (for R_SEC) are: principal diagnosis, discharge medications, medication changes, pending results, hospital course, discharge condition, and escalation flags.

---

### 4.3 Simulated Reviewer

**File**: [simulated_reviewer.py](file:///d:/GEN%20AI/discharge_summary_agent/src/simulated_reviewer.py)

Since we don't have access to a real doctor during training, the reviewer is simulated with:

1. **7 deterministic rules** (pattern-matching corrections):
   - REV-001: Missing section headings
   - REV-002: Unclear/ambiguous medication dosing
   - REV-003: Missing source page citations
   - REV-004: Escalation flags not prominently displayed
   - REV-005: Pending results not listed
   - REV-006: Allergies marked `[MISSING]` when "NOT KNOWN" is documented
   - REV-007: DAMA not flagged prominently

2. **An LLM correction pass** (using Gemini) that applies clinically-motivated edits with a fabrication guard — the LLM is instructed to never add clinical facts not present in the source data.

---

### 4.4 Correction Memory Bank

**File**: [correction_memory.py](file:///d:/GEN%20AI/discharge_summary_agent/src/correction_memory.py)

A **JSONL-backed persistent store** that records every correction the reviewer makes. Each entry contains:
- The section name
- The original text
- The corrected text
- The rule that triggered the correction
- An impact score

Retrieval uses a two-stage process:
1. **Exact section match**: Find corrections for the same section.
2. **Similarity ranking**: Sort by impact score and recency.

The top-K corrections are injected into the compiler prompt (for arms that use per-section or global exemplars).

---

### 4.5 Gaming Detector

**File**: [bandit.py — GamingDetector](file:///d:/GEN%20AI/discharge_summary_agent/src/bandit.py#L466-L614)

Runs after every 5 training iterations. Detects two specific gaming patterns:

1. **Vagueness gaming**: Reward increased but `principal_diagnosis` accuracy decreased → the agent may be producing vaguer output that's easier to "edit" but less clinically accurate.
2. **Summarization collapse**: Hospital course word count decreased > 20% over 3 consecutive iterations → the agent is learning to write shorter (cheaper) narratives instead of more accurate ones.

Gaming alerts are logged to `gaming_alerts.log` and are informational — they never modify the learning loop. They signal that human review is needed.

---

### 4.6 Learning Loop Orchestrator

**File**: [learning_loop.py](file:///d:/GEN%20AI/discharge_summary_agent/src/learning_loop.py)

Ties everything together:

```
for iteration in range(n_train):
    1. Bandit selects an arm (prompt strategy)
    2. Compile the discharge summary using that arm's prompt
    3. Simulated reviewer edits the draft
    4. Edit Signal Engine computes the reward
    5. Correction Memory Bank stores new corrections
    6. Bandit updates arm statistics with the reward
    7. Gaming Detector checks for reward hacking (every 5 iterations)
```

After training, it runs a held-out evaluation with the best arm and produces:
- `training_curve.json` — per-iteration metrics
- `before_after_report.md` — baseline vs best arm comparison
- `improvement_curve.png` — visual learning curve
- `correction_memory_summary.md` — top correction patterns
- `limitations_analysis.md` — documented failure modes

---

## 5. Production Bugs Found & Fixed

We ran a comprehensive clinical safety audit and found 5 production-grade bugs. All were fixed and verified.

### Bug 1 — CR-3 Substring False Positives (21 false CRITICAL alerts)

**Root cause**: The CR-3 lab threshold matching used Python substring matching:

```python
# BEFORE (broken)
if threshold_key in test_name:  # "ph" in "neutrophils" → True!
```

This meant the pH threshold (7.25–7.55) was applied to neutrophil percentages (58%), lymphocyte percentages (32%), eosinophil percentages (2%), and basophil percentages (1%). All of these were flagged as CRITICAL because their percentage values fall outside pH range.

**Fix**: Word-boundary regex matching in [cross_reference.py](file:///d:/GEN%20AI/discharge_summary_agent/src/cross_reference.py#L431):

```python
# AFTER (fixed)
if re.search(rf'(?<![a-z]){re.escape(threshold_key)}(?![a-z])', test):
```

Now `"ph"` only matches `"ph"`, `"blood ph"`, `"arterial ph"` — not `"neutrophils"` or `"lymphocytes"`.

**Impact**: 21 false CRITICAL alerts eliminated.

---

### Bug 2 — Urine pH vs Blood pH

**Root cause**: Even with word-boundary matching, `"urine ph"` would still match the `"ph"` threshold key. But urine pH of 5.5 is clinically normal (range 4.5–8.0), while blood pH of 5.5 would be fatal.

**Fix**: Added `"exclude"` support to [CRITICAL_LAB_THRESHOLDS](file:///d:/GEN%20AI/discharge_summary_agent/src/config.py#L224):

```python
"ph": {"low": 7.25, "high": 7.55, "unit": "", "exclude": ["urine"]}
```

The CR-3 logic now checks exclusion patterns before comparing values.

**Impact**: 2 more false CRITICAL alerts eliminated.

---

### Bug 3 — WBC Unit Mismatch

**Root cause**: The WBC threshold is defined as 20 × 10³/µL, but the lab report gives raw counts in Cells/cumm (e.g., 7160 Cells/cumm). Without unit normalization, 7160 > 20 → false CRITICAL.

**Fix**: Added unit normalization in [cross_reference.py](file:///d:/GEN%20AI/discharge_summary_agent/src/cross_reference.py#L418-L424):

```python
if any(kw in test for kw in ("wbc", "total count")) and value > 100:
    if any(u in unit for u in ("cells", "cumm", "/ul")):
        comparison_value = value / 1000.0  # 7160 → 7.16
```

Now 7.16 < 20 → correctly not flagged.

---

### Bug 4 — Gemini Response Content as List

**Root cause**: Some Gemini models return `response.content` as a list of content parts (`[{"text": "..."}]`) instead of a plain string. This caused `'list' object has no attribute 'strip'` during page classification.

**Fix**: Added type-checking normalization in [tools.py](file:///d:/GEN%20AI/discharge_summary_agent/src/tools.py#L318-L330):

```python
if isinstance(raw_content, list):
    text_parts = [p if isinstance(p, str) else p.get("text", "") for p in raw_content]
    raw_content = "\n".join(text_parts)
```

---

### Bug 5 — Medication Deduplication & Empty Names

**Root cause**: OCR artifacts produced blank medication names and near-duplicate spellings.

**Fix**: Two changes:
1. Filter out blank/empty drug names before processing.
2. Implemented the 3-layer fuzzy dedup pipeline (normalize → Levenshtein ≥ 75% → substring containment).

**Impact**: 20 raw entries → 16 unique medications. No duplicate or ghost entries.

---

### Bug 6 — Non-Lab Items in Pending Results

**Root cause**: "IV CANNULA" was being flagged as a pending investigation result.

**Fix**: Added [NON_LAB_PENDING_KEYWORDS](file:///d:/GEN%20AI/discharge_summary_agent/src/config.py#L230-L234) (cannula, catheter, drain, tube, etc.) and filtered them in CR-3 before escalation.

---

## 6. Test Suite

```
56 tests total
├── tests/test_agent.py  — 35 tests (Part 1)
│   ├── CR-1 through CR-5 cross-reference rules
│   ├── Medication reconciliation (stopped, added, continued, dose changed)
│   ├── Hard cap escalation behavior
│   ├── State validation completeness
│   ├── Trace emission correctness
│   ├── Graph extraction routing (batch, dedup, classification cache)
│   ├── Ollama fallback routing
│   └── OCR caching
│
└── tests/test_part2.py  — 21 tests (Part 2)
    ├── Edit signal computation (all 4 sub-signals + safety clamp)
    ├── Contextual bandit (UCB1 selection, arm updates, persistence)
    ├── Simulated reviewer (7 deterministic rules)
    ├── Correction memory (deduplication, retrieval ranking)
    └── Gaming detection (vagueness + summarization collapse)
```

All 56 tests pass. Run time: ~23 seconds.

---

## 7. Final Results

### Patient 2 — 71 Pages, 15 Document Types

| Metric | Early Build | After CR-3 Fix | **Production Grade** |
|---|---|---|---|
| **Conflicts** | 27 | 6 | **3** |
| **Escalation flags** | 45 | 24 | **18** |
| **CRITICAL flags** | 25 | 4 | **2** |
| **False positives** | 21+ | 2 | **0** |
| **Fabrication blocks** | 0 | 0 | **0** |
| **Runtime** | 246s | 246s | **208s** |
| **Tests passing** | 56/56 | 56/56 | **56/56** |

### The 2 Genuine CRITICAL Alerts
1. 🚨 **Sodium [Na+] 114 mmol/L** — severe hyponatremia (reference: 136–146 mmol/L). This is a life-threatening electrolyte imbalance.
2. 🚨 **DAMA** — Discharge Against Medical Advice. Detected from "not willing" (Page 2) and "discharge on request" (Page 56). This changes clinical responsibility and follow-up urgency.

### The 3 Legitimate Conflicts
1. **CR-2**: Chief complaint mismatch between ER and admission record.
2. **CR-3**: Critical sodium 114 mmol/L — acknowledged and escalated.
3. **CR-5**: DAMA detection — patient discharged against advice.

### The 16 WARNING Flags
All are medication reconciliation warnings — inpatient drugs with no documented reason for discontinuation at discharge. Each one is a genuine safety check for the reviewing clinician.

### Part 2 Learning Results (10 iterations)

| Metric | Value |
|--------|-------|
| Mean composite reward | 0.9934 |
| R_SED (edit distance) | 0.981 |
| R_SEC (section accuracy) | 1.000 |
| R_PEND (pending results) | 1.000 |
| R_SAFE (safety flags) | 1.000 |
| Safety clamps triggered | 0 |
| Fabrication blocks | 0 |
| Gaming alerts | 0 |
| Best arm | BASELINE |

The flat reward curve reflects the high quality of Part 1's draft — the only consistent correction is REV-006 (allergies `[MISSING]` → `NOT KNOWN`). With a multi-patient training corpus, arm differentiation would emerge.

---

## 8. Project File Map

```
d:\GEN AI\discharge_summary_agent\
│
├── src/                              # Core source code
│   ├── __init__.py
│   ├── state.py                      # AgentState TypedDict, ClinicalFlag, MedicationEntry
│   ├── config.py                     # All constants, model routing, thresholds, templates
│   ├── trace.py                      # Trace emission, state validation, summary
│   ├── pdf_processor.py              # PDF → page images (PyMuPDF)
│   ├── tools.py                      # Multi-model LLM extraction tools (1432 lines)
│   ├── cross_reference.py            # CR-1 through CR-5 audit rules (907 lines)
│   ├── compiler.py                   # Final Markdown summary compiler
│   ├── graph.py                      # LangGraph StateGraph definition (1359 lines)
│   ├── edit_signal.py                # Part 2: Weighted edit distance reward
│   ├── simulated_reviewer.py         # Part 2: 7-rule deterministic reviewer + LLM
│   ├── correction_memory.py          # Part 2: JSONL-backed correction pattern store
│   ├── bandit.py                     # Part 2: UCB1 contextual bandit + GamingDetector
│   └── learning_loop.py              # Part 2: Training loop orchestrator
│
├── tests/
│   ├── test_agent.py                 # 35 unit tests (Part 1)
│   └── test_part2.py                 # 21 unit tests (Part 2)
│
├── run_agent.py                      # CLI entry point — Part 1
├── run_learning.py                   # CLI entry point — Part 2
├── requirements.txt                  # Python dependencies
├── .env                              # API keys (not committed)
├── README.md                         # Project documentation
│
├── output/                           # Agent outputs
│   ├── discharge_summary.md          # Final 17-section summary
│   ├── trace.json                    # Full audit trail (108 entries)
│   ├── state.json                    # Final agent state
│   └── ocr_cache/                    # Cached OCR + extraction results
│
└── patient 2 (1).pdf                 # Test patient record (71 pages)
```

### Total Codebase Size

| Component | Lines of Code |
|-----------|--------------|
| [tools.py](file:///d:/GEN%20AI/discharge_summary_agent/src/tools.py) | 1,432 |
| [graph.py](file:///d:/GEN%20AI/discharge_summary_agent/src/graph.py) | 1,359 |
| [cross_reference.py](file:///d:/GEN%20AI/discharge_summary_agent/src/cross_reference.py) | 907 |
| [bandit.py](file:///d:/GEN%20AI/discharge_summary_agent/src/bandit.py) | 614 |
| [edit_signal.py](file:///d:/GEN%20AI/discharge_summary_agent/src/edit_signal.py) | 411 |
| [compiler.py](file:///d:/GEN%20AI/discharge_summary_agent/src/compiler.py) | 403 |
| [learning_loop.py](file:///d:/GEN%20AI/discharge_summary_agent/src/learning_loop.py) | ~800 |
| [simulated_reviewer.py](file:///d:/GEN%20AI/discharge_summary_agent/src/simulated_reviewer.py) | ~750 |
| [correction_memory.py](file:///d:/GEN%20AI/discharge_summary_agent/src/correction_memory.py) | ~500 |
| [config.py](file:///d:/GEN%20AI/discharge_summary_agent/src/config.py) | 247 |
| [state.py](file:///d:/GEN%20AI/discharge_summary_agent/src/state.py) | 194 |
| [trace.py](file:///d:/GEN%20AI/discharge_summary_agent/src/trace.py) | ~250 |
| [pdf_processor.py](file:///d:/GEN%20AI/discharge_summary_agent/src/pdf_processor.py) | ~120 |
| Test files | ~1,800 |
| **Total** | **~9,800** |

---

## 9. Known Limitations & Future Work

### Current Limitations

1. **Demographics**: The agent correctly marks demographics as `[MISSING]` rather than guessing, but could improve by parsing admission record headers for name/age/gender/MRN.
2. **Drug interaction lookup**: Currently mocked with a static dictionary. Production use requires a real pharmacopeia API (RxNorm, DrugBank).
3. **Single-patient evaluation**: All testing and training on Patient 2's clinical profile. Multi-patient corpus would demonstrate broader robustness.
4. **OCR quality**: Gemini Vision handles handwriting well but heavily degraded scans produce `[UNCLEAR]` markers.

### Future Improvements

1. **Multi-patient batch mode**: Process multiple PDFs in parallel with rate-limiter coordination across workers.
2. **Real drug interaction API**: Integrate RxNorm or DrugBank for genuine contraindication/interaction detection.
3. **Structured output validation (Pydantic)**: Define strict schemas for every LLM response. Auto-retry on validation failure.
4. **Token budget tracking**: Log `usage_metadata` on every LLM call, save cumulative cost to trace.
5. **Semantic citation verification**: For every `[Page N]` citation, verify the claim actually exists on that page.
6. **Thompson Sampling**: Implement alongside UCB1 and compare regret bounds.
