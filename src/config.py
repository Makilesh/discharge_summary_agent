"""
config.py — Constants & Configuration
======================================

All magic numbers, document taxonomy, extraction priority, output templates,
and safety-critical constants live here. Nothing is hardcoded in agent logic.

Clinical Safety:
    - MAX_ITERATIONS caps runaway agent loops.
    - MAX_RETRIES prevents infinite retry spirals.
    - Templates ensure missing/conflicting data is ALWAYS surfaced, never hidden.
"""

from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# ─── LLM CONFIGURATION ──────────────────────────────────────────────────────────

GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
LLM_TEMPERATURE: float = 0.0  # Deterministic for clinical safety — no creative sampling
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "auto").lower()
# LLM_BACKEND: "auto" tries Gemini first, then local Ollama. "gemini" disables local
# fallback. "local" uses Ollama only.
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
REASONING_BACKUP_MODEL: str = os.getenv("REASONING_BACKUP_MODEL", "deepseek-r1:14b")
VISION_BACKUP_MODEL: str = os.getenv("VISION_BACKUP_MODEL", "qwen2.5vl:7b")

# ─── AGENT CONTROL ───────────────────────────────────────────────────────────────

MAX_ITERATIONS: int = 20       # Hard cap on agent steps per patient document set
MAX_RETRIES: int = 2           # Max retries per failed tool call before marking [UNRESOLVED]
MIN_TEXT_LENGTH: int = 30      # Pages with OCR text below this are considered unreadable
BATCH_SIZE: int = 5            # Max pages to batch-process in a single tool call

# ─── DOCUMENT TAXONOMY ──────────────────────────────────────────────────────────

DOC_TYPES: list[str] = [
    "ADMISSION_RECORD",
    "NURSING_NOTES",
    "NURSING_ASSESSMENT",
    "ER_OBSERVATION_CHART",
    "ICU_CHART",
    "DRUG_CHART",
    "INVESTIGATION_CHECKLIST",
    "LAB_REPORT_BIOCHEMISTRY",
    "LAB_REPORT_HAEMATOLOGY",
    "LAB_REPORT_URINE",
    "LAB_REPORT_ABG",
    "LAB_REPORT_CULTURE",
    "IMAGING_REPORT_USG",
    "IMAGING_REPORT_CT",
    "ECHO_REPORT",
    "PROCEDURE_CHART",
    "MONITORING_CHART_DIABETES",
    "MONITORING_CHART_VITALS",
    "INTAKE_OUTPUT_CHART",
    "CAUTI_CHART",
    "BED_SORES_CHART",
    "CONSULTATION_SHEET",
    "DISCHARGE_CHECKLIST",
    "TYPED_DISCHARGE_SUMMARY",
    "UNKNOWN",
]

# ─── EXTRACTION PRIORITY ORDER ──────────────────────────────────────────────────
# Process in this strict sequence. Do not skip ahead.

EXTRACTION_PRIORITY_ORDER: list[str] = [
    "TYPED_DISCHARGE_SUMMARY",       # 1. Anchor for demographics, dates, official diagnoses
    "ADMISSION_RECORD",              # 2. Chief complaints, past history, provisional diagnosis
    "ER_OBSERVATION_CHART",          # 3. ER diagnosis (may differ from final — log conflict)
    "ICU_CHART",                     # 4. Critical meds, diagnoses written on ICU sheet
    "DRUG_CHART",                    # 5. Full medication history, dates, doses, routes
    "CONSULTATION_SHEET",            # 6. Specialist opinions, evolving diagnoses
    "LAB_REPORT_BIOCHEMISTRY",       # 7a. Lab results
    "LAB_REPORT_HAEMATOLOGY",        # 7b.
    "LAB_REPORT_URINE",             # 7c.
    "LAB_REPORT_ABG",               # 7d.
    "LAB_REPORT_CULTURE",           # 7e.
    "IMAGING_REPORT_USG",           # 8a. Imaging reports
    "IMAGING_REPORT_CT",            # 8b.
    "ECHO_REPORT",                  # 8c.
    "NURSING_NOTES",                # 9. Cross-check treatments described vs charted meds
    "NURSING_ASSESSMENT",           # 9b.
    "MONITORING_CHART_DIABETES",    # 10. Blood glucose + insulin doses (DKA correlation)
    "MONITORING_CHART_VITALS",      # 10b.
    "INTAKE_OUTPUT_CHART",          # 11. Fluid management, Foley status
    "PROCEDURE_CHART",              # 12. Enumerate all bedside procedures
    "DISCHARGE_CHECKLIST",          # 13. Pending items, patient status at exit
    "INVESTIGATION_CHECKLIST",      # 14.
    "CAUTI_CHART",                  # 15.
    "BED_SORES_CHART",             # 16.
]

# ─── OUTPUT SECTIONS ────────────────────────────────────────────────────────────

DISCHARGE_SUMMARY_SECTIONS: list[str] = [
    "patient_demographics",
    "admission_date",
    "discharge_date",
    "principal_diagnosis",
    "secondary_diagnoses",
    "allergies",
    "hospital_course",
    "investigations_summary",
    "procedures_performed",
    "admission_medications",
    "discharge_medications",
    "medication_changes",
    "pending_results",
    "follow_up_instructions",
    "discharge_condition",
    "escalation_flags_for_clinician",
    "conflicts_requiring_review",
    "summary_status",
]

# ─── NON-FABRICATION TEMPLATES ──────────────────────────────────────────────────
# These strings must appear LITERALLY in the output. Never substitute a plausible value.

MISSING_FIELD_TEMPLATE: str = (
    "[MISSING — not documented in source records. Clinician must supply.]"
)
PENDING_FIELD_TEMPLATE: str = (
    "[PENDING — {item} sent {date}. Result not available in source documents.]"
)
CONFLICT_FIELD_TEMPLATE: str = (
    "[CONFLICT — Multiple sources disagree. See escalation_flags. Clinician must resolve.]"
)
UNCLEAR_FIELD_TEMPLATE: str = (
    "[UNCLEAR — Source text partially legible: '{best_guess}'. Verify against original.]"
)

# ─── CRITICAL LAB THRESHOLDS ────────────────────────────────────────────────────
# Used by CR-3 and CR-5 rules to flag dangerously abnormal values

CRITICAL_LAB_THRESHOLDS: dict = {
    "sodium": {"low": 120, "high": 155, "unit": "mmol/L"},
    "potassium": {"low": 2.5, "high": 6.5, "unit": "mmol/L"},
    "glucose": {"low": 50, "high": 400, "unit": "mg/dL"},
    "creatinine": {"high": 5.0, "unit": "mg/dL"},
    "ph": {"low": 7.25, "high": 7.55, "unit": ""},
    "hco3": {"low": 15, "unit": "mmol/L"},
    "wbc": {"high": 20, "unit": "x10^3/uL"},
}

# ─── EXPECTED CONFLICT DETECTIONS (Test Harness) ────────────────────────────────

EXPECTED_CONFLICT_IDS: list[str] = [
    "CONFLICT-001",  # Diagnosis mismatch across ER/Admission/ICU/Consultation
    "CONFLICT-002",  # Insulin treatment without DM in final diagnosis
    "CONFLICT-003",  # Urine culture negative but treated with IV Meropenem
    "CONFLICT-004",  # ABG critical sodium, metabolic acidosis
    "CONFLICT-005",  # Discharge against medical advice
    "CONFLICT-006",  # Pending blood culture result at discharge
    "CONFLICT-007",  # Chief complaint mismatch between ER and Case Record
]
