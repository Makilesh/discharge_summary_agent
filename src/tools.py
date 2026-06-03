"""
tools.py — Tool Functions for Clinical Data Extraction
========================================================

Each tool wraps a Gemini Vision LLM call with structured prompting.
Tools are the agent's interface to the patient documents.

Clinical Safety:
    - Every tool returns structured data or None — never fabricated values.
    - safe_tool_call() enforces max 2 retries before marking [UNRESOLVED].
    - LLM calls use temperature=0.0 for deterministic extraction.
    - All tool outputs are logged in the trace for audit.
"""

from __future__ import annotations
import json
import re
import os
from pathlib import Path
from typing import Optional, Callable, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .config import (
    GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    LLM_BACKEND, OLLAMA_BASE_URL, REASONING_BACKUP_MODEL, VISION_BACKUP_MODEL,
    DOC_TYPES, MAX_RETRIES, MIN_TEXT_LENGTH,
)
from .trace import emit_trace


# ─── LLM SINGLETON ──────────────────────────────────────────────────────────────

_llm: Optional[ChatGoogleGenerativeAI] = None
_llm_model_name: Optional[str] = None


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def get_backend() -> str:
    backend = _env("LLM_BACKEND", LLM_BACKEND).lower()
    return backend if backend in {"auto", "gemini", "local"} else "auto"


def get_gemini_model_name() -> str:
    return _env("LLM_MODEL", LLM_MODEL)


def get_reasoning_model_name() -> str:
    return _env("REASONING_BACKUP_MODEL", REASONING_BACKUP_MODEL)


def get_vision_model_name() -> str:
    return _env("VISION_BACKUP_MODEL", VISION_BACKUP_MODEL)


def get_ollama_base_url() -> str:
    return _env("OLLAMA_BASE_URL", OLLAMA_BASE_URL)


def get_ocr_cache_file(page_num: int) -> Path:
    """
    Return the OCR cache path for a page.

    Runtime agent runs set OCR_CACHE_NAMESPACE from the PDF path so stale text
    from another record cannot be reused. Unit tests and direct tool calls keep
    the legacy filename unless a namespace is configured.
    """
    namespace = os.getenv("OCR_CACHE_NAMESPACE", "").strip()
    if namespace:
        cache_dir = Path(os.getenv("OCR_CACHE_DIR", "output/ocr_cache"))
        return cache_dir / namespace / f"page_{page_num}.txt"
    return Path(f"patient2_page_{page_num}.txt")


def get_llm() -> ChatGoogleGenerativeAI:
    """Get or create the Gemini Vision LLM instance."""
    global _llm, _llm_model_name
    model_name = get_gemini_model_name()
    if _llm is None or _llm_model_name != model_name:
        _llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=_env("GOOGLE_API_KEY", GOOGLE_API_KEY),
            temperature=LLM_TEMPERATURE,
            max_output_tokens=8192,
        )
        _llm_model_name = model_name
    return _llm


def _call_ollama_text(prompt: str, model: Optional[str] = None) -> str:
    ollama_llm = ChatOpenAI(
        model=model or get_reasoning_model_name(),
        openai_api_key="ollama",
        base_url=get_ollama_base_url(),
        temperature=0.0,
    )
    response = ollama_llm.invoke([HumanMessage(content=prompt)])
    content = response.content
    content = re.sub(r'<think>[\s\S]*?</think>', '', content)
    content = re.sub(r'<thought>[\s\S]*?</thought>', '', content)
    return content.strip()


def _call_ollama_vision(prompt: str, image_b64_list: list[str]) -> str:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img_b64 in image_b64_list:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })
    ollama_llm = ChatOpenAI(
        model=get_vision_model_name(),
        openai_api_key="ollama",
        base_url=get_ollama_base_url(),
        temperature=0.0,
    )
    response = ollama_llm.invoke([HumanMessage(content=content)])
    content_text = response.content
    content_text = re.sub(r'<think>[\s\S]*?</think>', '', content_text)
    content_text = re.sub(r'<thought>[\s\S]*?</thought>', '', content_text)
    return content_text.strip()


def classify_page_from_text(text: str) -> str:
    """
    Classify page type from its OCR text using Ollama deepseek-r1:14b.
    
    Clinical Safety:
        Returns UNKNOWN if classification fails or is not in taxonomy.
    """
    doc_types_str = "\n".join(f"- {dt}" for dt in DOC_TYPES)
    prompt = f"""You are a clinical document classifier. Classify the following page text into one of these taxonomy types:
{doc_types_str}

Page text snippet:
{text[:2000]}

Rules:
- Select exactly one document type from the taxonomy above.
- Return ONLY the classification type name (e.g. "DRUG_CHART" or "LAB_REPORT_BIOCHEMISTRY"), with no other explanation or markdown formatting.
- If it doesn't fit any type, return "UNKNOWN"."""
    
    try:
        content = _call_ollama_text(prompt)
        cleaned_type = content.strip().upper()
        # Find matches in DOC_TYPES
        for dt in DOC_TYPES:
            if dt in cleaned_type:
                return dt
        return "UNKNOWN"
    except Exception as e:
        print(f"[CLASSIFY] Warning: Failed to classify page from text: {e}")
        return "UNKNOWN"


def _heuristic_doc_type_from_text(text: str) -> Optional[str]:
    """Conservative text overrides for high-signal document anchors."""
    normalized = re.sub(r"\s+", " ", text.lower())
    if not normalized:
        return None

    discharge_signals = [
        "condition at discharge",
        "advice on discharge",
        "course in the hospital",
        "follow-up instructions",
        "review immediately in case of",
    ]
    if (
        "advice on discharge" in normalized
        or "condition at discharge" in normalized
        or ("diagnosis:" in normalized and "course in the hospital" in normalized)
        or ("course in the hospital" in normalized and "follow-up" in normalized)
    ):
        return "TYPED_DISCHARGE_SUMMARY"

    if "medication name" in normalized and "dosage" in normalized and "frequency" in normalized:
        return "TYPED_DISCHARGE_SUMMARY"

    if "urine culture" in normalized and "report awaited" in normalized:
        return "TYPED_DISCHARGE_SUMMARY"

    return None


def _read_cached_ocr_text(page_num: int) -> str:
    cache_file = get_ocr_cache_file(page_num)
    if not cache_file.exists():
        return ""
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    except Exception:
        return ""
    return text.replace(f"=== PAGE {page_num} ===", "").strip()


def _call_vision_llm(prompt: str, image_b64_list: list[str]) -> str:
    """
    Call Gemini Vision with text prompt and one or more page images.
    
    Returns the raw text response from the configured LLM backend.
    Uses Gemini, Ollama text reasoning, or Ollama vision depending on backend and inputs.
    """
    backend = get_backend()
    google_api_key = _env("GOOGLE_API_KEY", GOOGLE_API_KEY)
    is_dummy_key = not google_api_key or "your-google-api-key" in google_api_key
    
    if backend in {"auto", "gemini"} and not is_dummy_key:
        try:
            llm = get_llm()
            content: list[dict] = [{"type": "text", "text": prompt}]
            for img_b64 in image_b64_list:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })
            msg = HumanMessage(content=content)
            response = llm.invoke([msg])
            return response.content
        except Exception as e:
            print(f"\n[BACKUP] Gemini call failed: {e}. Checking local fallback...")
            if backend == "gemini":
                raise
    elif backend == "gemini":
        raise RuntimeError("LLM_BACKEND=gemini but GOOGLE_API_KEY is missing or dummy.")
    elif backend == "auto":
        print("\n[BACKUP] Gemini API key is missing or dummy. Checking local fallback...")

    try:
        if image_b64_list:
            print(f"[BACKUP] Using local vision model {get_vision_model_name()} via Ollama.")
            return _call_ollama_vision(prompt, image_b64_list)
        print(f"[BACKUP] Using local reasoning model {get_reasoning_model_name()} via Ollama.")
        return _call_ollama_text(prompt)
    except Exception as e:
        print(f"[BACKUP] Ollama call failed: {e}")
        raise e


def _parse_json_response(response_text: str) -> dict | list:
    """
    Extract JSON from LLM response, handling markdown code fences.
    
    Clinical Safety: Returns empty dict on parse failure — never invents data.
    """
    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        # Remove closing fence
        text = re.sub(r"\n?```\s*$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object/array in the text
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        return {}


# ─── TOOL: CLASSIFY DOCUMENT PAGES ──────────────────────────────────────────────

def classify_document_pages(
    page_images_b64: list[tuple[int, str]],
) -> list[dict]:
    """
    Classify multiple pages into the DOC_TYPES taxonomy in a single LLM call.

    Purpose:
        Batch-classifies pages to build the document inventory efficiently.
        This is the first step in the extraction pipeline.

    Clinical Safety:
        - Pages classified as UNKNOWN with confidence < 0.5 must be logged
          in unreadable_pages and mentioned in pending_results.
        - Confidence scores enable downstream quality gates.

    Args:
        page_images_b64: List of (page_num, base64_image) tuples.

    Returns:
        List of {page_num, doc_type, confidence} dicts.
    """
    doc_types_str = "\n".join(f"- {dt}" for dt in DOC_TYPES)
    page_nums = [p[0] for p in page_images_b64]

    prompt = f"""You are a clinical document classifier. Classify each page of this hospital patient record.

For each page image provided, identify the document type from this taxonomy:
{doc_types_str}

The pages are numbered: {page_nums}

Return a JSON array with one entry per page:
[
    {{"page_num": <int>, "doc_type": "<string from taxonomy>", "confidence": <float 0.0-1.0>}}
]

Rules:
- If a page is mostly illegible or blank, classify as "UNKNOWN" with low confidence.
- If a page contains multiple types, classify by the PRIMARY content type.
- Handwritten pages should still be classified by their form type (e.g., nursing notes, drug chart).
- Be specific: prefer "LAB_REPORT_BIOCHEMISTRY" over generic "LAB_REPORT" when distinguishable.

    Return ONLY the JSON array, no other text."""

    images = [img for _, img in page_images_b64]
    backend = get_backend()

    if backend == "local":
        classifications = []
        for page_num, img_b64 in page_images_b64:
            cached_text = _read_cached_ocr_text(page_num)
            heuristic_type = _heuristic_doc_type_from_text(cached_text)
            if heuristic_type:
                classifications.append({
                    "page_num": page_num,
                    "doc_type": heuristic_type,
                    "confidence": 0.95,
                })
                continue

            single_prompt = f"""You are a clinical document classifier. Classify this single page image from a hospital patient record.

The page number is: {page_num}

Choose exactly one document type from this taxonomy:
{doc_types_str}

Rules:
- If the page is mostly illegible or blank, classify as "UNKNOWN" with low confidence.
- If a page contains multiple types, classify by the PRIMARY content type.
- Handwritten pages should still be classified by their form type, such as NURSING_NOTES or DRUG_CHART.
- Medication administration sheets/treatment charts should be classified as DRUG_CHART.
- Be specific: prefer "LAB_REPORT_BIOCHEMISTRY" over generic lab types when distinguishable.

Return ONLY this JSON object:
{{"page_num": {page_num}, "doc_type": "<string from taxonomy>", "confidence": <float 0.0-1.0>}}"""
            try:
                response = _call_vision_llm(single_prompt, [img_b64])
                parsed = _parse_json_response(response)
                if isinstance(parsed, list) and parsed:
                    parsed = parsed[0]
                if isinstance(parsed, dict):
                    doc_type = str(parsed.get("doc_type", "UNKNOWN")).upper()
                    matched_type = next((dt for dt in DOC_TYPES if dt in doc_type), "UNKNOWN")
                    if page_num <= 3 or matched_type in {"DRUG_CHART", "NURSING_ASSESSMENT", "UNKNOWN"}:
                        ocr_text = cached_text
                        if len(ocr_text) < MIN_TEXT_LENGTH:
                            try:
                                ocr_result = extract_page_text(img_b64, page_num)
                                ocr_text = ocr_result.get("text", "").replace(f"=== PAGE {page_num} ===", "").strip()
                            except Exception:
                                ocr_text = ""
                        heuristic_type = _heuristic_doc_type_from_text(ocr_text)
                        if heuristic_type:
                            matched_type = heuristic_type
                    confidence = parsed.get("confidence", 0.0)
                    try:
                        confidence = float(confidence)
                    except (TypeError, ValueError):
                        confidence = 0.0
                    classifications.append({
                        "page_num": int(parsed.get("page_num") or page_num),
                        "doc_type": matched_type,
                        "confidence": max(0.0, min(1.0, confidence)),
                    })
                    continue
            except Exception as e:
                print(f"[CLASSIFY] Local vision classification failed for page {page_num}: {e}. Falling back to OCR text.")

            classifications.extend(_classify_pages_from_text([(page_num, img_b64)]))

        return classifications
    
    google_api_key = _env("GOOGLE_API_KEY", GOOGLE_API_KEY)
    is_dummy_key = not google_api_key or "your-google-api-key" in google_api_key
    
    if not is_dummy_key:
        try:
            response = _call_vision_llm(prompt, images)
            parsed = _parse_json_response(response)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed
        except Exception as e:
            print(f"[CLASSIFY] Gemini Vision classification failed: {e}. Falling back to text-based classification...")

    return _classify_pages_from_text(page_images_b64)


def _classify_pages_from_text(page_images_b64: list[tuple[int, str]]) -> list[dict]:
    """Fallback classifier based on OCR text and the configured reasoning model."""
    print(f"[CLASSIFY] Running text-based classification via Ollama {get_reasoning_model_name()}...")
    classifications = []
    for page_num, img_b64 in page_images_b64:
        cache_file = get_ocr_cache_file(page_num)
        page_text = ""
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                page_text = f.read().strip()
                
        header_pattern = f"=== PAGE {page_num} ==="
        clean_text = page_text.replace(header_pattern, "").strip()
        
        # If cache is missing or empty, try extracting via OCR
        if len(clean_text) < MIN_TEXT_LENGTH:
            try:
                ocr_result = extract_page_text(img_b64, page_num)
                page_text = ocr_result.get("text", "")
            except Exception:
                page_text = ""
                
        clean_text = page_text.replace(header_pattern, "").strip()
        if len(clean_text) >= MIN_TEXT_LENGTH:
            doc_type = _heuristic_doc_type_from_text(clean_text) or classify_page_from_text(clean_text)
            classifications.append({
                "page_num": page_num,
                "doc_type": doc_type,
                "confidence": 0.8,
            })
        else:
            classifications.append({
                "page_num": page_num,
                "doc_type": "UNKNOWN",
                "confidence": 0.0,
            })
            
    return classifications


# ─── TOOL: EXTRACT PAGE TEXT (OCR) ──────────────────────────────────────────────

def extract_page_text(image_b64: str, page_num: int) -> dict:
    """
    Extract all text from a single scanned page using the configured vision OCR backend.

    Purpose:
        Full OCR extraction of a page, including handwritten text.
        Returns both the extracted text and a confidence estimate.
        Loads from and saves to local text cache when possible.

    Clinical Safety:
        - Always attempts full extraction even when confidence is low.
        - Preserves ambiguous readings with [UNCLEAR: best_guess] annotations.
        - For drug names, emits [DRUG NAME PARTIALLY LEGIBLE: ...] annotations.
        - Never silently drops content.

    Returns:
        {"text": str, "confidence": float, "page_num": int}
        On failure: {"text": "", "confidence": 0.0, "page_num": page_num}
    """
    # 1. Check if local text cache exists and is populated
    cache_file = get_ocr_cache_file(page_num)
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                text = f.read().strip()
            header_pattern = f"=== PAGE {page_num} ==="
            clean_text = text.replace(header_pattern, "").strip()
            if len(clean_text) >= MIN_TEXT_LENGTH:
                print(f"  Loaded Page {page_num} OCR text from local cache {cache_file} ({len(clean_text)} chars)")
                return {
                    "text": text,
                    "confidence": 1.0,
                    "page_num": page_num,
                }
        except Exception as cache_err:
            print(f"[CACHE] Warning: Failed to read OCR cache for Page {page_num}: {cache_err}")

    prompt = f"""You are a clinical document OCR system. Extract ALL text from this scanned hospital page (Page {page_num}).

Rules:
1. Extract EVERY piece of text visible, including handwritten notes, printed text, stamps, and filled form fields.
2. Preserve the original structure (tables, columns, headers) as much as possible using plain text formatting.
3. For partially legible text, use: [UNCLEAR: your_best_guess]
4. For partially legible drug names, use: [DRUG NAME PARTIALLY LEGIBLE: "what_you_read" — likely actual_drug_name. Verify against drug chart.]
5. For dates, always try dd/mm/yy and dd/mm/yyyy format. Note any format ambiguity.
6. Do NOT skip any section even if it appears blank — note "[SECTION APPEARS BLANK]" instead.
7. For tables, preserve column alignment and headers.
8. Note any stamps, signatures, or checkmarks.

After the extracted text, add a final line:
EXTRACTION_CONFIDENCE: <0.0-1.0>

Where 1.0 = fully legible typed text, 0.5 = partially legible, 0.0 = completely illegible."""

    response = _call_ollama_vision(prompt, [image_b64])

    # Parse confidence from response
    confidence = 0.5  # Default
    conf_match = re.search(r'EXTRACTION_CONFIDENCE:\s*([\d.]+)', response)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except ValueError:
            pass

    # Remove confidence line from text
    text = re.sub(r'\nEXTRACTION_CONFIDENCE:.*$', '', response, flags=re.MULTILINE).strip()

    # 2. Write response to local text cache file
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(f"=== PAGE {page_num} ===\n\n{text}")
        print(f"  Saved Page {page_num} OCR text to local cache {cache_file}")
    except Exception as cache_err:
        print(f"[CACHE] Warning: Failed to save OCR cache for Page {page_num}: {cache_err}")

    return {
        "text": text,
        "confidence": confidence,
        "page_num": page_num,
    }


# ─── TOOL: EXTRACT TYPED DISCHARGE SUMMARY ──────────────────────────────────────

def extract_typed_summary(text: str, page_num: int) -> dict:
    """
    Parse the structured typed discharge summary block.

    Purpose:
        The typed discharge summary is the anchor document — it provides
        demographics, dates, and official diagnoses to cross-reference
        against all other documents.

    Clinical Safety:
        - Fields not found must be returned as None, never guessed.
        - Diagnoses must be extracted exactly as written — no paraphrasing.
        - This is the PRIMARY source for demographics and dates.

    Returns:
        {"demographics": dict, "diagnoses": dict, "medications": list,
         "follow_up": list, "condition_at_discharge": str}
    """
    prompt = f"""You are a clinical data extractor. Parse this typed discharge summary (Page {page_num}).

Extract the following into a JSON object:
{{
    "demographics": {{
        "name": "<patient name or null>",
        "age": "<age or null>",
        "gender": "<gender or null>",
        "mrn": "<MRN number or null>",
        "ip_no": "<IP number or null>",
        "blood_group": "<blood group or null>",
        "weight": "<weight or null>",
        "address": "<address or null>"
    }},
    "admission_date": "<dd/mm/yyyy or null>",
    "discharge_date": "<dd/mm/yyyy or null>",
    "diagnoses": {{
        "principal": ["<exact text>"],
        "secondary": ["<exact text>"],
        "provisional": ["<exact text>"],
        "final": ["<exact text>"]
    }},
    "chief_complaints": ["<exact text>"],
    "past_history": "<exact text or null>",
    "medications": [
        {{
            "name": "<drug name>",
            "dose": "<dose or null>",
            "route": "<route or null>",
            "frequency": "<frequency or null>"
        }}
    ],
    "follow_up": ["<instruction>"],
    "condition_at_discharge": "<exact text or null>",
    "allergies": ["<allergy or 'NOT KNOWN'>"]
}}

Rules:
- Extract EXACTLY what is written. Do not paraphrase or infer.
- If a field is not present, set it to null.
- For diagnoses, capture ALL diagnosis entries, distinguishing provisional from final.
- For medications, capture every single medication listed.
- Preserve original spelling and abbreviations.

SOURCE TEXT:
{text}

Return ONLY the JSON object."""

    response = _call_vision_llm(prompt, [])
    return _parse_json_response(response)


# ─── TOOL: EXTRACT CLINICAL DATA FROM ANY PAGE ─────────────────────────────────

def extract_clinical_data(
    text: str,
    page_num: int,
    doc_type: str,
    image_b64: Optional[str] = None,
) -> dict:
    """
    Extract structured clinical data from any classified page.

    Purpose:
        Generic extraction tool that adapts its prompt based on document type.
        Handles admission records, ER charts, ICU charts, consultation sheets,
        nursing notes, monitoring charts, and procedure charts.

    Clinical Safety:
        - Every diagnosis must include the source document type.
        - Medications must include dates and routes when visible.
        - Nursing note events must be timestamped.
        - Hospital course sentences must include source page references.

    Returns:
        Dict with extracted fields relevant to the document type.
    """
    type_specific_instructions = _get_extraction_instructions(doc_type)

    prompt = f"""You are a clinical data extractor. Extract structured data from this {doc_type} page (Page {page_num}).

{type_specific_instructions}

SOURCE TEXT:
{text}

Return a JSON object with the extracted data. Use null for fields not found.
Do NOT invent or infer any clinical facts not explicitly present in the text.
If text is partially legible, use [UNCLEAR: best_guess] annotations."""

    images = [image_b64] if image_b64 else []
    response = _call_vision_llm(prompt, images)
    result = _parse_json_response(response)
    if isinstance(result, dict):
        result["source_page"] = page_num
        result["source_type"] = doc_type
    return result


def _get_extraction_instructions(doc_type: str) -> str:
    """Get document-type-specific extraction instructions."""
    instructions = {
        "ADMISSION_RECORD": """Extract:
{
    "chief_complaints": ["<symptom>"],
    "history_of_present_illness": "<text>",
    "past_history": "<text or null>",
    "provisional_diagnosis": ["<diagnosis>"],
    "final_diagnosis": ["<diagnosis>"],
    "examination_findings": "<text or null>",
    "allergies": ["<allergy or 'NOT KNOWN'>"]
}""",
        "ER_OBSERVATION_CHART": """Extract:
{
    "arrival_time": "<time or null>",
    "er_diagnosis": ["<diagnosis>"],
    "chief_complaints": ["<symptom>"],
    "vitals": {"bp": "", "pulse": "", "temp": "", "spo2": "", "rr": ""},
    "treatments_given": [{"drug": "", "dose": "", "route": "", "time": ""}],
    "disposition": "<admitted/discharged/transferred or null>"
}""",
        "ICU_CHART": """Extract:
{
    "icu_diagnoses": ["<diagnosis as written on chart header>"],
    "medications": [{"name": "", "dose": "", "route": "", "frequency": "", "date": ""}],
    "vitals_timeline": [{"time": "", "bp": "", "pulse": "", "temp": "", "spo2": ""}],
    "iv_fluids": [{"fluid": "", "rate": "", "duration": ""}],
    "ventilator_settings": "<if applicable, null otherwise>",
    "io_balance": "<intake/output summary or null>"
}""",
        "CONSULTATION_SHEET": """Extract:
{
    "consultation_date": "<date or null>",
    "consulting_doctor": "<name or null>",
    "specialty": "<specialty or null>",
    "consultation_diagnosis": ["<diagnosis>"],
    "recommendations": ["<recommendation>"],
    "discharge_plan": "<text or null>"
}""",
        "DRUG_CHART": """Extract ALL medications visible on this drug chart page:
{
    "medications": [
        {
            "name": "<drug name>",
            "dose": "<dose>",
            "route": "<PO/IV/SC/IM/etc>",
            "frequency": "<OD/BD/TID/QID/SOS/etc>",
            "start_date": "<date or null>",
            "stop_date": "<date or null>",
            "dates_administered": ["<date1>", "<date2>"],
            "signatures_present": true/false
        }
    ],
    "chart_date_range": {"from": "<date>", "to": "<date>"}
}
IMPORTANT: Extract EVERY drug entry. Do not skip any row. Include PRN/SOS medications.
For column headers showing dates (D1, D2, D3...), map them to calendar dates if a start date is visible.""",
        "NURSING_NOTES": """Extract timestamped nursing events:
{
    "events": [
        {
            "timestamp": "<date and time>",
            "event_type": "<assessment/medication/procedure/observation/vital_sign>",
            "description": "<exact text of the note>",
            "medications_mentioned": ["<drug names mentioned>"],
            "vitals_mentioned": {"bp": "", "pulse": "", "temp": "", "spo2": ""}
        }
    ]
}
IMPORTANT: Preserve chronological order. Include ALL entries even if partially legible.""",
        "NURSING_ASSESSMENT": """Extract:
{
    "assessment_date": "<date>",
    "patient_condition": "<text>",
    "pain_score": "<score or null>",
    "fall_risk": "<score or null>",
    "pressure_ulcer_risk": "<score or null>",
    "diet": "<text or null>",
    "mobility": "<text or null>",
    "consciousness": "<text or null>"
}""",
        "MONITORING_CHART_DIABETES": """Extract ALL blood glucose readings and insulin doses:
{
    "readings": [
        {
            "date": "<date>",
            "time": "<time or period: BL/BB/BL/BD/HS/3AM>",
            "glucose_value": <number>,
            "glucose_unit": "mg/dL",
            "insulin_type": "<Regular/Lantus/Actrapid/etc or null>",
            "insulin_dose": "<dose or null>",
            "insulin_route": "<SC/IV or null>"
        }
    ]
}
CRITICAL: Extract EVERY reading. These are forensic evidence for DKA/uncontrolled DM.""",
        "MONITORING_CHART_VITALS": """Extract all vital sign readings:
{
    "readings": [
        {
            "date": "<date>",
            "time": "<time>",
            "bp_systolic": <number or null>,
            "bp_diastolic": <number or null>,
            "pulse": <number or null>,
            "temperature": <number or null>,
            "spo2": <number or null>,
            "respiratory_rate": <number or null>
        }
    ]
}""",
        "DISCHARGE_CHECKLIST": """Extract:
{
    "discharge_type": "<routine/against_medical_advice/on_request/null>",
    "pending_items": ["<item>"],
    "patient_education_done": true/false/null,
    "follow_up_appointments": ["<appointment>"],
    "discharge_condition": "<text or null>"
}""",
    }

    # Default for lab/imaging types
    default = """Extract all data present on this page into a structured JSON format.
Include all fields, values, dates, and reference ranges visible.
Use null for missing values. Never invent data."""

    return instructions.get(doc_type, default)


# ─── TOOL: EXTRACT LAB REPORT ──────────────────────────────────────────────────

def extract_lab_report(text: str, page_num: int, report_type: str) -> dict:
    """
    Parse any lab report page. Detect abnormals against reference range.

    Purpose:
        Extracts every lab result with value, unit, reference range, date,
        and abnormal flag. Critical for CR-3 audit.

    Clinical Safety:
        - Must flag: result outside reference range (abnormal=True).
        - Must note: "PENDING" or "AWAITED" results verbatim.
        - Must NOT: interpolate a result if the value cell is blank.
        - Critical labs (Na, K, glucose, creatinine, pH, HCO3, WBC) are
          individually flagged per spec.

    Returns:
        {"results": [{"test", "value", "unit", "ref_range", "date", "abnormal_flag"}]}
    """
    prompt = f"""You are a clinical lab result parser. Extract ALL lab results from this {report_type} report (Page {page_num}).

Return a JSON object:
{{
    "report_date": "<date or null>",
    "results": [
        {{
            "test_name": "<full test name>",
            "result_value": "<value as string, or 'PENDING' or 'AWAITED'>",
            "unit": "<unit or null>",
            "reference_range": "<range as string or null>",
            "date": "<date of test or null>",
            "abnormal_flag": true/false,
            "critically_abnormal": true/false
        }}
    ]
}}

Rules:
- Extract EVERY result row, even if the value appears normal.
- If a value is outside the reference range, set abnormal_flag=true.
- If a value is DANGEROUSLY outside the range (e.g., Na < 120, K > 6.5, glucose > 400, pH < 7.25), set critically_abnormal=true.
- If a value cell is blank or unreadable, set result_value to null. Do NOT guess.
- If the result says "PENDING", "AWAITED", or similar, record it exactly.
- Preserve original units and reference ranges exactly as printed.

SOURCE TEXT:
{text}

Return ONLY the JSON object."""

    response = _call_vision_llm(prompt, [])
    return _parse_json_response(response)


# ─── TOOL: EXTRACT DRUG CHART (BATCH) ──────────────────────────────────────────

def extract_drug_chart_batch(
    page_texts: list[tuple[int, str]],
    page_images: Optional[list[tuple[int, str]]] = None,
) -> dict:
    """
    Parse multiple drug chart pages in a single batch call.

    Purpose:
        Drug charts typically span 2-3 pages. Batch processing saves agent
        steps and allows cross-page medication timeline reconstruction.

    Clinical Safety:
        - Must extract EVERY medication entry. No silent drops.
        - Must map column headers (D1, D2, D3...) to calendar dates.
        - Must detect medications present in inpatient charts but absent
          from discharge — potential CR-1 violation.

    Returns:
        {"medications": [MedicationEntry-like dicts]}
    """
    combined_text = "\n\n".join(
        f"--- PAGE {pn} ---\n{text}" for pn, text in page_texts
    )

    prompt = f"""You are a clinical drug chart parser. Extract ALL medications from these drug chart pages.

Return a JSON object:
{{
    "medications": [
        {{
            "name": "<drug name>",
            "dose": "<dose or null>",
            "route": "<PO/IV/SC/IM/etc or null>",
            "frequency": "<OD/BD/TID/QID/SOS/etc or null>",
            "start_date": "<date or null>",
            "stop_date": "<date or null>",
            "status": "<ADMISSION|INPATIENT_ONLY|DISCHARGE|UNKNOWN>",
            "change_reason": "<reason for starting/stopping or null>",
            "change_reason_documented": true/false,
            "source_pages": [<page numbers>]
        }}
    ],
    "chart_date_range": {{"from": "<date>", "to": "<date>"}},
    "notes": "<any important notes about the drug chart>"
}}

Rules:
- Extract EVERY medication. Do not skip any row, even PRN/SOS.
- Map D1/D2/D3 columns to actual calendar dates using any date hints visible.
- If a drug has tick marks on certain days but not others, note the specific dates.
- For partially legible drug names: [DRUG NAME PARTIALLY LEGIBLE: "what_you_read" — likely actual_name]
- Set status to INPATIENT_ONLY if the drug only appears during hospital stay.
- Set change_reason_documented to false if no reason is written for starting or stopping a drug.

SOURCE TEXT:
{combined_text}

Return ONLY the JSON object."""

    images = [img for _, img in (page_images or [])]
    response = _call_vision_llm(prompt, images)
    return _parse_json_response(response)


# ─── TOOL: RECONCILE MEDICATIONS ────────────────────────────────────────────────

def reconcile_medications(
    admission: list[dict],
    inpatient: list[dict],
    discharge: list[dict],
) -> dict:
    """
    Compare admission vs inpatient vs discharge medications. Flag all changes.

    Purpose:
        Medication reconciliation is a critical safety check. Every change
        (added, stopped, dose changed) must be surfaced with a reason
        or flagged for clinician review.

    Clinical Safety:
        - A medication in inpatient but absent from discharge MUST be flagged
          as potentially omitted (CR-1 check).
        - A medication added without documented reason MUST be flagged.
        - A medication stopped without documented reason MUST be flagged.
        - This function NEVER resolves discrepancies — it reports them.

    Returns:
        {"reconciliation": [dict], "flags": [ClinicalFlag-like dicts]}
    """
    reconciliation: list[dict] = []
    flags: list[dict] = []

    # Normalize medication names for comparison
    def normalize(name: str) -> str:
        return name.strip().upper().replace("INJ ", "").replace("TAB ", "").replace("CAP ", "")

    adm_names = {normalize(m.get("name", "")): m for m in admission}
    inp_names = {normalize(m.get("name", "")): m for m in inpatient}
    dis_names = {normalize(m.get("name", "")): m for m in discharge}

    all_drugs = set(adm_names.keys()) | set(inp_names.keys()) | set(dis_names.keys())

    for drug in sorted(all_drugs):
        in_adm = drug in adm_names
        in_inp = drug in inp_names
        in_dis = drug in dis_names

        entry = {
            "drug": drug,
            "in_admission": in_adm,
            "in_inpatient": in_inp,
            "in_discharge": in_dis,
            "change_type": None,
            "documented_reason": None,
            "flag": None,
        }

        if in_inp and not in_dis:
            entry["change_type"] = "STOPPED_AT_DISCHARGE"
            med = inp_names[drug]
            reason = med.get("change_reason")
            entry["documented_reason"] = reason
            if not reason:
                entry["flag"] = "MEDICATION_STOPPED_NO_REASON"
                flags.append({
                    "field": f"medication_reconciliation.{drug}",
                    "severity": "WARNING",
                    "reason": f"{drug} was active during admission but absent from discharge medications. No documented reason for discontinuation.",
                    "source_page": None,
                    "requires_clinician": True,
                })

        elif in_inp and not in_adm:
            entry["change_type"] = "ADDED_DURING_STAY"
            med = inp_names[drug]
            reason = med.get("change_reason")
            entry["documented_reason"] = reason
            if not reason:
                entry["flag"] = "MEDICATION_ADDED_NO_REASON"
                flags.append({
                    "field": f"medication_reconciliation.{drug}",
                    "severity": "WARNING",
                    "reason": f"{drug} was added during hospital stay. No documented reason for initiation.",
                    "source_page": None,
                    "requires_clinician": True,
                })

        elif in_adm and in_dis:
            # Check for dose changes
            adm_dose = adm_names[drug].get("dose", "")
            dis_dose = dis_names[drug].get("dose", "")
            if adm_dose and dis_dose and adm_dose != dis_dose:
                entry["change_type"] = "DOSE_CHANGED"
                entry["flag"] = "DOSE_CHANGE"
                flags.append({
                    "field": f"medication_reconciliation.{drug}",
                    "severity": "WARNING",
                    "reason": f"{drug} dose changed from {adm_dose} to {dis_dose}. Verify reason.",
                    "source_page": None,
                    "requires_clinician": True,
                })
            else:
                entry["change_type"] = "CONTINUED"

        elif in_dis and not in_adm and not in_inp:
            entry["change_type"] = "NEW_AT_DISCHARGE"
            entry["flag"] = "NEW_MEDICATION_AT_DISCHARGE"
            flags.append({
                "field": f"medication_reconciliation.{drug}",
                "severity": "WARNING",
                "reason": f"{drug} appears in discharge medications but was not documented during admission or inpatient stay.",
                "source_page": None,
                "requires_clinician": True,
            })

        reconciliation.append(entry)

    return {"reconciliation": reconciliation, "flags": flags}


# ─── TOOL: DRUG INTERACTION LOOKUP (MOCKED) ─────────────────────────────────────

def lookup_drug_interactions(medications: list[str]) -> dict:
    """
    Check for known drug-drug interactions. [MOCKED — not a real pharmacopeia call]

    Purpose:
        In production, this would call a real drug interaction database.
        The mock returns plausible flagged pairs for common clinical combos.

    Clinical Safety:
        Clearly labeled as MOCKED in all outputs. Never presented as
        authoritative pharmacological advice.

    Returns:
        {"interactions": [dict], "mocked": True}
    """
    interactions: list[dict] = []
    med_upper = [m.upper() for m in medications]

    # Known interaction patterns (simplified mock)
    known_pairs = [
        ({"MEROPENEM", "VALPROIC ACID"}, "Meropenem may reduce valproic acid levels", "CRITICAL"),
        ({"METFORMIN", "CONTRAST"}, "Hold metformin before/after IV contrast", "WARNING"),
        ({"INSULIN", "METFORMIN"}, "Monitor for hypoglycemia with dual therapy", "WARNING"),
        ({"HEPARIN", "ASPIRIN"}, "Increased bleeding risk", "WARNING"),
    ]

    for pair, description, severity in known_pairs:
        if pair.issubset(set(med_upper)):
            interactions.append({
                "drugs": list(pair),
                "description": description,
                "severity": severity,
                "source": "[MOCKED — not a real pharmacopeia call]",
            })

    return {
        "interactions": interactions,
        "mocked": True,
        "note": "[MOCKED — This is not a real pharmacopeia lookup. In production, integrate with a clinical drug interaction database.]"
    }


# ─── SAFE TOOL CALL WRAPPER ─────────────────────────────────────────────────────

def safe_tool_call(
    tool_fn: Callable,
    inputs: dict,
    state: dict,
    tool_name: str,
    target_field: str,
    step_number: int,
) -> tuple[Optional[Any], dict]:
    """
    Wrap every tool call with retry logic and failure handling.

    Purpose:
        Ensures no tool failure crashes the agent. Failed tools result in
        [UNRESOLVED] markers, never fabricated data.

    Clinical Safety:
        - Max 2 retries (3 total attempts) before giving up.
        - On final failure, marks the target field as [UNRESOLVED].
        - Logs every attempt in the trace, including failures.
        - NEVER guesses a value on tool failure.

    Returns:
        (result_or_None, updated_state)
    """
    max_retries = MAX_RETRIES

    for attempt in range(max_retries + 1):
        try:
            result = tool_fn(**inputs)

            # Validate result is not empty/None
            if not result or (isinstance(result, dict) and all(v is None for v in result.values())):
                raise ValueError("Tool returned empty result")

            emit_trace(
                state=state,
                step_number=step_number,
                phase="CALL_TOOL",
                reasoning=f"Calling {tool_name} for {target_field}",
                action="TOOL_CALL",
                tool_name=tool_name,
                tool_inputs=inputs,
                tool_output_summary=f"Success on attempt {attempt + 1}",
                observation=f"Tool {tool_name} returned valid data",
                decision=f"Proceed with extracted data for {target_field}",
                fields_updated=[target_field],
            )
            return result, state

        except Exception as e:
            state["retry_counts"][tool_name] = state.get("retry_counts", {}).get(tool_name, 0) + 1

            emit_trace(
                state=state,
                step_number=step_number,
                phase="CALL_TOOL",
                reasoning=f"Tool {tool_name} failed on attempt {attempt + 1}",
                action="RETRY" if attempt < max_retries else "GIVE_UP",
                tool_name=tool_name,
                tool_inputs=inputs,
                tool_output_summary=f"FAILED: {str(e)[:200]}",
                observation=f"Attempt {attempt + 1}/{max_retries + 1} failed",
                decision="Retry" if attempt < max_retries else "Mark as UNRESOLVED",
                fields_updated=[],
                fallback_taken=True,
                fallback_reason=f"Attempt {attempt + 1} failed: {str(e)[:200]}",
            )

            if attempt == max_retries:
                # Mark field as unresolved — DO NOT guess
                state["fabrication_blocks"].append(
                    f"{target_field}: Tool {tool_name} failed after {max_retries + 1} attempts. "
                    f"Marked [UNRESOLVED]."
                )
                return None, state

    return None, state
