"""
edit_signal.py — Edit Signal Engine
=====================================

Computes a scalar reward signal from (draft, edited) pairs that the
contextual bandit can optimize. The signal is clinically meaningful,
not just surface string similarity.

Clinical Safety:
    - R_SAFE is a hard binary check: was any escalation flag silently dropped?
    - If R_SAFE = 0.0, the composite reward is clamped to max(R, 0.10).
    - A draft that drops an escalation flag can NEVER earn a high reward.
    - The safety clamp overrides all other sub-signals.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import re


# ─── CONSTANTS ──────────────────────────────────────────────────────────────────

# Weight for each sub-signal in the composite reward.
# Equal emphasis on edit distance and section accuracy prevents gaming via shorter text.
W_SED: float = 0.35   # Normalized edit distance weight
W_SEC: float = 0.35   # Section-level match rate weight
W_PEND: float = 0.15  # Pending results coverage weight
W_SAFE: float = 0.15  # Safety preservation weight

# Maximum composite reward when safety is violated (R_SAFE = 0.0).
# A draft that drops an escalation flag can never earn more than this.
SAFETY_CLAMP_CEILING: float = 0.10

# The 7 sections that count toward R_SEC (section-level match rate).
# These are the clinically highest-value sections.
SCORED_SECTIONS: list[str] = [
    "principal_diagnosis",
    "discharge_medications",
    "medication_changes",
    "pending_results",
    "hospital_course",
    "discharge_condition",
    "escalation_flags_for_clinician",
]

# Mapping from scored section name to the Markdown heading used in compiled summaries.
# Used by the section parser to identify each section's boundaries.
SECTION_HEADING_MAP: dict[str, str] = {
    "principal_diagnosis": "## 4. Principal Diagnosis",
    "discharge_medications": "## 11. Discharge Medications",
    "medication_changes": "## 12. Medication Changes",
    "pending_results": "## 13. Pending Results",
    "hospital_course": "## 7. Hospital Course",
    "discharge_condition": "## 15. Discharge Condition",
    "escalation_flags_for_clinician": "## 16.",
}


# ─── DATA CONTRACT ──────────────────────────────────────────────────────────────

@dataclass
class EditSignal:
    """
    The reward signal computed from comparing a draft against an edited version.

    Each sub-signal measures a clinically distinct quality dimension.
    The composite_reward is what the bandit optimizes.

    Clinical Significance:
        - r_sed: Lower edit burden → the agent got closer to what the doctor wants
        - r_sec: Per-section accuracy on the 7 most critical sections
        - r_pend: Coverage of pending results (missed pending items = patient safety risk)
        - r_safe: Binary check that no escalation flag was dropped (non-negotiable)
    """
    draft_id: str                       # Unique ID of the draft being scored
    r_sed: float                        # Normalized edit distance component (0.0–1.0)
    r_sec: float                        # Section match rate component (0.0–1.0)
    r_pend: float                       # Pending results coverage (0.0–1.0)
    r_safe: float                       # Safety flag preservation: 0.0 or 1.0
    composite_reward: float             # Weighted composite: the bandit's reward signal
    section_scores: dict[str, float]    # Per-section match rates for all SCORED_SECTIONS
    weights_used: dict[str, float]      # Record of weights used (for audit reproducibility)
    safety_clamped: bool                # True if composite was clamped due to R_SAFE = 0


# ─── LEVENSHTEIN DISTANCE (PURE PYTHON) ─────────────────────────────────────────

def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.

    Purpose:
        Measures the minimum number of single-character edits (insertions,
        deletions, substitutions) to transform s1 into s2.

    Args:
        s1: Source string (draft).
        s2: Target string (edited).

    Returns:
        Integer edit distance.

    Safety Constraint:
        Uses O(min(len(s1), len(s2))) memory via the single-row optimization.
        Will not OOM on large clinical summaries.

    Failure Behavior:
        Pure computation — cannot fail unless memory exhausted.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# ─── SECTION PARSER ─────────────────────────────────────────────────────────────

def parse_summary_sections(markdown_text: str) -> dict[str, str]:
    """
    Split a compiled Markdown discharge summary into named sections.

    Purpose:
        Extracts each section's content by finding ## headings and capturing
        all text until the next ## heading or end of document.

    Args:
        markdown_text: Full Markdown text of a discharge summary.

    Returns:
        Dict mapping section name (from SCORED_SECTIONS) to section content string.

    Safety Constraint:
        Sections not found in the text are returned as empty strings.
        Never fabricates section content.

    Failure Behavior:
        Returns empty dict if markdown_text is empty or unparseable.
    """
    if not markdown_text:
        return {}

    sections: dict[str, str] = {}
    lines = markdown_text.split("\n")

    for section_name, heading_prefix in SECTION_HEADING_MAP.items():
        section_content: list[str] = []
        in_section = False

        for line in lines:
            if line.strip().startswith(heading_prefix):
                in_section = True
                continue
            elif in_section and line.strip().startswith("## "):
                # Hit the next section heading — stop capturing
                break
            elif in_section:
                section_content.append(line)

        # Strip leading/trailing blank lines from section content
        content = "\n".join(section_content).strip()
        sections[section_name] = content

    return sections


def _extract_escalation_flags(markdown_text: str) -> list[str]:
    """
    Extract all escalation flag descriptions from a discharge summary.

    Purpose:
        Identifies each distinct escalation flag in the summary to enable
        the R_SAFE check (did the draft preserve all flags from the edited version?).

    Args:
        markdown_text: Full Markdown text of a discharge summary.

    Returns:
        List of escalation flag field names found in the text.

    Safety Constraint:
        Must find flags regardless of formatting variations.

    Failure Behavior:
        Returns empty list if no flags found or text is empty.
    """
    flags: list[str] = []
    # Look for patterns like: N. **field_name** — reason
    pattern = r'\d+\.\s+\*\*([^*]+)\*\*\s*—'
    matches = re.findall(pattern, markdown_text)
    for match in matches:
        flags.append(match.strip())
    return flags


def _extract_pending_results(markdown_text: str) -> list[str]:
    """
    Extract pending result items from a discharge summary.

    Purpose:
        Identifies each pending result line to enable the R_PEND check.

    Args:
        markdown_text: Full Markdown text of a discharge summary.

    Returns:
        List of pending result strings.

    Safety Constraint:
        Must capture all pending items, including those with ⏳ prefix.

    Failure Behavior:
        Returns empty list if no pending results found.
    """
    results: list[str] = []
    # Match lines starting with "- ⏳" or "- " in the pending results section
    sections = parse_summary_sections(markdown_text)
    pending_text = sections.get("pending_results", "")
    for line in pending_text.split("\n"):
        line = line.strip()
        if line.startswith("- ⏳"):
            results.append(line.replace("- ⏳ ", "").strip())
        elif line.startswith("- "):
            results.append(line.replace("- ", "", 1).strip())
        elif line and not line.startswith("No pending"):
            results.append(line)
    return results


# ─── SECTION MATCH RATE ─────────────────────────────────────────────────────────

def _compute_section_match(draft_section: str, edited_section: str) -> float:
    """
    Compute match rate for a single section.

    Purpose:
        Returns 1.0 if sections are identical, or a fractional score based on
        normalized edit distance for partially matching sections.

    Args:
        draft_section: Section text from the agent's draft.
        edited_section: Section text from the reviewer-edited version.

    Returns:
        Float between 0.0 and 1.0. 1.0 = identical, 0.0 = completely different.

    Safety Constraint:
        Empty sections in both draft and edited → 1.0 (both correctly empty).
        Empty in draft but non-empty in edited → 0.0 (draft missed content).

    Failure Behavior:
        Cannot fail — pure string comparison.
    """
    if not draft_section and not edited_section:
        return 1.0
    if not draft_section or not edited_section:
        return 0.0

    # Normalize whitespace for comparison
    draft_norm = " ".join(draft_section.split())
    edited_norm = " ".join(edited_section.split())

    if draft_norm == edited_norm:
        return 1.0

    distance = _levenshtein_distance(draft_norm, edited_norm)
    max_len = max(len(draft_norm), len(edited_norm))
    return 1.0 - (distance / max_len) if max_len > 0 else 1.0


# ─── MAIN COMPUTATION ───────────────────────────────────────────────────────────

def compute_edit_signal(
    draft_id: str,
    draft_text: str,
    edited_text: str,
) -> EditSignal:
    """
    Compute the full edit signal from a (draft, edited) pair.

    Purpose:
        Produces the scalar reward signal that the contextual bandit optimizes.
        Combines 4 clinically meaningful sub-signals into a weighted composite.

    Args:
        draft_id: Unique identifier for this draft.
        draft_text: Full Markdown text of the agent's draft.
        edited_text: Full Markdown text after reviewer edits.

    Returns:
        EditSignal dataclass with all sub-signals and the composite reward.

    Safety Constraint:
        If R_SAFE = 0.0 (escalation flag dropped), the composite reward is
        clamped to max(R, 0.10) regardless of other sub-scores.

    Failure Behavior:
        If parsing fails for any section, that section scores 0.0.
        The function always returns a valid EditSignal — never raises.
    """
    try:
        # ─── R_SED: Normalized Edit Distance ────────────────────────────────
        draft_norm = " ".join(draft_text.split())
        edited_norm = " ".join(edited_text.split())

        if max(len(draft_norm), len(edited_norm)) > 0:
            distance = _levenshtein_distance(draft_norm, edited_norm)
            r_sed = 1.0 - (distance / max(len(draft_norm), len(edited_norm)))
        else:
            r_sed = 1.0  # Both empty → perfect match

        r_sed = max(0.0, min(1.0, r_sed))

        # ─── R_SEC: Section-Level Match Rate ────────────────────────────────
        draft_sections = parse_summary_sections(draft_text)
        edited_sections = parse_summary_sections(edited_text)

        section_scores: dict[str, float] = {}
        for section_name in SCORED_SECTIONS:
            draft_sec = draft_sections.get(section_name, "")
            edited_sec = edited_sections.get(section_name, "")
            section_scores[section_name] = _compute_section_match(draft_sec, edited_sec)

        r_sec = sum(section_scores.values()) / len(SCORED_SECTIONS) if SCORED_SECTIONS else 0.0

        # ─── R_PEND: Pending Results Coverage ───────────────────────────────
        draft_pending = set(_extract_pending_results(draft_text))
        edited_pending = set(_extract_pending_results(edited_text))

        if edited_pending:
            r_pend = len(draft_pending & edited_pending) / len(edited_pending)
        else:
            # No pending results in edited version → agent correctly had none
            r_pend = 1.0 if not draft_pending else 0.0

        r_pend = max(0.0, min(1.0, r_pend))

        # ─── R_SAFE: Safety Flag Preservation ──────────────────────────────
        draft_flags = set(_extract_escalation_flags(draft_text))
        edited_flags = set(_extract_escalation_flags(edited_text))

        if edited_flags:
            # All flags from the edited version must be present in the draft
            r_safe = 1.0 if edited_flags.issubset(draft_flags) else 0.0
        else:
            # No flags in edited → agent shouldn't have any either
            r_safe = 1.0

        # ─── COMPOSITE REWARD ───────────────────────────────────────────────
        composite = (
            W_SED * r_sed +
            W_SEC * r_sec +
            W_PEND * r_pend +
            W_SAFE * r_safe
        )

        # ─── SAFETY CLAMP ──────────────────────────────────────────────────
        safety_clamped = False
        if r_safe == 0.0:
            composite = min(composite, SAFETY_CLAMP_CEILING)
            safety_clamped = True

        composite = max(0.0, min(1.0, composite))

        weights_used = {
            "w_sed": W_SED,
            "w_sec": W_SEC,
            "w_pend": W_PEND,
            "w_safe": W_SAFE,
        }

        return EditSignal(
            draft_id=draft_id,
            r_sed=r_sed,
            r_sec=r_sec,
            r_pend=r_pend,
            r_safe=r_safe,
            composite_reward=composite,
            section_scores=section_scores,
            weights_used=weights_used,
            safety_clamped=safety_clamped,
        )

    except Exception as e:
        print(f"[EDIT_SIGNAL] ERROR: {e}")
        # Return a minimal valid signal — never crash
        return EditSignal(
            draft_id=draft_id,
            r_sed=0.0,
            r_sec=0.0,
            r_pend=0.0,
            r_safe=0.0,
            composite_reward=0.0,
            section_scores={s: 0.0 for s in SCORED_SECTIONS},
            weights_used={"w_sed": W_SED, "w_sec": W_SEC, "w_pend": W_PEND, "w_safe": W_SAFE},
            safety_clamped=True,
        )
