"""
correction_memory.py — Correction Memory Bank
================================================

A persistent, queryable store of extracted correction patterns. When the
agent generates a new draft, the top-k most relevant corrections are
injected into the compilation prompt as few-shot examples.

This is the primary mechanism for prompt-based learning without fine-tuning.

Clinical Safety:
    - All writes are atomic (write to .tmp, then os.replace()).
    - Patterns are human-readable JSONL for audit purposes.
    - Corrections never modify the underlying clinical data — they only
      inform future prompt construction.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ─── DEFAULT STORAGE PATH ────────────────────────────────────────────────────────

# The JSONL file where correction patterns are persisted.
DEFAULT_MEMORY_PATH: str = "output/correction_memory.jsonl"


# ─── DATA MODEL ─────────────────────────────────────────────────────────────────

@dataclass
class CorrectionPattern:
    """
    A single correction pattern extracted from a (draft, edited) pair.

    Clinical Significance:
        - source_section: Which summary section this correction targets
        - clinical_category: The clinical domain (diagnosis, medication, safety, etc.)
        - before_snippet/after_snippet: The exact text change for few-shot injection
        - frequency: How often this pattern has appeared across training iterations
        - reward_delta: How much the composite reward improved when this pattern was injected
    """
    pattern_id: str           # uuid4 — unique identifier for this pattern
    source_section: str       # Which summary section this correction belongs to
    rule_origin: str          # e.g. "REV-001", "PHASE2_LLM", "BANDIT_ARM_2"
    before_snippet: str       # The incorrect/incomplete text (max 500 chars)
    after_snippet: str        # The corrected text (max 500 chars)
    clinical_category: str    # "diagnosis", "medication", "pending_result", "safety_flag", "citation"
    frequency: int            # How many times this pattern type has appeared
    reward_delta: float       # Mean R improvement observed after injecting this pattern
    last_seen_iso: str        # ISO-8601 timestamp of last occurrence


# ─── CORRECTION MEMORY BANK ─────────────────────────────────────────────────────

class CorrectionMemoryBank:
    """
    Persistent store of correction patterns for few-shot prompt injection.

    Purpose:
        Stores, indexes, and retrieves correction patterns. When the agent
        generates a new draft, the top-k most relevant corrections are
        injected into the prompt as few-shot examples.

    Safety Constraint:
        All writes are atomic. The JSONL file is human-readable for audit.
        Patterns never modify clinical data — they only inform prompts.

    Failure Behavior:
        On load failure, starts with empty memory.
        On save failure, logs error but does not crash.
    """

    def __init__(self, memory_path: str = DEFAULT_MEMORY_PATH):
        """
        Initialize the correction memory bank.

        Args:
            memory_path: Path to the JSONL persistence file.
        """
        self.memory_path = memory_path
        self.patterns: list[CorrectionPattern] = []
        self._load()

    def _load(self) -> None:
        """
        Load all patterns from the JSONL file into memory.

        Purpose:
            Restores patterns from disk on startup.

        Safety Constraint:
            Corrupt lines are skipped, not crash-inducing.

        Failure Behavior:
            If file doesn't exist, starts with empty list.
            If a line is corrupt, it is skipped with a warning.
        """
        if not os.path.exists(self.memory_path):
            return

        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        pattern = CorrectionPattern(**data)
                        self.patterns.append(pattern)
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"[CORRECTION_MEMORY] WARNING: Skipping corrupt line {line_num}: {e}")
        except Exception as e:
            print(f"[CORRECTION_MEMORY] ERROR: Failed to load from {self.memory_path}: {e}")

    def save(self) -> None:
        """
        Persist all patterns to the JSONL file (atomic write).

        Purpose:
            Writes the complete pattern list to disk using a tmp file
            and os.replace() to ensure atomicity.

        Safety Constraint:
            Never produces a partial write. Either all patterns are saved
            or the previous file is preserved.

        Failure Behavior:
            On failure, logs error. The previous file remains intact.
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.memory_path) or ".", exist_ok=True)

            tmp_path = self.memory_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                for pattern in self.patterns:
                    json_line = json.dumps(asdict(pattern), ensure_ascii=False)
                    f.write(json_line + "\n")
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, self.memory_path)

        except Exception as e:
            print(f"[CORRECTION_MEMORY] ERROR: Failed to save to {self.memory_path}: {e}")

    def add_pattern(self, pattern: CorrectionPattern) -> None:
        """
        Add a correction pattern to the memory bank.

        Purpose:
            Inserts a new pattern or increments frequency if a similar
            pattern already exists (deduplication by section + category + snippet hash).

        Args:
            pattern: The CorrectionPattern to add.

        Safety Constraint:
            Snippets are truncated to 500 chars to prevent memory bloat.

        Failure Behavior:
            Never raises. Logs errors and continues.
        """
        # Truncate snippets
        pattern.before_snippet = pattern.before_snippet[:500]
        pattern.after_snippet = pattern.after_snippet[:500]

        # Check for existing similar pattern (deduplicate)
        for existing in self.patterns:
            if (existing.source_section == pattern.source_section and
                    existing.clinical_category == pattern.clinical_category and
                    existing.before_snippet == pattern.before_snippet):
                # Update existing pattern
                existing.frequency += 1
                existing.reward_delta = (
                    existing.reward_delta * (existing.frequency - 1) + pattern.reward_delta
                ) / existing.frequency
                existing.last_seen_iso = pattern.last_seen_iso
                return

        self.patterns.append(pattern)

    def update_reward_delta(self, pattern_id: str, reward_delta: float) -> None:
        """
        Update the reward delta for a specific pattern.

        Purpose:
            Called after a training iteration to record how much reward
            improvement was observed when this pattern was injected.

        Args:
            pattern_id: UUID of the pattern to update.
            reward_delta: The observed improvement in composite reward.

        Safety Constraint:
            Only updates reward_delta — never modifies the pattern's clinical content.

        Failure Behavior:
            If pattern_id not found, logs warning.
        """
        for pattern in self.patterns:
            if pattern.pattern_id == pattern_id:
                # Running average
                pattern.reward_delta = (
                    pattern.reward_delta * max(pattern.frequency - 1, 0) + reward_delta
                ) / max(pattern.frequency, 1)
                return

        print(f"[CORRECTION_MEMORY] WARNING: Pattern {pattern_id} not found for reward update")

    def get_relevant_corrections(
        self,
        section_name: str,
        draft_snippet: str = "",
        top_k: int = 3,
    ) -> list[CorrectionPattern]:
        """
        Retrieve the top-k most relevant corrections for a given section.

        Purpose:
            Two-stage retrieval for few-shot injection into the compiler prompt.

        Args:
            section_name: The section to retrieve corrections for.
            draft_snippet: Optional text snippet for contextual matching (unused in v1).
            top_k: Maximum number of corrections to return.

        Returns:
            List of CorrectionPattern sorted by relevance score descending.

        Safety Constraint:
            Only returns patterns matching the requested section.
            Never returns patterns from unrelated sections.

        Failure Behavior:
            Returns empty list if no matching patterns found.
        """
        # Stage 1: Hard filter by section
        candidates = [
            p for p in self.patterns
            if p.source_section == section_name
        ]

        if not candidates:
            return []

        # Stage 2: Soft rank by reward_delta * log(1 + frequency)
        scored = [
            (p, p.reward_delta * math.log(1 + p.frequency))
            for p in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [p for p, score in scored[:top_k]]

    def format_correction_examples(self, patterns: list[CorrectionPattern]) -> str:
        """
        Format correction patterns as few-shot examples for prompt injection.

        Purpose:
            Produces the exact text block that is injected into the compiler's
            LLM prompt to guide future draft generation.

        Args:
            patterns: List of CorrectionPattern to format.

        Returns:
            Formatted string block with BEFORE/AFTER examples.

        Safety Constraint:
            Output is plain text — never executable code.

        Failure Behavior:
            Returns empty string if patterns list is empty.
        """
        if not patterns:
            return ""

        blocks: list[str] = []
        for pattern in patterns:
            block = (
                f"CORRECTION EXAMPLE (section: {pattern.source_section}, "
                f"category: {pattern.clinical_category}):\n"
                f"BEFORE: {pattern.before_snippet}\n"
                f"AFTER: {pattern.after_snippet}"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    def extract_patterns_from_edit(
        self,
        edited_draft,
        reward_delta: float = 0.0,
    ) -> list[CorrectionPattern]:
        """
        Extract correction patterns from an EditedDraft.

        Purpose:
            Converts the reviewer's edits into reusable correction patterns
            that can be injected into future compilation prompts.

        Args:
            edited_draft: An EditedDraft from the SimulatedReviewer.
            reward_delta: The reward improvement observed for this edit.

        Returns:
            List of newly extracted CorrectionPattern objects.

        Safety Constraint:
            Only extracts patterns from validated edits (not fabricated ones).

        Failure Behavior:
            Returns empty list if extraction fails.
        """
        new_patterns: list[CorrectionPattern] = []
        now = datetime.now(timezone.utc).isoformat()

        # Extract from Phase 1 rules
        for rule_id in edited_draft.phase1_rules_applied:
            diff = edited_draft.diff_by_section.get(rule_id, {})
            category = _rule_to_category(rule_id)

            pattern = CorrectionPattern(
                pattern_id=str(uuid.uuid4()),
                source_section=_rule_to_section(rule_id),
                rule_origin=rule_id,
                before_snippet=diff.get("original", "")[:500],
                after_snippet=diff.get("edited", "")[:500],
                clinical_category=category,
                frequency=1,
                reward_delta=reward_delta,
                last_seen_iso=now,
            )
            new_patterns.append(pattern)
            self.add_pattern(pattern)

        # Extract from Phase 2 LLM corrections
        for correction in edited_draft.phase2_corrections:
            pattern = CorrectionPattern(
                pattern_id=str(uuid.uuid4()),
                source_section=correction.get("section", "unknown"),
                rule_origin="PHASE2_LLM",
                before_snippet=f"[Original text for {correction.get('section', 'unknown')}]",
                after_snippet=correction.get("change", "")[:500],
                clinical_category=_section_to_category(correction.get("section", "")),
                frequency=1,
                reward_delta=reward_delta,
                last_seen_iso=now,
            )
            new_patterns.append(pattern)
            self.add_pattern(pattern)

        return new_patterns

    def get_top_patterns(self, n: int = 10) -> list[CorrectionPattern]:
        """
        Get the top-N most impactful patterns by reward_delta * frequency.

        Purpose:
            Used for the correction_memory_summary.md output artifact.

        Args:
            n: Number of top patterns to return.

        Returns:
            List of patterns sorted by impact score descending.

        Safety Constraint:
            Read-only operation.

        Failure Behavior:
            Returns empty list if no patterns exist.
        """
        scored = [
            (p, p.reward_delta * p.frequency)
            for p in self.patterns
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, score in scored[:n]]


# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────────

def _rule_to_category(rule_id: str) -> str:
    """Map a reviewer rule ID to a clinical category."""
    mapping = {
        "REV-001": "diagnosis",
        "REV-002": "medication",
        "REV-003": "pending_result",
        "REV-004": "safety_flag",
        "REV-005": "citation",
        "REV-006": "safety_flag",
        "REV-007": "safety_flag",
    }
    return mapping.get(rule_id, "other")


def _rule_to_section(rule_id: str) -> str:
    """Map a reviewer rule ID to the target summary section."""
    mapping = {
        "REV-001": "principal_diagnosis",
        "REV-002": "medication_changes",
        "REV-003": "pending_results",
        "REV-004": "discharge_condition",
        "REV-005": "hospital_course",
        "REV-006": "allergies",
        "REV-007": "escalation_flags_for_clinician",
    }
    return mapping.get(rule_id, "unknown")


def _section_to_category(section_name: str) -> str:
    """Map a section name to a clinical category."""
    mapping = {
        "principal_diagnosis": "diagnosis",
        "discharge_medications": "medication",
        "medication_changes": "medication",
        "pending_results": "pending_result",
        "hospital_course": "citation",
        "discharge_condition": "safety_flag",
        "escalation_flags": "safety_flag",
        "allergies": "safety_flag",
    }
    section_key = section_name.lower().replace(" ", "_")
    return mapping.get(section_key, "other")
