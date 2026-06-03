"""
bandit.py — Contextual Bandit (UCB1) for Prompt Strategy Selection
===================================================================

A UCB1 (Upper Confidence Bound) contextual bandit that selects among
K prompt strategy arms for the compilation phase. Each arm is a different
system prompt template. The bandit is rewarded by the composite reward
signal from the Edit Signal Engine.

Clinical Safety:
    - Arm 0 (BASELINE) is the control condition — always available.
    - All arms include the anti-vagueness instruction.
    - GamingDetector runs after every 5 iterations to catch reward hacking.
    - State is persisted atomically to bandit_state.json.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import math
import os
from datetime import datetime, timezone
from typing import Optional


# ─── CONSTANTS ──────────────────────────────────────────────────────────────────

# Optimistic initialization value for unpulled arms.
# Arms with no data default to this mean reward, encouraging exploration.
OPTIMISTIC_INIT_REWARD: float = 0.5

# Number of arms in the bandit.
NUM_ARMS: int = 5

# Default path for persisted bandit state.
DEFAULT_BANDIT_STATE_PATH: str = "output/bandit_state.json"

# Anti-vagueness instruction included in every arm's prompt.
# This is a non-negotiable safety constraint.
ANTI_VAGUENESS_INSTRUCTION: str = (
    "Never omit a clinical fact to reduce the amount of editing. "
    "Vagueness is a safety failure. Every known clinical fact MUST appear "
    "in the output, even if it increases the draft length."
)


# ─── INJECTION POLICY ENUM ──────────────────────────────────────────────────────

class InjectionPolicy(str, Enum):
    """How correction examples are injected into the compiler prompt."""
    NONE = "NONE"            # No correction injection
    PER_SECTION = "PER_SECTION"  # Corrections injected per-section
    GLOBAL = "GLOBAL"        # All corrections as a global preamble


# ─── PROMPT STRATEGY ────────────────────────────────────────────────────────────

@dataclass
class PromptStrategy:
    """
    A prompt strategy arm for the contextual bandit.

    Each arm defines a different compiler system prompt and correction
    injection policy. Arms vary in clinical content strategy, not merely
    formatting or verbosity.

    Clinical Significance:
        - arm_id 0 (BASELINE) is the control condition
        - Arms 1-4 test different approaches to incorporating learned corrections
        - All arms include the anti-vagueness instruction
    """
    arm_id: int                      # 0-4
    name: str                        # Human-readable arm name
    compiler_system_prompt: str      # The system prompt template for compilation
    injection_policy: InjectionPolicy  # How corrections are injected


# ─── ARM DEFINITIONS ────────────────────────────────────────────────────────────

# These are module-level constants. Each arm varies in clinical content strategy.

ARM_0_BASELINE = PromptStrategy(
    arm_id=0,
    name="BASELINE",
    compiler_system_prompt=(
        "You are a clinical document compiler. Assemble the discharge summary "
        "from the extracted state fields. Use exact template strings for missing data. "
        "Never fabricate clinical facts. Every claim must have a source page reference. "
        f"{ANTI_VAGUENESS_INSTRUCTION}"
    ),
    injection_policy=InjectionPolicy.NONE,
)

ARM_1_SECTION_EXEMPLARS = PromptStrategy(
    arm_id=1,
    name="SECTION_EXEMPLARS",
    compiler_system_prompt=(
        "You are a clinical document compiler with access to correction examples. "
        "For each section, review the provided correction examples showing common "
        "errors and their fixes. Apply these lessons to produce a more accurate draft. "
        "Prioritize completeness: include ALL clinical facts from the source data. "
        "Use exact template strings for genuinely missing data. "
        f"{ANTI_VAGUENESS_INSTRUCTION}\n\n"
        "{{correction_examples}}"
    ),
    injection_policy=InjectionPolicy.PER_SECTION,
)

ARM_2_GLOBAL_PREAMBLE = PromptStrategy(
    arm_id=2,
    name="GLOBAL_PREAMBLE",
    compiler_system_prompt=(
        "You are a clinical document compiler. Before generating any section, "
        "review the following correction history showing patterns that have improved "
        "accuracy in previous drafts:\n\n"
        "{{correction_examples}}\n\n"
        "Apply these learned corrections proactively to every section. "
        "Focus on: complete medication reconciliation, accurate diagnosis attribution, "
        "and proper escalation flag coverage. "
        f"{ANTI_VAGUENESS_INSTRUCTION}"
    ),
    injection_policy=InjectionPolicy.GLOBAL,
)

ARM_3_CONFLICT_FIRST = PromptStrategy(
    arm_id=3,
    name="CONFLICT_FIRST",
    compiler_system_prompt=(
        "You are a clinical document compiler with a safety-first approach. "
        "BEFORE generating each section, first check: are there any escalation flags, "
        "conflicts, or safety concerns relevant to this section? If yes, address them "
        "prominently at the start of the section. "
        "Escalation flags must NEVER be omitted or downgraded. "
        "Conflicts must ALWAYS show all source documents that disagree. "
        f"{ANTI_VAGUENESS_INSTRUCTION}"
    ),
    injection_policy=InjectionPolicy.NONE,
)

ARM_4_HYBRID_EXEMPLAR = PromptStrategy(
    arm_id=4,
    name="HYBRID_EXEMPLAR",
    compiler_system_prompt=(
        "You are a clinical document compiler combining safety-first review with "
        "learned correction patterns. For each section:\n"
        "1. First, check for escalation flags and conflicts relevant to this section.\n"
        "2. Then, review the correction examples for common errors in this section.\n"
        "3. Generate the section content incorporating both safety checks and lessons.\n\n"
        "{{correction_examples}}\n\n"
        "Escalation flags must NEVER be omitted. Conflicts must show all sources. "
        f"{ANTI_VAGUENESS_INSTRUCTION}"
    ),
    injection_policy=InjectionPolicy.PER_SECTION,
)

# All arms in order
ARMS: list[PromptStrategy] = [
    ARM_0_BASELINE,
    ARM_1_SECTION_EXEMPLARS,
    ARM_2_GLOBAL_PREAMBLE,
    ARM_3_CONFLICT_FIRST,
    ARM_4_HYBRID_EXEMPLAR,
]


# ─── BANDIT DECISION ────────────────────────────────────────────────────────────

@dataclass
class BanditDecision:
    """
    The output of the bandit's arm selection process.

    Provides full transparency into why an arm was selected.

    Clinical Significance:
        - selected_arm: Which prompt strategy will be used
        - ucb1_scores: The UCB1 score for each arm (for debugging)
        - reason: Human-readable explanation for audit trail
    """
    selected_arm: int               # Index of the selected arm (0-4)
    arm_name: str                   # Human-readable name of the selected arm
    ucb1_scores: dict[int, float]   # UCB1 score for each arm
    exploration_bonus: dict[int, float]  # Exploration bonus for each arm
    total_pulls: int                # Total number of pulls across all arms
    reason: str                     # Human-readable explanation


# ─── ARM STATISTICS ──────────────────────────────────────────────────────────────

@dataclass
class ArmStats:
    """Statistics for a single bandit arm."""
    arm_id: int
    n_pulls: int = 0
    total_reward: float = 0.0
    mean_reward: float = OPTIMISTIC_INIT_REWARD  # Optimistic initialization
    rewards_history: list[float] = field(default_factory=list)


# ─── UCB1 CONTEXTUAL BANDIT ─────────────────────────────────────────────────────

class ContextualBandit:
    """
    UCB1 contextual bandit for prompt strategy selection.

    Purpose:
        Selects among K prompt strategy arms to find the compilation
        approach that minimizes edit burden while preserving safety.

    Safety Constraint:
        All arms include the anti-vagueness instruction.
        GamingDetector runs periodically to catch reward hacking.
        State is persisted atomically after every update.

    Failure Behavior:
        On state load failure, starts fresh with optimistic initialization.
        On save failure, logs error but continues in-memory.
    """

    def __init__(self, state_path: str = DEFAULT_BANDIT_STATE_PATH):
        """
        Initialize the bandit.

        Args:
            state_path: Path to persist bandit state JSON.
        """
        self.state_path = state_path
        self.arms: dict[int, ArmStats] = {
            i: ArmStats(arm_id=i) for i in range(NUM_ARMS)
        }
        self.total_pulls: int = 0
        self._load_state()

    def _load_state(self) -> None:
        """
        Load bandit state from JSON file.

        Purpose:
            Restores arm statistics from previous runs for continuity.

        Safety Constraint:
            On failure, starts with fresh optimistic initialization.

        Failure Behavior:
            Logs warning on failure, continues with default state.
        """
        if not os.path.exists(self.state_path):
            return

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.total_pulls = data.get("total_pulls", 0)
            for arm_data in data.get("arms", []):
                arm_id = arm_data.get("arm_id", 0)
                if arm_id in self.arms:
                    self.arms[arm_id].n_pulls = arm_data.get("n_pulls", 0)
                    self.arms[arm_id].total_reward = arm_data.get("total_reward", 0.0)
                    self.arms[arm_id].mean_reward = arm_data.get("mean_reward", OPTIMISTIC_INIT_REWARD)
                    self.arms[arm_id].rewards_history = arm_data.get("rewards_history", [])

        except Exception as e:
            print(f"[BANDIT] WARNING: Failed to load state from {self.state_path}: {e}")

    def save_state(self) -> None:
        """
        Persist bandit state to JSON (atomic write).

        Purpose:
            Saves all arm statistics for continuity across runs.

        Safety Constraint:
            Atomic write using tmp file + os.replace().
            Never produces a partial write.

        Failure Behavior:
            Logs error on failure. In-memory state is preserved.
        """
        try:
            os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)

            state_data = {
                "total_pulls": self.total_pulls,
                "last_updated_iso": datetime.now(timezone.utc).isoformat(),
                "arms": [
                    {
                        "arm_id": arm.arm_id,
                        "n_pulls": arm.n_pulls,
                        "total_reward": arm.total_reward,
                        "mean_reward": arm.mean_reward,
                        "rewards_history": arm.rewards_history,
                    }
                    for arm in self.arms.values()
                ],
            }

            tmp_path = self.state_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            os.replace(tmp_path, self.state_path)

        except Exception as e:
            print(f"[BANDIT] ERROR: Failed to save state to {self.state_path}: {e}")

    def select_arm(self, exploit_only: bool = False) -> BanditDecision:
        """
        Select an arm using UCB1 (or pure exploitation if exploit_only).

        Purpose:
            Returns the arm with the highest UCB1 score, or the arm with
            the highest mean reward if in exploit-only mode.

        Args:
            exploit_only: If True, skip exploration and select the best known arm.

        Returns:
            BanditDecision with the selected arm and full scoring breakdown.

        Safety Constraint:
            Unpulled arms are always selected first (optimistic initialization
            gives them UCB1 = infinity effectively).

        Failure Behavior:
            Returns arm 0 (BASELINE) on any error.
        """
        try:
            ucb1_scores: dict[int, float] = {}
            exploration_bonus: dict[int, float] = {}

            if exploit_only:
                # Pure exploitation: select arm with highest mean reward
                for arm_id, stats in self.arms.items():
                    ucb1_scores[arm_id] = stats.mean_reward
                    exploration_bonus[arm_id] = 0.0

                best_arm = max(ucb1_scores, key=ucb1_scores.get)
                return BanditDecision(
                    selected_arm=best_arm,
                    arm_name=ARMS[best_arm].name,
                    ucb1_scores=ucb1_scores,
                    exploration_bonus=exploration_bonus,
                    total_pulls=self.total_pulls,
                    reason=f"Exploit-only mode: selected arm {best_arm} ({ARMS[best_arm].name}) "
                           f"with highest mean reward {ucb1_scores[best_arm]:.4f}",
                )

            # Check for unpulled arms first (optimistic initialization)
            for arm_id, stats in self.arms.items():
                if stats.n_pulls == 0:
                    # Unpulled arm — select it immediately
                    for aid in self.arms:
                        if self.arms[aid].n_pulls == 0:
                            ucb1_scores[aid] = float('inf')
                            exploration_bonus[aid] = float('inf')
                        else:
                            bonus = math.sqrt(
                                2 * math.log(max(self.total_pulls, 1)) / self.arms[aid].n_pulls
                            )
                            ucb1_scores[aid] = self.arms[aid].mean_reward + bonus
                            exploration_bonus[aid] = bonus

                    return BanditDecision(
                        selected_arm=arm_id,
                        arm_name=ARMS[arm_id].name,
                        ucb1_scores=ucb1_scores,
                        exploration_bonus=exploration_bonus,
                        total_pulls=self.total_pulls,
                        reason=f"Arm {arm_id} ({ARMS[arm_id].name}) has never been pulled — "
                               f"optimistic initialization selects it first",
                    )

            # All arms have been pulled at least once — use UCB1 formula
            for arm_id, stats in self.arms.items():
                bonus = math.sqrt(
                    2 * math.log(self.total_pulls) / stats.n_pulls
                )
                ucb1_scores[arm_id] = stats.mean_reward + bonus
                exploration_bonus[arm_id] = bonus

            best_arm = max(ucb1_scores, key=ucb1_scores.get)

            return BanditDecision(
                selected_arm=best_arm,
                arm_name=ARMS[best_arm].name,
                ucb1_scores=ucb1_scores,
                exploration_bonus=exploration_bonus,
                total_pulls=self.total_pulls,
                reason=f"UCB1 selected arm {best_arm} ({ARMS[best_arm].name}) "
                       f"with score {ucb1_scores[best_arm]:.4f} "
                       f"(mean={self.arms[best_arm].mean_reward:.4f}, "
                       f"bonus={exploration_bonus[best_arm]:.4f})",
            )

        except Exception as e:
            print(f"[BANDIT] ERROR: Arm selection failed: {e}")
            return BanditDecision(
                selected_arm=0,
                arm_name="BASELINE",
                ucb1_scores={i: 0.0 for i in range(NUM_ARMS)},
                exploration_bonus={i: 0.0 for i in range(NUM_ARMS)},
                total_pulls=self.total_pulls,
                reason=f"Fallback to BASELINE due to error: {e}",
            )

    def update(self, arm_id: int, reward: float) -> None:
        """
        Update arm statistics after observing a reward.

        Purpose:
            Incremental mean update for the selected arm's reward.

        Args:
            arm_id: The arm that was pulled.
            reward: The observed composite reward (0.0–1.0).

        Safety Constraint:
            Reward is clamped to [0.0, 1.0] before updating.
            State is persisted atomically after update.

        Failure Behavior:
            Logs error on failure. In-memory state is still updated.
        """
        reward = max(0.0, min(1.0, reward))

        if arm_id not in self.arms:
            print(f"[BANDIT] ERROR: Invalid arm_id {arm_id}")
            return

        stats = self.arms[arm_id]
        stats.n_pulls += 1
        stats.total_reward += reward
        stats.mean_reward = stats.total_reward / stats.n_pulls
        stats.rewards_history.append(reward)
        self.total_pulls += 1

        self.save_state()

    def get_strategy(self, arm_id: int) -> PromptStrategy:
        """
        Get the PromptStrategy for a given arm.

        Args:
            arm_id: The arm index (0-4).

        Returns:
            The PromptStrategy for the requested arm.

        Safety Constraint:
            Returns BASELINE if arm_id is invalid.

        Failure Behavior:
            Returns ARM_0_BASELINE on any error.
        """
        if 0 <= arm_id < len(ARMS):
            return ARMS[arm_id]
        return ARM_0_BASELINE


# ─── GAMING DETECTOR ─────────────────────────────────────────────────────────────

class GamingDetector:
    """
    Detects reward gaming patterns in the learning loop.

    Purpose:
        Runs after every 5 iterations to check if the bandit is learning
        to produce vague or shorter output to game the reward metric,
        rather than actually improving clinical accuracy.

    Safety Constraint:
        Gaming alerts are logged to a file and trigger human review.
        The detector NEVER modifies the bandit's state or decisions.

    Failure Behavior:
        On failure, logs error but does not interrupt the training loop.
    """

    def __init__(self, alert_log_path: str = "output/part2/gaming_alerts.log"):
        """
        Initialize the gaming detector.

        Args:
            alert_log_path: Path to the gaming alerts log file.
        """
        self.alert_log_path = alert_log_path
        self.iteration_history: list[dict] = []
        # Ensure directory exists
        os.makedirs(os.path.dirname(alert_log_path) or ".", exist_ok=True)

    def record_iteration(
        self,
        iteration: int,
        composite_reward: float,
        section_scores: dict[str, float],
        hospital_course_word_count: int,
    ) -> None:
        """
        Record metrics for a training iteration.

        Purpose:
            Stores per-iteration metrics for trend analysis.

        Args:
            iteration: The iteration number.
            composite_reward: The composite reward for this iteration.
            section_scores: Per-section match scores.
            hospital_course_word_count: Word count of the hospital course section.

        Safety Constraint:
            Only appends to history — never modifies previous entries.

        Failure Behavior:
            Never raises.
        """
        self.iteration_history.append({
            "iteration": iteration,
            "composite_reward": composite_reward,
            "section_scores": section_scores,
            "hospital_course_word_count": hospital_course_word_count,
        })

    def check_for_gaming(self, iteration: int) -> list[str]:
        """
        Check for gaming patterns. Should be called after every 5 iterations.

        Purpose:
            Detects two specific gaming patterns:
            1. Reward increased but principal_diagnosis accuracy decreased
            2. Hospital course word count decreased >20% over 3 consecutive iterations

        Args:
            iteration: Current iteration number.

        Returns:
            List of alert messages (empty if no gaming detected).

        Safety Constraint:
            Alerts are informational — they never modify the learning loop.

        Failure Behavior:
            Returns empty list on error.
        """
        alerts: list[str] = []

        if iteration % 5 != 0 or len(self.iteration_history) < 2:
            return alerts

        try:
            # Check 1: Reward up but principal_diagnosis accuracy down
            recent = self.iteration_history[-1]
            previous = self.iteration_history[-2]

            recent_reward = recent["composite_reward"]
            previous_reward = previous["composite_reward"]
            recent_pd = recent["section_scores"].get("principal_diagnosis", 0.0)
            previous_pd = previous["section_scores"].get("principal_diagnosis", 0.0)

            if recent_reward > previous_reward and recent_pd < previous_pd:
                alert = (
                    f"GAMING_ALERT [Iteration {iteration}]: reward gain may reflect "
                    f"vagueness, not accuracy improvement. "
                    f"Reward: {previous_reward:.4f} → {recent_reward:.4f}, "
                    f"principal_diagnosis: {previous_pd:.4f} → {recent_pd:.4f}"
                )
                alerts.append(alert)
                self._log_alert(alert)

            # Check 2: Hospital course word count progressive decline
            if len(self.iteration_history) >= 3:
                last_3 = self.iteration_history[-3:]
                wc = [h["hospital_course_word_count"] for h in last_3]

                if wc[0] > 0:
                    decline_pct = (wc[0] - wc[2]) / wc[0]
                    if decline_pct > 0.20 and wc[1] < wc[0] and wc[2] < wc[1]:
                        alert = (
                            f"GAMING_ALERT [Iteration {iteration}]: progressive summarization "
                            f"collapse detected. Hospital course word count: "
                            f"{wc[0]} → {wc[1]} → {wc[2]} "
                            f"({decline_pct:.1%} decline over 3 iterations)"
                        )
                        alerts.append(alert)
                        self._log_alert(alert)

        except Exception as e:
            print(f"[GAMING_DETECTOR] ERROR: {e}")

        return alerts

    def _log_alert(self, alert: str) -> None:
        """
        Append a gaming alert to the log file.

        Purpose:
            Persistent record of all gaming alerts for audit.

        Safety Constraint:
            Append-only. Never removes existing alerts.

        Failure Behavior:
            Logs error to stdout if file write fails.
        """
        try:
            with open(self.alert_log_path, "a", encoding="utf-8") as f:
                timestamp = datetime.now(timezone.utc).isoformat()
                f.write(f"[{timestamp}] {alert}\n")
        except Exception as e:
            print(f"[GAMING_DETECTOR] ERROR: Failed to write alert to log: {e}")
