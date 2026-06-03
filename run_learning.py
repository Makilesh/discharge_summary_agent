"""
run_learning.py — CLI Entry Point for Part 2 Learning Loop
=============================================================

Usage:
    python run_learning.py --pdf "patient 2 (1).pdf" --n-train 10 --output output/part2/
    python run_learning.py --pdf "patient 2 (1).pdf" --exploit-only

Runs the Part 2 learning loop:
    1. Loads the Part 1 agent state (runs the agent if no cached state exists)
    2. Executes N training iterations with the learning loop
    3. Runs held-out evaluation
    4. Generates all output artifacts
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.state import create_initial_state
from src.graph import build_agent_graph
from src.learning_loop import LearningOrchestrator
from src.config import GOOGLE_API_KEY


def _load_or_run_part1(pdf_path: str, output_dir: str) -> dict:
    """
    Load cached Part 1 state or run the agent.

    Purpose:
        Avoids re-running the expensive Part 1 pipeline on every Part 2 iteration.
        If output/state.json exists, loads it directly.

    Args:
        pdf_path: Path to the patient PDF.
        output_dir: Directory containing Part 1 output.

    Returns:
        The Part 1 agent state dict.

    Safety Constraint:
        If the cached state is empty or corrupt, re-runs the agent.

    Failure Behavior:
        On agent failure, returns a minimal valid state.
    """
    cached_state_path = os.path.join(output_dir, "state.json")

    if os.path.exists(cached_state_path):
        try:
            with open(cached_state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Basic validity check
            if isinstance(state, dict) and "current_phase" in state:
                print(f"  ✓ Loaded cached Part 1 state from {cached_state_path}")
                return state

        except Exception as e:
            print(f"  ⚠ Failed to load cached state: {e}. Re-running Part 1 agent.")

    # Run Part 1 agent
    print("  Running Part 1 agent (this may take several minutes)...")
    initial_state = create_initial_state()
    initial_state["_pdf_path"] = str(pdf_path)

    agent = build_agent_graph()
    current_state = dict(initial_state)

    try:
        for event in agent.stream(initial_state, {"recursion_limit": 150}):
            for node_name, node_output in event.items():
                for k, v in node_output.items():
                    if k in (
                        "loaded_documents", "unreadable_pages", "procedures",
                        "admission_medications", "inpatient_medications", "discharge_medications",
                        "medication_reconciliation", "lab_results", "imaging_results",
                        "pending_results", "follow_up_instructions", "conflicts",
                        "escalation_flags", "fabrication_blocks", "trace"
                    ):
                        current_state[k] = current_state.get(k, []) + (v if isinstance(v, list) else [v])
                    else:
                        current_state[k] = v

        return current_state

    except Exception as e:
        print(f"  ERROR: Part 1 agent failed: {e}")
        return current_state


def main() -> None:
    """Main entry point for the Part 2 Learning Loop."""
    parser = argparse.ArgumentParser(
        description="Discharge Summary Agent — Part 2: Learning from Doctor Edits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_learning.py --pdf "patient 2 (1).pdf"
    python run_learning.py --pdf "patient 2 (1).pdf" --n-train 10 --output output/part2/
    python run_learning.py --pdf "patient 2 (1).pdf" --exploit-only
        """
    )
    parser.add_argument(
        "--pdf", required=False, default=None,
        help="Path to the patient PDF file (optional if --state is provided)"
    )
    parser.add_argument(
        "--state", required=False, default=None,
        help="Path to a cached Part 1 state.json (skips PDF processing)"
    )
    parser.add_argument(
        "--n-train", type=int, default=10,
        help="Number of training iterations (default: 10)"
    )
    parser.add_argument(
        "--output", default="output/part2",
        help="Output directory for Part 2 results (default: output/part2/)"
    )
    parser.add_argument(
        "--exploit-only", action="store_true",
        help="Skip exploration — use best known arm only"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Validate inputs
    if args.state:
        state_path = Path(args.state).resolve()
        if not state_path.exists():
            print(f"ERROR: State file not found: {state_path}")
            sys.exit(1)
        pdf_path = Path(args.pdf).resolve() if args.pdf else Path("N/A")
    elif args.pdf:
        pdf_path = Path(args.pdf).resolve()
        if not pdf_path.exists():
            # Check if cached state exists as fallback
            cached = Path("output/state.json")
            if cached.exists():
                print(f"  ⚠ PDF not found at {pdf_path}, but cached state.json exists. Using cached state.")
                args.state = str(cached)
            else:
                print(f"ERROR: PDF file not found: {pdf_path}")
                sys.exit(1)
    else:
        # Neither --pdf nor --state: check for default cached state
        cached = Path("output/state.json")
        if cached.exists():
            print(f"  Using cached state from output/state.json")
            args.state = str(cached)
            pdf_path = Path("N/A")
        else:
            print("ERROR: Must provide either --pdf or --state")
            sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate run manifest
    run_id = str(uuid.uuid4())
    run_manifest = {
        "run_id": run_id,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "pdf_path": str(pdf_path),
        "n_train": args.n_train,
        "output_dir": str(output_dir.resolve()),
        "exploit_only": args.exploit_only,
        "seed": args.seed,
        "python_version": sys.version,
        "cli_args": vars(args),
    }

    # Try to get git commit hash
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=str(Path(__file__).parent),
        ).decode().strip()
        run_manifest["git_commit"] = git_hash
    except Exception:
        run_manifest["git_commit"] = "N/A"

    print("=" * 70)
    print("  DISCHARGE SUMMARY AGENT — Part 2")
    print("  Learning from Doctor Edits")
    print("=" * 70)
    print(f"\n  Run ID:     {run_id}")
    print(f"  PDF:        {pdf_path}")
    print(f"  N-Train:    {args.n_train}")
    print(f"  Output:     {output_dir.resolve()}")
    print(f"  Exploit:    {'YES' if args.exploit_only else 'NO'}")
    print(f"  Seed:       {args.seed}")
    print(f"  Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'=' * 70}\n")

    # Step 1: Load or run Part 1
    print("[1/4] Loading Part 1 agent state...")
    if args.state:
        with open(args.state, "r", encoding="utf-8") as f:
            agent_state = json.load(f)
        print(f"  ✓ Loaded state directly from {args.state}")
    else:
        agent_state = _load_or_run_part1(str(pdf_path), "output")
    print(f"  ✓ State loaded. Phase: {agent_state.get('current_phase', '?')}")
    print(f"  ✓ Documents: {len(agent_state.get('loaded_documents', []))} loaded")
    print(f"  ✓ Conflicts: {len(agent_state.get('conflicts', []))}")
    print(f"  ✓ Escalation flags: {len(agent_state.get('escalation_flags', []))}")

    # Step 2: Run training
    print("\n[2/4] Running training loop...")
    start_time = time.time()

    orchestrator = LearningOrchestrator(
        agent_state=agent_state,
        output_dir=str(output_dir),
        n_train=args.n_train,
        exploit_only=args.exploit_only,
    )

    training_records = orchestrator.run_training()
    elapsed = time.time() - start_time
    print(f"\n  Training completed in {elapsed:.1f}s ({len(training_records)} iterations)")

    # Step 3: Run evaluation
    print("\n[3/4] Running held-out evaluation...")
    eval_results = orchestrator.run_evaluation()

    # Step 4: Generate output artifacts
    print(f"\n[4/4] Writing output artifacts to {output_dir}/...")
    orchestrator.generate_output_artifacts(eval_results)

    # Write run manifest
    manifest_path = output_dir / "run_manifest.json"
    run_manifest["elapsed_seconds"] = elapsed
    run_manifest["n_iterations_completed"] = len(training_records)
    run_manifest["final_reward"] = training_records[-1].composite_reward if training_records else 0.0
    run_manifest["best_arm"] = eval_results.get("best_arm_name", "N/A")
    run_manifest["improvement_delta"] = eval_results.get("improvement_delta", 0.0)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"  + run_manifest.json")

    # Initialize empty gaming_alerts.log if it doesn't exist
    alerts_path = output_dir / "gaming_alerts.log"
    if not alerts_path.exists():
        alerts_path.touch()
        print(f"  + gaming_alerts.log (empty)")

    # Print summary
    if training_records:
        rewards = [r.composite_reward for r in training_records]
        print(f"\n{'=' * 70}")
        print(f"  RESULTS")
        print(f"{'=' * 70}")
        print(f"  Iterations completed:  {len(training_records)}")
        print(f"  First reward:          {rewards[0]:.4f}")
        print(f"  Final reward:          {rewards[-1]:.4f}")
        print(f"  Mean reward:           {sum(rewards)/len(rewards):.4f}")
        print(f"  Max reward:            {max(rewards):.4f}")
        print(f"  Baseline reward:       {eval_results.get('baseline_reward', 0):.4f}")
        print(f"  Best arm reward:       {eval_results.get('best_reward', 0):.4f}")
        print(f"  Improvement delta:     {eval_results.get('improvement_delta', 0):+.4f}")
        print(f"  Best arm:              {eval_results.get('best_arm_name', 'N/A')}")
        print(f"  Total time:            {elapsed:.1f}s")
        print(f"{'=' * 70}\n")
    else:
        print("\n  WARNING: No training iterations completed successfully.\n")


if __name__ == "__main__":
    main()
