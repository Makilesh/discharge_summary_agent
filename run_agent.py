"""
run_agent.py — CLI Entry Point for the Discharge Summary Agent
================================================================

Usage:
    python run_agent.py --pdf "patient 2 (1).pdf" --output output/

Runs the LangGraph agent on a patient PDF and produces:
    - discharge_summary.md     — The structured summary draft
    - trace.json               — Full step-by-step audit trace
    - state.json               — Final agent state (for debugging)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.state import create_initial_state
from src.graph import build_agent_graph
from src.config import GOOGLE_API_KEY


def main() -> None:
    """Main entry point for the Discharge Summary Agent."""
    parser = argparse.ArgumentParser(
        description="Discharge Summary Agent — AI-powered clinical document extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_agent.py --pdf "patient 2 (1).pdf"
    python run_agent.py --pdf "patient 2 (1).pdf" --output results/
        """
    )
    parser.add_argument(
        "--pdf", required=True,
        help="Path to the patient PDF file"
    )
    parser.add_argument(
        "--output", default="output",
        help="Output directory for results (default: output/)"
    )
    args = parser.parse_args()

    # Validate inputs
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF file not found: {pdf_path}")
        sys.exit(1)

    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not set. Please create a .env file with your API key.")
        print("  echo GOOGLE_API_KEY=your-key-here > .env")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DISCHARGE SUMMARY AGENT — Part 1")
    print("  Clinical Document Extraction System")
    print("=" * 70)
    print(f"\n  PDF:    {pdf_path}")
    print(f"  Output: {output_dir.resolve()}")
    print(f"  Model:  Gemini 2.0 Flash")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n{'=' * 70}\n")

    # Initialize state
    print("[1/4] Initializing agent state...")
    initial_state = create_initial_state()
    initial_state["_pdf_path"] = str(pdf_path)

    # Build and run the graph
    print("[2/4] Building LangGraph agent...")
    agent = build_agent_graph()

    print("[3/4] Running agent (this may take several minutes)...")
    print("      Processing 71-page scanned PDF with Gemini Vision OCR...")
    print()

    start_time = time.time()

    try:
        # Stream execution for observability
        step_count = 0
        final_state = None

        for event in agent.stream(initial_state, {"recursion_limit": 50}):
            step_count += 1
            for node_name, node_output in event.items():
                phase = node_output.get("current_phase", "")
                steps_rem = node_output.get("steps_remaining", "?")
                print(f"  Step {step_count:2d} | Node: {node_name:<20s} | "
                      f"Phase: {phase:<15s} | Steps left: {steps_rem}")

                # Capture final state
                if node_name in ("compile", "hard_cap_escalate"):
                    final_state = node_output

        elapsed = time.time() - start_time
        print(f"\n  Agent completed in {elapsed:.1f}s ({step_count} graph steps)")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n  ERROR: Agent failed after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not final_state:
        print("\n  ERROR: Agent did not produce a final state.")
        sys.exit(1)

    # Write outputs
    print(f"\n[4/4] Writing outputs to {output_dir}/...")

    # 1. Discharge Summary
    summary = final_state.get("final_summary", "")
    summary_path = output_dir / "discharge_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  ✓ discharge_summary.md ({len(summary):,} chars)")

    # 2. Trace
    trace = final_state.get("trace", [])
    trace_path = output_dir / "trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=str)
    print(f"  ✓ trace.json ({len(trace)} entries)")

    # 3. State (sanitized — remove large binary data)
    state_output = {k: v for k, v in final_state.items()
                    if k not in ("page_images", "_pdf_path") and not k.startswith("_")}
    # Truncate raw_text in loaded_documents for output
    if "loaded_documents" in state_output:
        for doc in state_output["loaded_documents"]:
            if "raw_text" in doc:
                doc["raw_text"] = doc["raw_text"][:500] + "..." if len(doc.get("raw_text", "")) > 500 else doc.get("raw_text", "")
    state_path = output_dir / "state.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state_output, f, indent=2, default=str)
    print(f"  ✓ state.json")

    # Print summary stats
    conflicts = final_state.get("conflicts", [])
    escalations = final_state.get("escalation_flags", [])
    critical = sum(1 for e in escalations if e.get("severity") == "CRITICAL")

    print(f"\n{'=' * 70}")
    print(f"  RESULTS")
    print(f"{'=' * 70}")
    print(f"  Conflicts detected:     {len(conflicts)}")
    print(f"  Escalation flags:       {len(escalations)} ({critical} CRITICAL)")
    print(f"  Fabrication blocks:     {len(final_state.get('fabrication_blocks', []))}")
    print(f"  Summary status:         DRAFT — NOT FOR CLINICAL USE WITHOUT REVIEW")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
