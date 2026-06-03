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
from src.config import (
    GOOGLE_API_KEY, LLM_MODEL, LLM_BACKEND,
    OLLAMA_BASE_URL, REASONING_BACKUP_MODEL, VISION_BACKUP_MODEL,
)


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
    parser.add_argument(
        "--skip-llm-preflight",
        action="store_true",
        help="Skip the one-request Gemini preflight check. Use only when quota/model availability is already known."
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "gemini", "local"],
        default=None,
        help="LLM backend: auto=Gemini then Ollama fallback, gemini=Gemini only, local=Ollama only."
    )
    parser.add_argument(
        "--gemini-model",
        default=None,
        help="Gemini model name, e.g. gemini-2.5-flash-lite."
    )
    parser.add_argument(
        "--reasoning-model",
        default=None,
        help="Ollama text reasoning model, default deepseek-r1:14b."
    )
    parser.add_argument(
        "--vision-model",
        default=None,
        help="Ollama vision/OCR model, default qwen2.5vl:7b."
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Ollama OpenAI-compatible base URL."
    )
    args = parser.parse_args()
    _apply_runtime_llm_config(args)

    # Validate inputs
    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"ERROR: PDF file not found: {pdf_path}")
        sys.exit(1)

    backend = os.getenv("LLM_BACKEND", LLM_BACKEND).lower()
    gemini_model = os.getenv("LLM_MODEL", LLM_MODEL)
    reasoning_model = os.getenv("REASONING_BACKUP_MODEL", REASONING_BACKUP_MODEL)
    vision_model = os.getenv("VISION_BACKUP_MODEL", VISION_BACKUP_MODEL)
    google_api_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    is_dummy_key = not google_api_key or "your-google-api-key" in google_api_key

    if backend == "local":
        print(f"Using local Ollama backend: reasoning={reasoning_model}, vision={vision_model}.")
    elif is_dummy_key:
        print(f"WARNING: GOOGLE_API_KEY not set or dummy. Running with local Ollama fallback: reasoning={reasoning_model}, vision={vision_model}.")
    else:
        print(f"GOOGLE_API_KEY detected. Backend={backend}. Gemini={gemini_model}; local fallback reasoning={reasoning_model}, vision={vision_model}.")
        if args.skip_llm_preflight:
            print("Skipping Gemini preflight check by request.")
        else:
            _preflight_gemini(gemini_model, google_api_key)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DISCHARGE SUMMARY AGENT — Part 1")
    print("  Clinical Document Extraction System")
    print("=" * 70)
    print(f"\n  PDF:    {pdf_path}")
    print(f"  Output: {output_dir.resolve()}")
    model_label = (
        f"Ollama reasoning={reasoning_model}, vision={vision_model}"
        if backend == "local" or is_dummy_key else
        f"{gemini_model} ({backend}; fallback reasoning={reasoning_model}, vision={vision_model})"
    )
    print(f"  Model:  {model_label}")
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
    print(f"      Processing scanned PDF with {model_label}...")
    print()

    start_time = time.time()

    try:
        # Stream execution for observability, maintaining full accumulated state
        step_count = 0
        current_state = dict(initial_state)

        for event in agent.stream(initial_state, {"recursion_limit": 150}):
            step_count += 1
            for node_name, node_output in event.items():
                phase = node_output.get("current_phase", "")
                steps_rem = node_output.get("steps_remaining", "?")
                print(f"  Step {step_count:2d} | Node: {node_name:<20s} | "
                      f"Phase: {phase:<15s} | Steps left: {steps_rem}")

                # Accumulate state updates matching LangGraph reducers
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

        final_state = current_state
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
    print(f"  + discharge_summary.md ({len(summary):,} chars)")

    # 2. Trace
    trace = final_state.get("trace", [])
    trace_path = output_dir / "trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, default=str)
    print(f"  + trace.json ({len(trace)} entries)")

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
    print(f"  + state.json")

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


def _apply_runtime_llm_config(args: argparse.Namespace) -> None:
    """Apply CLI LLM choices before graph execution."""
    if args.backend:
        os.environ["LLM_BACKEND"] = args.backend
    else:
        os.environ.setdefault("LLM_BACKEND", LLM_BACKEND)
    if args.gemini_model:
        os.environ["LLM_MODEL"] = args.gemini_model
    else:
        os.environ.setdefault("LLM_MODEL", LLM_MODEL)
    if args.reasoning_model:
        os.environ["REASONING_BACKUP_MODEL"] = args.reasoning_model
    else:
        os.environ.setdefault("REASONING_BACKUP_MODEL", REASONING_BACKUP_MODEL)
    if args.vision_model:
        os.environ["VISION_BACKUP_MODEL"] = args.vision_model
    else:
        os.environ.setdefault("VISION_BACKUP_MODEL", VISION_BACKUP_MODEL)
    if args.ollama_base_url:
        os.environ["OLLAMA_BASE_URL"] = args.ollama_base_url
    else:
        os.environ.setdefault("OLLAMA_BASE_URL", OLLAMA_BASE_URL)


def _preflight_gemini(model_name: str, google_api_key: str) -> None:
    """Fail fast if the configured Gemini model/key cannot serve requests."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0.0,
            max_output_tokens=8,
        )
        llm.invoke([HumanMessage(content="Return only OK.")])
    except Exception as e:
        print(f"ERROR: Gemini preflight failed for model {model_name}: {e}")
        print("       Fix GOOGLE_API_KEY/LLM_MODEL or wait for quota reset before running scanned-PDF extraction.")
        sys.exit(1)


if __name__ == "__main__":
    main()
