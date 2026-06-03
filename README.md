# Discharge Summary Agent — Part 1

> An agentic AI system that reads messy, multi-page hospital PDFs and produces a structured, clinically safe discharge summary draft for clinician review.

## Architecture

### Agent Loop Design

The agent is implemented as a **deterministic state machine** using LangGraph's `StateGraph`, with the following phase cycle:

```
INITIALIZE → REASON → PLAN → CALL_TOOL → OBSERVE → VERIFY → [REASON | COMPILE | ESCALATE]
```

**Key design decisions:**
- **LangGraph StateGraph** — gives us typed state, conditional edges, and streaming observability for free
- **Gemini 2.0 Flash Vision** — handles OCR of handwritten + typed scanned pages without a separate OCR library
- **Batch page processing** — same-type pages are grouped and processed together to maximize the 20-step budget
- **Deterministic routing** — `route_after_verify()` function with explicit conditions, not LLM-based routing

### Control Mechanisms

| Mechanism | Implementation |
|-----------|---------------|
| **Step cap** | `MAX_ITERATIONS = 20`. Decremented every transition. Cap breach → `HARD_CAP_ESCALATE` with `[MISSING]` markers. |
| **Retry limit** | `MAX_RETRIES = 2` per tool. Third failure → `[UNRESOLVED]` marker, field added to `fabrication_blocks`. |
| **Priority order** | 13-level extraction priority (typed summary first, discharge checklist last). Never skips ahead. |

## No-Fabrication Guardrail

This is the core safety constraint. The agent **never** invents clinical facts:

1. **Missing data** → `[MISSING — not documented in source records. Clinician must supply.]`
2. **Pending results** → `[PENDING — {item} sent {date}. Result not available in source documents.]`
3. **Conflicts** → `[CONFLICT — Multiple sources disagree. See escalation_flags. Clinician must resolve.]`
4. **Unclear text** → `[UNCLEAR — Source text partially legible: '{best_guess}'. Verify against original.]`

These literal strings appear in the output. The agent never substitutes a plausible value.

### Cross-Reference Audit (CR Rules)

Before compiling the final summary, a mandatory `cross_reference_audit()` runs 5 safety rules:

| Rule | What it detects | Example in Patient 2 |
|------|----------------|---------------------|
| **CR-1** | Treatment without matching diagnosis | Insulin given but final diagnosis omits DM/DKA |
| **CR-2** | Diagnoses disagree across documents | ER says DKA, admission says TAFE, ICU says DKA+T2DM |
| **CR-3** | Labs critical but not addressed, or "resolved" contradicted by labs | Na 114 mmol/L, glucose 443 mg/dL |
| **CR-4** | Negative culture but broad-spectrum antibiotics given | Urine culture sterile but IV Meropenem administered |
| **CR-5** | Discharge condition contradicted by vitals / DAMA indicators | "Discharge on Request" noted in multiple documents |

Every conflict triggers `escalate_to_clinician()` — the agent **never silently resolves** a conflict.

## Failure & Conflict Handling

- **Tool failures**: Wrapped in `safe_tool_call()` with 2 retries. On final failure, the field is marked `[UNRESOLVED]` and added to `fabrication_blocks`.
- **OCR failures**: Pages with < 30 chars of extracted text are added to `unreadable_pages`.
- **Conflicting data**: Both values are preserved. The output shows `[CONFLICT]` with all sources listed.
- **DAMA (Discharge Against Medical Advice)**: Automatically detected from keywords in nursing notes and consultation sheets.

## Project Structure

```
src/
├── state.py           # AgentState TypedDict, ClinicalFlag, MedicationEntry
├── config.py          # Constants, DOC_TYPES, templates, thresholds
├── trace.py           # Trace emission, state validation, trace summary
├── pdf_processor.py   # PDF → page images (PyMuPDF)
├── tools.py           # LLM-backed extraction tools + safe_tool_call
├── cross_reference.py # CR-1 through CR-5 audit rules
├── compiler.py        # Final Markdown summary compiler
└── graph.py           # LangGraph StateGraph definition
tests/
└── test_agent.py      # 20 unit tests for all CR rules, reconciliation, caps
run_agent.py           # CLI entry point
```

## Quick Start

```bash
# 1. Create and activate venv
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
echo GOOGLE_API_KEY=your-key-here > .env

# 4. Run tests
python -m pytest tests/ -v

# 5. Run the agent
python run_agent.py --pdf "patient 2 (1).pdf" --output output/
```

## Output Files

| File | Contents |
|------|----------|
| `discharge_summary.md` | Structured 17-section discharge summary with citations and escalation flags |
| `trace.json` | Full step-by-step audit trail of every agent decision |
| `state.json` | Final agent state (all extracted data, conflicts, flags) |

## Limitations

- **OCR quality**: Gemini Vision handles handwriting reasonably well, but heavily degraded scans may produce low-confidence extractions. These are flagged, not silently dropped.
- **Step budget**: The 20-step cap may not be sufficient for very large PDFs. The agent prioritizes high-value documents first and compiles whatever it has if the cap is hit.
- **Drug interaction lookup**: Currently mocked. In production, this would integrate with a clinical pharmacopeia database.
- **No learning loop**: ~~Part 2 (learning from doctor edits) is not implemented in this submission.~~ **Now implemented — see Part 2 below.**

## What I'd Do With More Time

1. ~~**Part 2**: Implement the doctor-edit learning loop with a simulated reviewer and contextual bandit over prompt strategies.~~ **Done.**
2. **Multi-patient batch mode**: Process multiple patient PDFs in parallel.
3. **Confidence-weighted extraction**: Re-read low-confidence pages with different prompting strategies.
4. **Real drug interaction API**: Replace the mock with a real pharmacopeia integration.
5. **Structured output validation**: Use Pydantic models to validate every tool's JSON output.
6. **Token budget tracking**: Monitor LLM token usage and optimize prompts for cost.

---

## Part 2 — Learning from Doctor Edits

### Architecture

```
                    ┌─────────────────────────┐
                    │   LearningOrchestrator   │
                    │   (learning_loop.py)     │
                    └───────────┬─────────────┘
                                │
        ┌───────────┬───────────┼───────────┬──────────────┐
        ▼           ▼           ▼           ▼              ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Contextual│ │Simulated │ │  Edit    │ │Correction│ │  Gaming  │
  │  Bandit   │ │ Reviewer │ │  Signal  │ │  Memory  │ │ Detector │
  │(bandit.py)│ │(sim_rev) │ │(edit_sig)│ │(corr_mem)│ │(bandit)  │
  └─────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
        │             │            │             │            │
        │  select_arm │  review()  │ compute()   │ retrieve() │ check()
        ▼             ▼            ▼             ▼            ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │                    Per-Iteration Pipeline                       │
  │  1. Bandit selects arm → 2. Compile with strategy              │
  │  3. Reviewer edits → 4. Compute reward → 5. Update memory      │
  │  6. Update bandit → 7. Check for gaming                        │
  └─────────────────────────────────────────────────────────────────┘
```

### New Modules

| Module | File | Purpose |
|--------|------|---------|
| **EditSignalEngine** | `src/edit_signal.py` | Weighted composite reward: `R = 0.35*R_SED + 0.35*R_SEC + 0.15*R_PEND + 0.15*R_SAFE` |
| **SimulatedReviewer** | `src/simulated_reviewer.py` | 7 deterministic rules + LLM correction with fabrication guard |
| **CorrectionMemoryBank** | `src/correction_memory.py` | JSONL-backed persistent store with two-stage retrieval |
| **ContextualBandit** | `src/bandit.py` | UCB1 over 5 prompt strategy arms |
| **LearningOrchestrator** | `src/learning_loop.py` | Training loop + held-out evaluation + output artifacts |

### Reward Function

The composite reward combines 4 clinically meaningful sub-signals:

| Sub-signal | Weight | What it measures |
|-----------|--------|-----------------|
| **R_SED** | 0.35 | Normalized edit distance (lower edits → higher reward) |
| **R_SEC** | 0.35 | Section-level accuracy on 7 critical sections |
| **R_PEND** | 0.15 | Pending results coverage |
| **R_SAFE** | 0.15 | Safety flag preservation (binary: 0.0 or 1.0) |

**Safety Clamp:** If `R_SAFE = 0.0` (any escalation flag was dropped), the composite reward is clamped to `max(R, 0.10)` regardless of other sub-scores.

### Quick Start — Part 2

```bash
# Run the learning loop (10 iterations)
python run_learning.py --pdf "patient 2 (1).pdf"

# Custom iterations + output directory
python run_learning.py --pdf "patient 2 (1).pdf" --n-train 10 --output output/part2/

# Exploit-only mode (skip exploration)
python run_learning.py --pdf "patient 2 (1).pdf" --exploit-only

# Run Part 2 tests
python -m pytest tests/test_part2.py -v
```

### Output Artifacts

| File | Contents |
|------|----------|
| `training_curve.json` | Per-iteration metrics (arm, reward, sub-signals) |
| `before_after_report.md` | Baseline vs best arm comparison with metric tables |
| `improvement_curve.png` | Visual learning curve with arm color coding |
| `correction_memory_summary.md` | Top-10 correction patterns by impact |
| `limitations_analysis.md` | 5 failure modes with evidence from the run |
| `gaming_alerts.log` | Gaming detector alerts (may be empty) |
| `run_manifest.json` | Run ID, git hash, CLI args, timing |
| `reviewer_prompt_used.txt` | Version-controlled reviewer system prompt |

### Part 2 Limitations

See `output/part2/limitations_analysis.md` for a detailed analysis with evidence from the run. Key limitations:

1. **Cold-start problem** — First 5 iterations are pure exploration
2. **Single-patient overfitting** — All training on patient_2's clinical profile
3. **Metric gaming** — Vagueness could game R_SED (mitigated by GamingDetector)
4. **Reviewer fabrication** — LLM corrections guarded but not semantically verified
5. **Reward-safety tradeoff** — R_SAFE is binary, doesn't check flag content

### Before/After Metrics (Actual Run)

| Metric | Value |
|--------|-------|
| Iterations completed | 10 |
| Mean composite reward | 0.9934 |
| R_SED (edit distance) | 0.981 |
| R_SEC (section accuracy) | 1.000 |
| R_PEND (pending results) | 1.000 |
| R_SAFE (safety flags) | 1.000 |
| Safety clamps triggered | 0 |
| Fabrication blocks | 0 |
| Gaming alerts | 0 |
| Best arm | BASELINE |
| Improvement delta | +0.0000 |
| Runtime | 361.5s |

The flat reward curve reflects the high quality of Part 1's draft — the only consistent correction is REV-006 (allergies `[MISSING]` → `NOT KNOWN`). With a multi-patient training corpus, arm differentiation would emerge.

### What I'd Do With More Time (Part 2)

1. **Multi-patient training corpus**: Train across 10+ diverse patient profiles to break arm symmetry and demonstrate real learning differentiation.
2. **Exponential backoff for rate limits**: Add retry logic with exponential backoff to `_run_llm_correction_pass` to gracefully handle API quota limits mid-run.
3. **Semantic citation verification**: Cross-reference each LLM correction's `source_citation` against the actual page content to catch plausible but fabricated citations.
4. **Thompson Sampling alternative**: Implement Thompson Sampling alongside UCB1 and compare regret bounds on the same patient corpus.
5. **Sliding window bandit**: Use a sliding window over the last K iterations to handle non-stationary reward distributions as correction patterns evolve.
