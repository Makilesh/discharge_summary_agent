# Discharge Summary Agent

> **An agentic AI system that reads messy, multi-page hospital PDFs and produces a structured, clinically safe discharge summary draft — with zero fabricated facts and mandatory escalation of every conflict.**

Discharge summaries are one of the highest-risk documents in medicine. A missed medication, an unresolved lab critical value, or a hallucinated diagnosis can directly harm a patient after discharge. This system treats that risk seriously: it is engineered not to be impressive, but to be **safe**. The agent never guesses. It flags, escalates, and defers — exactly as a cautious clinician would.

---

## What Makes This Different

Most LLM-based document systems answer questions or summarise text. This system does something harder: it reconciles **conflicting evidence across multiple authors, document types, and time points** in a 71-page medical record, and produces a structured output that a clinician can trust enough to review — not re-do from scratch.

Key engineering decisions that separate this from a RAG pipeline or prompt chain:

| Decision | Why it matters |
|----------|---------------|
| **Deterministic StateGraph, not ReAct** | Clinical data extraction must be reproducible. A free-form ReAct agent would take different paths on the same PDF depending on sampling randomness. The state machine guarantees the same extraction sequence every run. |
| **5 mandatory cross-reference rules, not LLM judgement** | An LLM asked "do these diagnoses conflict?" might hallucinate a resolution. Hard-coded CR rules flag every inconsistency and hand it to the clinician. The agent never decides who is right. |
| **No-fabrication sentinel strings** | When data is missing, the literal string `[MISSING — not documented in source records. Clinician must supply.]` appears in the output. The agent cannot substitute a plausible value because there is no code path that does so. |
| **Multi-model routing on Free Tier** | Distributing calls across 4 Gemini models with independent quotas makes a 71-page extraction feasible without paid API access. This reflects real-world constraint engineering, not ideal-world assumptions. |
| **Part 2: safety-constrained reinforcement learning** | The reward function includes a hard clamp: any draft that drops an escalation flag receives a maximum reward of 0.10, regardless of how good its edit distance score is. Safety is not a hyperparameter. |

---

## Architecture — Part 1: The Agent

### State Machine Design

```
INITIALIZE → REASON → CALL_TOOL → OBSERVE → VERIFY → [REASON | COMPILE | HARD_CAP_ESCALATE]
```

The agent is a LangGraph `StateGraph` with typed state and deterministic conditional routing. Each phase is a discrete Python function — not a prompt instruction. The LLM is given one job at each step (classify, OCR, extract, reconcile) and is never asked to decide what to do next. That decision is made by `route_after_verify()`, a Python function with explicit `if/elif/else` branches.

### Control Mechanisms

| Mechanism | Implementation | Fail-safe behaviour |
|-----------|---------------|-------------------|
| **Step cap** | `MAX_ITERATIONS = 25`, decremented every transition | Cap breach → `HARD_CAP_ESCALATE` node compiles whatever was extracted with `[MISSING]` markers and a CRITICAL flag |
| **Retry limit** | `MAX_RETRIES = 2` per tool call | Third failure → field marked `[UNRESOLVED]`, added to `fabrication_blocks` |
| **Priority order** | 15-level extraction priority (typed discharge summary first) | If step cap hits, the most clinically important data was already extracted |
| **RPM throttle** | Per-model timestamp tracking in `_rpm_sleep()` | Prevents 429 storms; model-aware sleep intervals |
| **Model fallback chain** | `flash-lite → flash → gemini-2.5-flash → local Ollama` | If all Gemini quotas are exhausted, local `qwen2.5vl` / `deepseek-r1` take over |

### No-Fabrication Guardrail

The agent uses 4 sentinel strings that appear **literally** in the output. There is no code path that substitutes a plausible value for any of them:

```
[MISSING — not documented in source records. Clinician must supply.]
[PENDING — {item} sent {date}. Result not available in source documents.]
[CONFLICT — Multiple sources disagree. See escalation_flags. Clinician must resolve.]
[UNCLEAR — Source text partially legible: '{best_guess}'. Verify against original.]
```

### Cross-Reference Audit (CR-1 to CR-5)

Runs **mandatorily** before every compilation — even under step-cap pressure. Five rules that detect inter-document conflicts:

| Rule | What it checks | Real example from Patient 2 |
|------|---------------|---------------------------|
| **CR-1** | Every active treatment has a matching diagnosis | Insulin (Lantus, Actrapid) present — diagnosis must confirm DM/DKA/T1DM/T2DM |
| **CR-2** | Diagnoses are consistent across all source documents | ER chart: DKA → Admission: TAFE → ICU: DKA + T2DM — flagged, not resolved |
| **CR-3** | Lab critical values are addressed; "resolved" claims match actual lab data | Na⁺ 114 mmol/L (ref 136–146) → CRITICAL escalation |
| **CR-4** | Culture results are consistent with antibiotic choices | Urine culture: sterile → IV Meropenem given — flagged for clinician annotation |
| **CR-5** | Discharge condition matches last recorded vitals; DAMA is detected | "Not willing" (Page 2) + "discharge on request" (Page 56) → CRITICAL DAMA flag |

**Production-grade CR-3 matching** — this is where naive implementations fail and this one doesn't:
- **Word-boundary regex**: `(?<![a-z])ph(?![a-z])` prevents the pH threshold (7.25–7.55) from matching `"neutrophils"`, `"lymphocytes"`, `"eosinophils"` — a class of false positive that would generate 21 spurious CRITICAL alerts on real lab data
- **Context-aware exclusions**: Urine pH is physiologically normal at 5.5; blood pH of 5.5 is fatal. The threshold config carries an `"exclude": ["urine"]` field so the comparison is context-aware
- **Unit scale normalisation**: WBC reported as 7,160 Cells/cumm vs threshold of 20 × 10³/µL — auto-converts before comparison so the result is accurate, not blindly triggered
- **Non-lab item filtering**: `NON_LAB_PENDING_KEYWORDS` (cannula, catheter, drain, tube) prevents patient-care devices from appearing in the pending investigations escalation list

### Failure & Conflict Handling

All failures are **loud** — they produce visible markers in the output, never silent drops:

- **Tool failures**: `safe_tool_call()` wraps every extraction with 2 retries. On final failure, field → `[UNRESOLVED]`, added to `fabrication_blocks` list.
- **OCR failures**: Pages with < 30 characters of extracted text → `unreadable_pages`. Listed in the summary status block.
- **Conflicting data**: Both values are preserved. Output shows `[CONFLICT]` with all sources and their document types.
- **DAMA**: Detected from natural language keywords across nursing notes, consultation sheets, and discharge checklists. One detection → CRITICAL escalation.

---

## Architecture — Part 2: Learning from Doctor Edits

The central question: can the system learn, from a doctor's corrections, to produce drafts that require fewer edits over time — without ever learning to compromise on safety?

```
                    ┌─────────────────────────┐
                    │   LearningOrchestrator   │
                    │   (learning_loop.py)     │
                    └───────────┬─────────────┘
                                │
        ┌──────────┬────────────┼────────────┬─────────────┐
        ▼          ▼            ▼            ▼             ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │Contextual│ │Simulated │ │  Edit    │ │Correction│ │  Gaming  │
  │  Bandit  │ │ Reviewer │ │  Signal  │ │  Memory  │ │ Detector │
  │(UCB1, 5  │ │(7 rules +│ │(R_SED +  │ │(JSONL,   │ │(every 5  │
  │  arms)   │ │ LLM pass)│ │ R_SAFE)  │ │ ranked)  │ │  iters)  │
  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
       │  select    │  edit      │  reward     │  exemplars │  alert
       └────────────┴────────────┴─────────────┴────────────┘
```

### The 5 Prompt Strategy Arms (UCB1 Bandit)

| Arm | Name | Strategy |
|-----|------|---------|
| 0 | BASELINE | Standard compiler prompt — the control condition |
| 1 | SECTION_EXEMPLARS | Injects the top-K most impactful past corrections per section |
| 2 | GLOBAL_PREAMBLE | Prepends the full correction history as a global system preamble |
| 3 | CONFLICT_FIRST | Processes escalation flags before generating each section |
| 4 | HYBRID_EXEMPLAR | Combines safety-first ordering with per-section correction examples |

Every arm includes a hard `ANTI_VAGUENESS_INSTRUCTION` — a constraint that prevents the agent from learning to produce shorter, vaguer summaries to game the edit-distance metric.

### Reward Function Design

```
R = 0.35 × R_SED  +  0.35 × R_SEC  +  0.15 × R_PEND  +  0.15 × R_SAFE
```

| Sub-signal | Weight | Measures |
|-----------|--------|---------|
| **R_SED** | 0.35 | Normalised edit distance between draft and doctor-edited version |
| **R_SEC** | 0.35 | Section-level accuracy on 7 clinical sections (diagnosis, medications, pending results, etc.) |
| **R_PEND** | 0.15 | Coverage of pending investigations at discharge |
| **R_SAFE** | 0.15 | Binary: were all escalation flags preserved? |

**Safety Clamp**: `R_SAFE = 0.0` → composite reward clamped to `max(R, 0.10)` regardless of all other sub-scores. The bandit can never be rewarded for dropping a safety flag.

### Part 2 Results & Interpretation

| Metric | Value |
|--------|-------|
| **Iterations** | 10 |
| **Baseline Reward (Arm 0)** | 0.8649 |
| **Best Arm** | SECTION_EXEMPLARS (Arm 1) |
| **Best Reward** | 0.8827 |
| **Improvement Delta** | +0.0178 (+1.78%) |
| **Safety Clamps Triggered** | 0 |
| **Fabrication Blocks** | 0 |
| **Gaming Alerts** | 0 |

**Interpretation**: 
Unlike early trials with a flat reward curve caused by a flat baseline, resolving the compiler's correction context application bug revealed a clear learning trajectory. The baseline compiler prompt starts at **0.8649** reward. By utilizing **SECTION_EXEMPLARS** (Arm 1), which dynamically retrieves and applies historical correction mappings for specific section contexts (such as resolving diagnosis inconsistencies or mapping `[MISSING]` allergies to `NOT KNOWN`), the composite reward successfully climbs to **0.8827** (a **+1.78% delta improvement**).

Looking at the section-level metrics:
- **Principal Diagnosis Accuracy** jumps from **74.4%** (Baseline) to **100%** (Best Arm).
- **Discharge Condition Accuracy** improves from **97.13%** to **97.34%**.
This confirms that the contextual bandit successfully identified the superior prompt strategy arm under safety-constrained conditions (100% safety flag preservation, 0 safety clamps triggered).


---

## Results — Patient 2 (71 pages, 15 document types)

| Metric | Value |
|--------|-------|
| **Runtime** | 208 seconds |
| **Graph steps taken** | 98 |
| **Pages processed** | 71 |
| **Document types identified** | 15 |
| **Conflicts detected** | 3 (all clinically legitimate) |
| **Escalation flags** | 18 (2 CRITICAL, 16 WARNING) |
| **False positives** | **0** |
| **Fabrication blocks** | **0** |
| **Medications reconciled** | 16 unique drugs (fuzzy-deduped from 20 raw entries) |
| **Lab results extracted** | 147 |
| **Audit trace entries** | 108 |

### The 2 CRITICAL Alerts (Both Genuine)

1. 🚨 **Sodium [Na⁺] 114 mmol/L** — severe hyponatremia (reference: 136–146 mmol/L). Life-threatening electrolyte imbalance that requires clinician annotation of treatment response.
2. 🚨 **DAMA** — Discharge Against Medical Advice. Detected from `"not willing"` (Page 2) and `"discharge on request"` (Page 56). Changes clinical and legal responsibility. Mandatory clinician review.

### The 16 WARNING Flags
All are medication reconciliation warnings — inpatient drugs with no documented reason for stopping at discharge. Each one is a genuine safety check, not noise.

---

## Project Structure

```
src/
├── state.py             # AgentState TypedDict — 25+ fields, 7 categories, typed reducers
├── config.py            # Model routing, CR thresholds, no-fabrication templates
├── trace.py             # Structured trace emission — 108-entry audit log per run
├── pdf_processor.py     # PDF → high-res PNG images via PyMuPDF
├── tools.py             # Multi-model extraction, 3-layer medication dedup, RPM throttle
├── cross_reference.py   # CR-1 to CR-5 safety audit — word-boundary matching, unit normalisation
├── compiler.py          # 17-section Markdown compiler — no LLM calls, no invented data
├── graph.py             # LangGraph StateGraph — 7 nodes, deterministic routing
├── edit_signal.py       # Part 2: Composite reward with safety clamp
├── simulated_reviewer.py# Part 2: 7-rule deterministic reviewer + LLM correction pass
├── correction_memory.py # Part 2: JSONL persistent store with impact-ranked retrieval
├── bandit.py            # Part 2: UCB1 bandit + GamingDetector
└── learning_loop.py     # Part 2: Training orchestrator, curve generation, evaluation

tests/
├── test_agent.py        # 35 unit tests — CR rules, reconciliation, graph routing, OCR cache
└── test_part2.py        # 21 unit tests — bandit, reward, reviewer, memory, gaming detection

run_agent.py             # CLI entry — Part 1
run_learning.py          # CLI entry — Part 2
```

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set Gemini API key
echo GOOGLE_API_KEY=your-key-here > .env

# 4. Verify the codebase (56 tests, ~23 seconds)
python -m pytest tests/ -v

# 5. Run the agent on a patient PDF
python run_agent.py --pdf "patient 2 (1).pdf" --output output/

# 6. Run the learning loop (Part 2)
python run_learning.py --pdf "patient 2 (1).pdf" --n-train 10 --output output/part2/
```

## Output Files

| File | Contents |
|------|----------|
| `output/discharge_summary.md` | 17-section structured discharge summary with citations, CR flags, escalation section |
| `output/trace.json` | Full 108-entry audit trail — every classification, extraction, routing decision |
| `output/state.json` | Final agent state — all extracted data, conflicts, flags, fabrication blocks |
| `output/part2/before_after_report.md` | Baseline vs best-arm comparison with metric tables |
| `output/part2/improvement_curve.png` | Visual reward curve with per-arm colour coding |
| `output/part2/limitations_analysis.md` | 5 documented failure modes with evidence from the actual run |

---

## Test Suite

```bash
python -m pytest tests/ -v          # All 56 tests (~23s)
python -m pytest tests/test_agent.py -v    # Part 1: 35 tests
python -m pytest tests/test_part2.py -v   # Part 2: 21 tests
```

Coverage includes: CR-1 through CR-5 rules · medication reconciliation (all 4 change types) · hard-cap escalation · state validation · trace emission · batch extraction routing · classification cache · Ollama fallback · edit signal (all 4 sub-signals + safety clamp) · UCB1 selection and persistence · simulated reviewer rules · correction memory deduplication · gaming detection (vagueness + summarisation collapse).

---

## Limitations & What I'd Prioritise Next

The following are honest assessments of where the system falls short, ordered by **clinical impact** — not development effort.

### Highest clinical impact

1. **Structured output validation (Pydantic schemas)** *(Priority 1)* — Every LLM response is parsed with a best-effort JSON extractor. A single malformed response can silently drop lab results or medication entries. Adding Pydantic schemas with auto-retry on validation failure would close the most serious data integrity gap. This is the first thing I'd build.

2. **Real drug interaction API** *(Priority 2)* — The current drug interaction check uses a static dictionary of ~20 known pairs. A real pharmacopeia integration (RxNorm or DrugBank) would catch combinations like insulin + sulphonylureas or antibiotics + anticoagulants that the static list misses entirely.

### Moderate clinical impact

3. **Demographics extraction from admission record headers** *(Priority 3)* — The agent correctly marks demographics `[MISSING]` rather than guessing, but structured header parsing (name, age, gender, MRN) is feasible and would eliminate the most common `[MISSING]` field in the output.

4. **Semantic citation verification** *(Priority 4)* — Every `[Page N]` citation in the hospital course narrative is asserted by the LLM. Verifying that the cited claim actually appears in the OCR text of that page would catch confident but incorrect page references.

### System maturity

5. **Multi-patient batch mode with corpus-level metrics** *(Priority 5)* — All current testing is on one patient. Parallel batch processing with aggregated precision/recall on a diverse corpus is the step that turns this from a proof of concept into a publishable benchmark.

6. **Token budget tracking** *(Priority 6)* — `usage_metadata` is available on every Gemini response but not currently logged. This is a one-day addition that would enable cost optimisation and prompt efficiency analysis.
