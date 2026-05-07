# Reply Mirror — Adaptive Fraud Detection System

**Date:** 2026-04-16  
**Competition:** Reply AI Agent Challenge — "Reply Mirror"  
**Task:** Binary classify every transaction in each dataset as fraudulent (1) or legitimate (0).

---

## 1. Problem Context

Fraud patterns evolve across 5 dataset levels of increasing complexity. Datasets have no ground-truth labels at runtime — the LLM must reason from first principles like a human investigator, not classify from learned examples.

Each dataset provides:
- `transactions.csv` — financial transaction records
- `users.json` — citizen profiles (age, salary, job, location, lifestyle description)
- `sms.json` — SMS conversation threads per user
- `mails.json` — email threads per user
- `locations.json` — GPS pings (BioTag, timestamp, lat/lng)
- `audio/*.mp3` — voice recordings (present from level 3 onward)

Scoring dimensions: detection accuracy (count-based + economic value-weighted), cost, latency, agent architecture quality.

**Hard rule:** The system must be LLM-centred at every stage. No fallback to deterministic heuristics if LLM calls fail — retry with same LLM strategy (smaller batches, stricter prompts).

---

## 2. Architecture

```
Raw data (CSV, JSON, MP3)
        │
        ▼
┌─────────────────────────────────┐
│  Phase 0: Pre-processing        │  ← Zero LLM cost
│  0a. Statistical signals        │    pandas
│  0b. Audio transcription        │    gemini-2.5-flash-lite (per MP3)
└──────────────┬──────────────────┘
               │
               ▼
   ┌────────────────────┐
   │ Communications     │  ← 1 call per user (7-12 total)
   │ Agent              │    qwen3-32b
   │ → user risk        │
   │   profile cached   │
   └────────┬───────────┘
            │
            ▼
   ┌─────────────────────────────────────────────┐
   │  Dynamic Orchestrator (strands Agent)       │
   │                                             │
   │  Receives: statistical signals per txn +    │
   │            cached user risk profiles        │
   │            threat intelligence (if level>1) │
   │                                             │
   │  Tools available (called selectively):      │
   │  ┌────────────────────────────────────┐     │
   │  │ @tool analyze_financial_patterns   │ ←Transaction Agent
   │  │ @tool analyze_recipient_network    │ ←NEW: dataset-level IBAN graph
   │  │ @tool get_location_check           │ ←GPS coherence
   │  │ @tool get_comms_detail             │ ←cached user profile
   │  └────────────────────────────────────┘     │
   │                                             │
   │  Low suspicion  → classify directly         │
   │  Uncertain      → call relevant tools       │
   │  High suspicion → call multiple tools       │
   └─────────────────┬───────────────────────────┘
                     │
                     ▼
            output/level_N.txt
                     │
                     ▼
   ┌────────────────────┐
   │  Memory Agent      │  ← Runs ONCE after level N completes
   │  Synthesises fraud │    Saves threat intelligence brief to disk
   │  patterns found    │    Loaded by orchestrator at level N+1 start
   └────────────────────┘
```

**Why dynamic tool use:** The problem scoring rewards "agent architecture quality" and explicitly requires the LLM to decide which tools to call and when. A static pipeline treats all transactions identically; a dynamic orchestrator reasons about what evidence it needs per case. It also handles dataset drift naturally: novel fraud patterns trigger the tools that make sense rather than a hardcoded sequence.

**Why a Memory Agent:** The problem statement explicitly states *"anticipating new attack patterns by leveraging the memory of past interactions."* Fraudster tactics evolve across levels. The Memory Agent converts completed-level findings into a threat intelligence brief that the orchestrator uses as prior knowledge at the next level — directly addressing this scoring dimension.

---

## 3. Phase 0a — Statistical Signals (pandas, no LLM)

Computed once per dataset, stored as dicts keyed by `transaction_id`.

**Per-user baseline** (computed across all transactions for that user):
- Mean and std of transaction amounts
- Typical active hours (hour-of-day distribution)
- Typical transaction types (frequency distribution)
- Usual recipient IBANs / recipient IDs
- Usual location (city from GPS pings)

**Per-transaction anomaly signals:**
- `amount_zscore`: z-score of amount vs user's own baseline
- `hour_deviation`: boolean — transaction at unusual hour for this user
- `balance_trajectory`: running balance trend (approaching zero = stress signal)
- `new_recipient`: boolean — first time transacting with this recipient
- `type_shift`: boolean — unusual transaction type for this user
- `location_coherence`: for in-person payments, was a GPS ping near the merchant location within ±2 hours?
- `velocity_spike`: transactions per day vs user's rolling average

**Dataset-level graph signals** (computed across all users):
- `shared_iban_count`: how many distinct senders have sent to this recipient IBAN — high count = potential mule
- `circular_transfer`: boolean — detect A→B→A or A→B→C→A patterns within 24-hour windows

These signals are formatted into compact text summaries (~300 tokens per transaction) for the LLM agents. The LLM never receives raw CSV rows.

---

## 4. Phase 0b — Audio Transcription

**Trigger:** Only if `audio/` directory exists in dataset folder.

**Model:** `google/gemini-2.5-flash-lite` — multimodal, $0.10/M text + $0.30/M audio tokens.

**Process:**
- Transcribe each MP3 file individually (one API call per file)
- Cache transcript as `audio/<filename>.txt` — never re-transcribe on retry
- Extract speaker name from filename convention: `YYYYMMDD_HHMMSS-<name>.mp3`
- Transcripts passed to the Communications Agent alongside SMS/email

---

## 5. Phase 1 — Communications Agent (per user)

**Purpose:** Build a user-level intelligence profile from all communications. Runs once per user — 7–12 calls total per dataset.

**Input per call:**
- User profile (age, salary, job, lifestyle description)
- All SMS threads for this user
- All email threads for this user
- All audio transcripts attributed to this user (matched by name from filename)

**Task:** Identify social engineering indicators, coercion, fraud recruitment, impersonation, and unusual urgency patterns. Reason about whether the user appears to be a victim, a perpetrator, or acting normally given their profile.

**Output:** Structured user risk profile (~200 words):
```
USER_RISK_PROFILE: <user_id>
RISK_LEVEL: low/medium/high
RED_FLAGS: [list of specific signals found, or "none"]
VICTIM_INDICATORS: [signs the user may be coerced or deceived]
PERPETRATOR_INDICATORS: [signs the user may be initiating fraud]
SUMMARY: one-paragraph narrative
```

**Model:**
- Dev: `qwen/qwen3-32b` ($0.08/M)
- Eval: `qwen/qwq-32b` ($0.15/M)

**Batch unit:** 1 user per call (not batched — each user's comms are processed holistically).

---

## 6. Transaction Agent (on-demand tool)

**Purpose:** Deep financial pattern reasoning for a targeted subset of transactions. Called by the orchestrator via `@tool analyze_financial_patterns()` only when the orchestrator decides more evidence is needed.

**Input:** Statistical signal summaries for the requested transaction IDs. Each summary includes the user's baseline stats and per-transaction anomaly scores — no raw CSV fields.

Example input entry:
```
TXN e1021ab7: user=FCHN-VTSI (86yo retired, salary=35200)
  amount=2252.22 (z=+0.3, normal), type=transfer (usual for user)
  hour=08:17 (within normal window), recipient=DOM58085 (seen before)
  balance_after=28548 (healthy), location_coherence=N/A
  new_recipient=False, velocity=1.2x avg (normal)
```

**Task:** For each requested transaction, reason about whether the statistical deviations are consistent with a legitimate explanation given the user's known profile, or indicative of fraud.

**Output:** One line per transaction:
```
TXN_ID: RISK_SCORE (0.0-1.0) | RATIONALE (one sentence)
```

**Model:**
- Dev: `qwen/qwen3-30b-a3b` ($0.08/M, 41K native context)
- Eval: `qwen/qwq-32b` ($0.15/M)

**Batch size within tool call:** up to 50 transactions. On context overflow (400): halve and retry with same model. No deterministic fallbacks.

**Cost note:** Because the orchestrator calls this selectively, not all transactions incur Transaction Agent cost. Low-suspicion transactions are classified by the orchestrator directly from statistical signals.

---

## 7. Network Analysis Tool (on-demand tool, dataset-level)

**Purpose:** Expose dataset-wide recipient patterns to the orchestrator. A mule account is invisible at the per-user level — it reveals itself only when multiple unrelated senders are all sending to the same IBAN.

**Tool signature:**
```python
@tool
def analyze_recipient_network(recipient_iban: str) -> str:
    """Analyse all transactions in the current dataset that share this recipient IBAN.
    Identifies whether the recipient looks like a merchant, a salary source, or a
    collection point for funds from multiple unrelated senders (mule indicator)."""
```

**Implementation:** Uses the pre-computed `shared_iban_count` signal from Phase 0a plus the full list of senders to that IBAN. Passes this compact summary to a lightweight LLM call that reasons about the network pattern.

**When the orchestrator calls it:** When it encounters a `new_recipient=True` or `shared_iban_count > 2` signal, it calls this tool to understand whether the recipient is suspicious at a dataset level — not just suspicious to this one user.

**Model:** Same as Transaction Agent (cheap, narrow task).

---

## 8. Dynamic Orchestrator

**Purpose:** The directing intelligence. Receives statistical signals and cached user profiles (plus threat intelligence from completed levels), then decides at runtime which tools to invoke before making each final binary classification.

**Implementation:** A strands `Agent` with `@tool` decorated helper functions. The LLM decides whether to call tools based on what it observes in the evidence.

**Full tool set:**

```python
@tool
def analyze_financial_patterns(txn_ids: list[str], user_id: str) -> str:
    """Run Transaction Agent on a specific set of transactions.
    Returns per-transaction risk scores and rationales."""

@tool
def analyze_recipient_network(recipient_iban: str) -> str:
    """Analyse all transactions sharing this recipient IBAN across the dataset.
    Returns a mule/merchant/normal classification with evidence."""

@tool
def get_location_check(user_id: str, txn_id: str, timestamp: str, location: str) -> str:
    """Check GPS pings for user around transaction timestamp.
    Returns coherence assessment: was user physically near the transaction location?"""

@tool
def get_comms_detail(user_id: str) -> str:
    """Retrieve the cached Communications Agent risk profile for this user."""
```

**Decision logic (LLM-driven, not hardcoded):**
- **Low suspicion** (clean z-scores, known recipients, matching profile): classify directly — fast and cheap
- **Uncertain**: call `analyze_financial_patterns()` or `analyze_recipient_network()` as appropriate
- **High suspicion or conflicting signals**: call multiple tools, gather full evidence, then classify

**Orchestrator system prompt principles:**
- **Economic weighting (critical):** Scale intervention threshold by transaction amount. A €50,000 uncertain transfer → lean fraud. A €5 uncertain coffee → lean legitimate. Missing high-value fraud carries far greater scoring cost than false-positiving a small transaction.
- **Convergence across signals:** Financial risk + comms risk agreeing = high confidence. Diverging signals → call more tools before deciding.
- **Profile-relative reasoning:** An 86-year-old retiree wiring money at 3am to a new foreign IBAN is alarming. A 30-year-old tech worker doing the same may not be. Always reason relative to the user's known lifestyle.
- **Threat intelligence awareness:** If prior-level patterns are provided, actively look for evolved versions of those tactics in the current data.
- **Intervention bias:** When genuinely uncertain after calling relevant tools, classify as fraud (1) — false negatives carry higher cost weight in scoring.

**Output:** One line per transaction:
```
TXN_ID: CLASSIFICATION (0 or 1) | CONFIDENCE (high/medium/low) | REASON (one sentence)
```

**Model:**
- Dev: `deepseek/deepseek-v3.2` ($0.26/M in, $0.38/M out)
- Eval: `google/gemini-2.5-flash` ($0.30/M in, $2.50/M out)

**Batch input size:** 20–25 transactions per orchestrator call. Tool calls within a batch are targeted at specific transaction subsets — the full batch is never re-run for a single uncertain transaction.

**On context overflow:** halve batch and retry with same model. On malformed output: re-prompt once with stricter format. No deterministic fallbacks.

---

## 9. Memory Agent (post-level, cross-level learning)

**Purpose:** Synthesise fraud patterns discovered in a completed level into a reusable threat intelligence brief. Directly addresses the problem statement's emphasis on *"leveraging the memory of past interactions."*

**Trigger:** Runs once after `output/level_N.txt` is written, before level N+1 begins.

**Input:**
- The full set of transactions classified as fraud in level N (with their orchestrator reasoning)
- The statistical signals for those transactions
- The user risk profiles involved

**Task:** Identify recurring tactics, techniques, and patterns across the detected frauds. What types of transactions were targeted? What user profiles were victimised or recruited? What communication patterns appeared? What new tactics should we watch for as they evolve?

**Output:** A threat intelligence brief (~400 words) saved to `memory/level_N_intel.md`:
```
THREAT_INTEL: Level N → Level N+1
FRAUD_PATTERNS_FOUND: [list of observed tactics]
HIGH_RISK_SIGNALS: [which signals were most predictive]
VICTIM_PROFILE: [what types of users were targeted]
EVOLVING_TACTICS_TO_WATCH: [what the orchestrator should look for next level]
NARRATIVE: paragraph summary
```

**How it's used:** At level N+1 start, `memory/level_N_intel.md` is loaded and prepended to the orchestrator's system prompt as prior knowledge. The orchestrator is instructed to actively look for evolved versions of the listed tactics.

**Model:** Same as orchestrator (one call — worth spending on quality).

**Cost:** One orchestrator-class LLM call per level transition. Negligible relative to per-transaction processing cost.

---

## 10. Robustness — LLM-Only Retry Policy

`call_with_retry(agent, prompt, expected_ids, max_retries=3)` handles all failure modes.

| Failure | Recovery | Forbidden |
|---------|----------|-----------|
| Context overflow (400) | Halve batch, retry both halves | Fall back to heuristics |
| Rate limit (429) | Exponential backoff + jitter, respect Retry-After | Skip transactions |
| Transient error (5xx/timeout) | Retry up to 3× with backoff | Mark as safe by default |
| Malformed output | Re-prompt once with stricter format, then halve | Parse partial output and guess |
| Missing transaction IDs | Re-run only missing IDs as mini-batch | Drop them |

`validate_output(response_text, expected_ids)` checks completeness after every call. Missing IDs trigger a mini-batch retry automatically. All retries logged to Langfuse.

---

## 11. Output Format

Plain ASCII `.txt` file, one fraudulent transaction UUID per line:

```
e1021ab7-c2de-4791-994b-bab86e6fbe3e
8830a720-ff34-4dce-a578-e5b8006b2976
```

Written to `output/level_<N>_output.txt`. Langfuse session ID printed to stdout for submission.

**Validity guards (checked before writing):**
- Not empty (no transactions reported = invalid)
- Not all transactions (all reported = invalid)
- If fewer than 5% of total transactions are flagged, log a warning (likely too conservative)

---

## 12. Module Structure

```
src/
  analysis.py        # Phase 0a: compute_statistical_signals(), format_txn_summary()
  transcription.py   # Phase 0b: transcribe_audio_files()
  sub_agents.py      # Comms Agent, Transaction Agent (called as tools)
  network.py         # Network Analysis Tool: analyze_recipient_network()
  orchestrator.py    # Dynamic Orchestrator: run_orchestrator(), parse_classifications()
  memory.py          # Memory Agent: run_memory_agent(), load_threat_intel()
  utils.py           # make_model(), call_with_retry(), validate_output(), estimate_tokens()
memory/              # Threat intelligence briefs: level_N_intel.md
config.py            # All tunable constants
main.py              # Pipeline: run_level(), main()
```

---

## 13. Config Constants

```python
# --- Model IDs (swap for final eval runs) ---
MODEL_TRANSCRIPTION      = "google/gemini-2.5-flash-lite"
MODEL_TXN_AGENT          = "qwen/qwen3-30b-a3b"       # dev
MODEL_COMMS_AGENT        = "qwen/qwen3-32b"            # dev
MODEL_ORCHESTRATOR       = "deepseek/deepseek-v3.2"    # dev
MODEL_MEMORY_AGENT       = "deepseek/deepseek-v3.2"    # same as orchestrator

# --- Batch sizes (halved automatically on 400) ---
BATCH_SIZE_TXN_AGENT     = 50
BATCH_SIZE_ORCHESTRATOR  = 20

# --- Retry policy ---
MAX_RETRIES              = 3

# --- Token estimation ---
TOKENS_PER_TXN_INPUT     = 300   # sub-agent
TOKENS_PER_TXN_OUTPUT    = 70    # sub-agent
TOKENS_PER_ORCH_INPUT    = 400   # orchestrator (denser payload)
```

---

## 14. Dataset Folder Convention & Level Mapping

```python
LEVEL_FOLDERS: dict[int, str] = {
    1: "The Truman Show - train",    # 80 txns, no audio
    2: "Brave New World - train",    # 522 txns, no audio
    3: "Deus Ex - train",            # 2017 txns, audio present
    # Levels 4-5 added when unlocked
}
```

CLI usage: `python main.py 1` or `python main.py all`

---

## 15. Transaction ID Format

Transaction IDs are full UUIDs: `e1021ab7-c2de-4791-994b-bab86e6fbe3e`

All output parsing uses:
```python
UUID_PATTERN = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
```
Defined once in `utils.py`, imported everywhere. Already corrected from the legacy `[A-Z]{8}` placeholder.

---

## 16. Scalability to Levels 4-5

The architecture is additive. New data modalities in levels 4-5:
1. Add `compute_<domain>_signals()` in `analysis.py`
2. Add a new `@tool` function in the orchestrator for that domain
3. The orchestrator discovers the tool and decides when to use it
4. Memory Agent automatically incorporates new signal types in its synthesis

Model upgrades for eval: swap `MODEL_*` constants in `config.py`. No code changes.
