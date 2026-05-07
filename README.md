# MirrorPay Fraud Detection — Multi-Agent System

> **A multi-agent LLM pipeline for financial fraud detection. A tool-using orchestrator coordinates five specialised LLM agents over OpenRouter, with statistical pre-processing, multimodal audio reasoning, retry/robustness layers, and end-to-end tracing through Langfuse.**

Built for the **Reply AI Agent Challenge 2026** (team **ICL_Hotdogs**).

---

## About this repository

This is a **sanitised public mirror** of the team submission. The original development repository is private. This mirror exists to make the architecture, agent design, and engineering approach visible as a portfolio piece.

**Not included in this mirror (intentionally):**

- `data/` — the hackathon dataset (transactions, audio recordings, SMS, emails, GPS pings) is Reply IP and not redistributable.
- `output/` and `memory/` — derived artefacts from running the pipeline against the hackathon dataset.
- Reply-provided reference materials (problem statement PDF, Langfuse template).
- `.env` — replaced with `.env.example`.

The code, agent prompts, orchestrator logic, retry/robustness layer, and design documents are all preserved.

---

## Tech Stack

**Core:** Python 3.11 · `strands-agents` · `openai` (OpenRouter routing)
**LLMs:** DeepSeek v3.2 · Qwen3 32B / 30B-a3b · Google Gemini 2.5 Flash Lite
**Observability:** Langfuse (session-level tracing, cost / token tracking)
**Data:** Pandas · NumPy · `ulid-py`
**Testing:** pytest

---

## Engineering Highlights

- **Tool-using orchestrator pattern.** A single LLM agent (DeepSeek v3.2) decides per-transaction which of five tools to call: `analyze_financial_patterns`, `analyze_recipient_network`, `get_location_check`, `get_comms_detail`, `analyze_identity`. The orchestrator drives the reasoning; sub-agents and deterministic helpers do narrow work and return structured signals.
- **Statistical pre-processing layer (zero LLM cost).** Per-user baselines, z-scores, hour anomaly detection, GPS / location-coherence checks, recipient IBAN network graphs, and salary/rent pattern detection are computed in pandas before any LLM call. The orchestrator then reasons over a compact signal summary instead of raw transactions, sharply reducing tokens and cost.
- **Cross-level memory.** A Memory agent runs at the end of each dataset level and synthesises observed fraud tactics into a concise threat intelligence brief. The brief is injected into the next level's orchestrator system prompt, so evolved variants of earlier patterns (e.g. updated phishing domains, new mule-account topologies) are easier to catch.
- **Multimodal audio reasoning.** Voice recordings (provided from level 3 onward) are transcribed once via Gemini 2.5 Flash Lite per speaker, cached to disk, and consumed by the Communications agent alongside SMS and email threads. The Communications agent flags social engineering, phishing, OTP coercion, and temporally aligns suspicious comms to specific transactions.
- **Production-grade robustness layer.** A generic `call_with_retry` wrapper handles context overflow (400) by halving batch size, rate limits (429) with exponential backoff and jitter, transient 5xx errors with capped retries, and malformed model output with lenient parsing or stricter re-prompting. Every retry and failure is logged to Langfuse.
- **Output validation.** Each batch response is validated by ID coverage — any dropped transaction is detected and re-run as a mini-batch, so no transaction silently disappears from the final classification.
- **Cost-aware model selection.** Cheaper models (Qwen 30B MoE, Gemini Flash Lite) handle high-volume narrow tasks; the strongest model (DeepSeek v3.2) is reserved for the orchestrator's final arbitration and the Memory synthesis step.

See [`docs/design.md`](docs/design.md) for the full pre-build design spec and [`docs/MODEL_WHITELIST.md`](docs/MODEL_WHITELIST.md) for the allowed model list with cost rationale.

---

## Architecture Overview

```
transactions.csv + users.json + sms.json + mails.json + audio/
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 0a — Statistical Pre-Processing  (zero LLM cost)     │
│                                                             │
│  ┌────────────────┐   ┌────────────────┐   ┌────────────┐  │
│  │ User Baselines │   │  Graph Signals │   │ Salary /   │  │
│  │ (mean, stddev, │   │ (recipient IBAN│   │ Rent Signals│  │
│  │  tx frequency) │   │  network graph)│   │ (multi-IBAN,│  │
│  └────────────────┘   └────────────────┘   │ dup month, │  │
│                                            │ city mismatch│ │
│  ┌────────────────────────────────────┐    └────────────┘  │
│  │ Statistical Signals per transaction│                    │
│  │ (z-score, hour anomaly, new recip, │                    │
│  │  location coherence, GPS pings)    │                    │
│  └────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 0b — Audio Transcription  (cached to disk)           │
│                                                             │
│  Audio files → Gemini 2.5 Flash Lite → transcript text     │
│  (one call per speaker, result saved so re-runs are free)   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 1 — Communications Agent  (one call per user)        │
│                                                             │
│  Input:  user profile + all SMS/email/audio for that user   │
│          + list of their transactions with timestamps        │
│                                                             │
│  Reasoning:                                                 │
│  • Social engineering (urgency, pressure, impersonation)    │
│  • Phishing URLs / lookalike domains (paypa1-secure.net)    │
│  • Fraud recruitment / OTP theft / coercion                 │
│  • Temporal matching: suspicious comms BEFORE each tx       │
│                                                             │
│  Output: one TXN_COMMS_RISK block per transaction           │
│    TXN_COMMS_RISK: <uuid>                                   │
│    PRE_TXN_FLAGS: yes | phishing link 3 days before tx      │
│    DAYS_BEFORE: 3                                           │
│    RISK_LEVEL: high                                         │
│    REASON: ...                                              │
│    ---                                                      │
│                                                             │
│  Model: qwen/qwen3-32b                                      │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 2 — Dynamic Orchestrator  (batched, tool-using)      │
│                                                             │
│  Input: statistical signals + inline comms risk per tx      │
│                                                             │
│  The orchestrator is an LLM agent with 5 callable tools:   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tool: analyze_financial_patterns(txn_ids, user_id) │   │
│  │  → Transaction Agent (qwen/qwen3-30b-a3b)           │   │
│  │    Deep reasoning over statistical signal summaries  │   │
│  │    Returns per-tx risk score 0.0–1.0 + rationale    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tool: analyze_recipient_network(recipient_iban)    │   │
│  │  → Network Agent                                    │   │
│  │    Aggregates all transactions to a given IBAN      │   │
│  │    Detects mule accounts, merchant fraud patterns   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tool: get_location_check(user, txn, ts, location)  │   │
│  │  → GPS coherence lookup (deterministic from signals)│   │
│  │    Returns COHERENT / INCOHERENT / NO_DATA          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tool: get_comms_detail(user_id, txn_id)            │   │
│  │  → Returns per-transaction TXN_COMMS_RISK block     │   │
│  │    from Phase 1 output (no new LLM call)            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tool: analyze_identity(user_id)                    │   │
│  │  → Identity Agent (qwen/qwen3-32b)                  │   │
│  │    Validates name / address / salary consistency    │   │
│  │    Detects synthetic identity or account takeover   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Output per transaction:                                    │
│    <uuid>: 0 or 1 | high/medium/low | one-sentence reason  │
│                                                             │
│  Model: deepseek/deepseek-v3.2                             │
│  Batch size: 20 tx/call, auto-halves on context overflow   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Memory Agent  (end of each level)                          │
│                                                             │
│  Synthesises fraud patterns found in this level into a      │
│  threat intelligence brief, persisted to memory/            │
│  Brief is injected into the orchestrator system prompt      │
│  for the next level so tactics are carried forward.         │
│                                                             │
│  Model: deepseek/deepseek-v3.2                             │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
  output/{session_id}_level_{N}.txt   ← fraudulent tx IDs
  output/level_{N}_run.log            ← full reasoning trace
```

---

## Decision Framework (Orchestrator)

The orchestrator is prompted to reason like a **senior fraud investigator**, not a rule engine. For each transaction:

1. **Innocent explanation first** — what is the most plausible non-fraudulent account of this transaction? Statistical anomalies are noise unless they form a pattern.
2. **Evidence against the innocent explanation** — what specifically makes the innocent account less credible? A single z-score spike is weak. New recipient + large amount + GPS contradiction + social engineering together is strong.
3. **Coherent fraud narrative** — do the flags hang together into a recognisable fraud pattern (motive, method, target)? If not, they are probably noise.
4. **Comms as a multiplier** — `PRE_TXN_FLAGS=yes` + `RISK_LEVEL=high` means the user was likely manipulated. A statistically routine transaction becomes suspicious if social engineering immediately preceded it.
5. **Economic calibration** — large amounts raise the urgency but never alone justify a fraud call. A salary that matches the user's prior salaries is legitimate regardless of amount.
6. **Profile-relative calibration** — compare to this specific user's history, not population averages.
7. **Threat intelligence carry-forward** — look for evolved versions of previous-level tactics.

**Final call rule:** classify 1 only when the evidence forms a coherent, convincing case. When uncertain: lean 1 on high-value, lean 0 on low-value. Never let a single isolated signal drive a fraud call.

> **Design note:** The hackathon's fictional setting is the year 2087. Birth years in the 2030s–2060s are pinned in the system prompt as normal working-age adults to prevent the orchestrator from misclassifying future-dated identities as synthetic.

---

## Robustness

| Failure Mode | Recovery |
|-------------|---------|
| Context overflow (400) | Batch size halved, retry |
| Rate limit (429) | Exponential backoff with jitter |
| Transient 5xx | Retry up to 3 times |
| Timeout | Per-call timeout, single retry |
| Malformed output | Lenient parsing, then stricter re-prompt |
| Dropped transaction IDs | Detect missing, re-run as mini-batch |
| Max-tokens truncation | Detect incomplete output, reduce batch and retry |

---

## Models Used

| Agent | Model | Rationale |
|-------|-------|-----------|
| Transcription | `google/gemini-2.5-flash-lite` | Cheap, fast, multimodal |
| Communications | `qwen/qwen3-32b` | Strong reasoning over long text threads |
| Transaction | `qwen/qwen3-30b-a3b` | MoE efficiency for batch signal analysis |
| Identity | `qwen/qwen3-32b` | Profile consistency validation |
| Orchestrator | `deepseek/deepseek-v3.2` | Strong instruction following, final arbiter |
| Memory | `deepseek/deepseek-v3.2` | Synthesis and cross-level pattern extraction |

Routed through OpenRouter via the `openai` Python client, using the `strands-agents` agent framework.

---

## Signal Flags Reference

| Flag | Source | Meaning |
|------|--------|---------|
| `z_score` | Statistical | Amount deviation from user's mean (in std deviations) |
| `hour_anomaly` | Statistical | Transaction at unusual hour for this user |
| `new_recipient` | Statistical | First-ever transfer to this recipient IBAN |
| `velocity_spike` | Statistical | Unusually high number of transactions in short window |
| `location_coherence` | GPS | GPS pings contradict transaction location |
| `SALARY_MULTI_SENDER_IBAN` | Salary/Rent | User receives salary from >1 unique sender IBAN (mule indicator) |
| `SALARY_DUPLICATE_MONTH` | Salary/Rent | >1 salary payment received in same calendar month |
| `RENT_DUPLICATE_MONTH` | Salary/Rent | >1 rent payment made in same calendar month |
| `RENT_CITY_MISMATCH` | Salary/Rent | Rent description city ≠ user's registered residence |

---

## Repository Layout

```
mirrorpay-fraud-detection/
├── main.py               # CLI entry — runs phases 0–2 + memory synthesis
├── config.py             # Model IDs, batch sizes, paths, retry limits
├── requirements.txt
├── .env.example          # Template — copy to .env and fill in
├── src/
│   ├── analysis.py       # Statistical signals (no LLM)
│   ├── comms_filter.py   # Communications pre-filtering
│   ├── orchestrator.py   # Tool-using orchestrator agent
│   ├── sub_agents.py     # Comms, Transaction, Identity agents
│   ├── network.py        # Recipient IBAN network analysis
│   ├── memory.py         # Threat-intelligence Memory agent
│   ├── transcription.py  # Audio → text (Gemini, disk-cached)
│   ├── tracking.py       # Langfuse session inspection helpers
│   └── utils.py          # make_model, call_with_retry, validate_output
├── tests/                # pytest unit tests (analysis, memory, orchestrator, utils)
└── docs/
    ├── design.md         # Pre-build design specification
    └── MODEL_WHITELIST.md
```

---

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env  # fill in OPENROUTER_API_KEY, LANGFUSE keys, TEAM_NAME

# Run a single dataset level
TEAM_NAME="my-team" python main.py 1

# Run all available levels in sequence
TEAM_NAME="my-team" python main.py all

# Inspect a Langfuse session (cost / token / trace summary)
python -m src.tracking <session_id>
```

The pipeline writes the fraud classifications to `output/{session_id}_level_{N}.txt` and a full reasoning trace to `output/level_{N}_run.log`. Note that the hackathon dataset is not redistributed in this mirror — see [About this repository](#about-this-repository).

---

## Tests

```bash
pytest tests/ -v
```

Unit tests cover statistical signal computation, the memory agent's threat-intel synthesis, orchestrator output parsing, and the retry / validation utilities.
