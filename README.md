# MirrorPay Fraud Detection — Multi-Agent System

A multi-agent LLM pipeline for real-time financial fraud detection, built for the **Reply AI Agent Challenge 2026** (team **ICL_Hotdogs**).

---

## About this repository

This is a **sanitised public mirror** of the team submission. The original development repository is private. This mirror exists to make the architecture, agent design, and engineering approach visible as a portfolio piece.

**Not included in this mirror (intentionally):**

- `data/` — the hackathon dataset (transactions, audio recordings, SMS, emails, GPS pings) is Reply IP and not redistributable.
- `output/` and `memory/` — derived artefacts from running the pipeline against the hackathon dataset.
- `AIAgentChallenge-ProblemStatement16April.pdf` and `langfuse-template-reference/` — Reply-provided reference materials.
- `.env` — replaced with `.env.example`.

The code, agent prompts, orchestrator logic, retry/robustness layer, and design documents are all preserved.

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

## Decision Framework (Orchestrator)

**Setting: year 2087.** Birth years in the 2030s–2060s are normal working-age adults — not synthetic identities.

The orchestrator reasons like a **senior fraud investigator**, not a rule engine. For each transaction it works through:

1. **Innocent explanation first** — what is the most plausible non-fraudulent account of this transaction? Statistical anomalies are noise unless they form a pattern.
2. **Evidence against the innocent explanation** — what specifically makes the innocent account less credible? A single z-score spike is weak. New recipient + large amount + GPS contradiction + social engineering together is strong.
3. **Coherent fraud narrative** — do the flags hang together into a recognisable fraud pattern (motive, method, target)? If not, they are probably noise.
4. **Comms as a multiplier** — `PRE_TXN_FLAGS=yes` + `RISK_LEVEL=high` means the user was likely manipulated. A statistically routine transaction becomes suspicious if social engineering immediately preceded it.
5. **Economic calibration** — large amounts raise the urgency but never alone justify a fraud call. A salary that matches the user's prior salaries is legitimate regardless of amount.
6. **Profile-relative calibration** — compare to this specific user's history, not population averages.
7. **Threat intelligence carry-forward** — look for evolved versions of previous-level tactics.

**Final call rule:** classify 1 only when the evidence forms a coherent, convincing case. When uncertain: lean 1 on high-value, lean 0 on low-value. Never let a single isolated signal drive a fraud call.

---

## Robustness

| Failure Mode | Recovery |
|-------------|---------|
| Context overflow (400) | Batch size halved, retry |
| Rate limit (429) | Exponential backoff with jitter |
| Transient 5xx | Retry up to 3 times |
| Malformed output | Lenient parsing, re-prompt |
| Dropped transaction IDs | Detect missing, retry as mini-batch |

---

## File Structure

```
reply-ai/
├── main.py               # Entry point — phases 0–2 + output
├── config.py             # All model IDs, batch sizes, paths
├── src/
│   ├── analysis.py       # Statistical signals (no LLM)
│   ├── orchestrator.py   # Dynamic orchestrator agent + tools
│   ├── sub_agents.py     # Comms, Transaction, Identity agents
│   ├── network.py        # Recipient IBAN network analysis
│   ├── memory.py         # Threat intelligence Memory agent
│   ├── transcription.py  # Audio → text (Gemini, cached)
│   ├── tracking.py       # Langfuse tracing helpers
│   └── utils.py          # make_model, call_with_retry, validate_output
├── data/                 # Dataset folders (not committed)
├── output/               # Results + run logs (not committed)
├── memory/               # Threat intel briefs per level
└── docs/
    └── MODEL_WHITELIST.md
```

---

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env  # fill in OPENROUTER_API_KEY, LANGFUSE keys, TEAM_NAME

# Run a single level
TEAM_NAME="my-team" python main.py 1

# Run all available levels
TEAM_NAME="my-team" python main.py all
```

Output file: `output/{session_id}_level_{N}.txt`
Log file: `output/level_{N}_run.log`

The session ID printed to stdout is the Langfuse session ID required for submission.

---

## Models Used

| Agent | Model | Rationale |
|-------|-------|-----------|
| Transcription | `google/gemini-2.5-flash-lite` | Cheap, fast, multimodal |
| Communications | `qwen/qwen3-32b` | Strong reasoning over long text threads |
| Transaction | `qwen/qwen3-30b-a3b` | MoE efficiency for batch signal analysis |
| Orchestrator | `deepseek/deepseek-v3.2` | Strong instruction following, final arbiter |
| Memory | `deepseek/deepseek-v3.2` | Synthesis and cross-level pattern extraction |
| Identity | `qwen/qwen3-32b` | Profile consistency validation |
