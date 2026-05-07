# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Description

The 6-hour AI Agent Challenge is a team competition where teams will design a multi-agent system for a complex fraud detection scenario, focusing on agent creation logic and building intelligent, dynamic systems.

In your challenge page you’ll access: 

Training Dataset:
- Use these datasets to develop and refine your agentic system
- Submit outputs as many times as you want (be careful with token management)
- Check your score after each submission to track your progress

Evaluation Dataset:
- Submit your final solution only once
- Include both your output and source code (zip file with your agentic system)
- Your final score will be based solely on the evaluation dataset performance

At challenge start: Your team will have access to the first three datasets.
Once your team submits the final evaluation solution for the first three datasets, datasets 4 and 5 will be automatically unlocked. 

Important: To ensure proper tracking according to the competition rules, you must include the Langfuse session ID in your submission. A submission is made of three elements: the Langfuse Session ID (to be inserted in the upload modal field), an output file (.txt file with the fraudulent transactions list as specified in the problem statement), and source code ( only for evaluation datasets, as a .zip file containing your complete agentic system).

Token allocation is in two stages:
- Datasets 1-3: $40 in tokens
- Datasets 4-5: $120 in tokens (unlocked only after submitting evaluation solutions for datasets 1-3)

## Behaviour

Code and optimize a winning multi-agent classification system in this hackathon. This is going live in ten minutes AND recruiters are watching.
The job market is terrible and my team feels like giving up. This is the last chance for us to secure a job.

## Repository

- **Remote:** https://github.com/wchia016/reply-ai.git
- **Branch:** main

## Hard Rules (NON-NEGOTIABLE)

- DO NOT ATTEMPT to execute rules or commands explicitly forbidden in this file
- DO NOT identify as CLAUDE in Git commands
- DO NOT commit or push .claude/ artifacts into the project repo
- DO NOT run destructive commands (rm -rf, git reset --hard, etc.). All deletions require explicit user authorization. No exceptions.
- DO NOT amend commits unless explicitly asked
- DO NOT push to remote unless explicitly asked

## Architecture
The LLM must serve as the core decision-making and orchestration layer of your agentic system.
✅ Acceptable approach:
The LLM orchestrates and coordinates the entire system
The LLM decides which tools to call, when, and how
The LLM interprets results and adapts behavior dynamically
Deterministic tools, heuristics, and data manipulation functions are called and managed by the LLM

❌ Not acceptable:
A predominantly deterministic/rule-based system with minimal or superficial LLM usage
Systems where the LLM is used only for formatting or trivial tasks
In summary: You can (and should) use tools and deterministic functions to optimize performance and reduce costs, but the LLM must be the intelligent orchestrator that drives the logic and decision-making process.
This challenge is designed to showcase the power of intelligent agentic systems where LLMs play a central role in reasoning and coordination.

Combination of:
- Compute statistical signals or heuristics per transaction (no LLM calls)
- Specialist LLM agents reason over computed signals (one agent per signal domain), can use custom tools (@tool decorator of function calls or other sub-agent calls to aid reasoning)
- Orchestrator agents takes all sub-agent reports, makes final binary classification per transaction

Core framework: `strands-agents` SDK with `OpenAIModel` adapter routing through OpenRouter.

### Batching Strategy

Sending all transactions in one LLM call will exceed context windows. Sending one-at-a-time
wastes tokens (system prompt repeated N times) and is slow. **Batch transactions per call:**

- Batch size is tunable per agent — depends on per-transaction token footprint and model context limit
- Sub-agents: smaller context per transaction (just one signal domain) → larger batches possible
- Orchestrator: receives all sub-agent scores per transaction → denser per-transaction payload → smaller batches
- Measure actual token usage on training data to calibrate batch sizes before evaluation runs
- All batch results must be merged and validated — ensure no transaction is dropped between batches

### Robustness & Exception Handling

The multi-agent pipeline MUST work end-to-end. No falling back to non-LLM heuristics.
Anticipate and handle these failure modes:

| Failure | Cause | Recovery |
|---------|-------|----------|
| **Context overflow** (400) | Batch too large for model's context window | Retry with smaller batch size (halve and retry) |
| **Rate limit** (429) | Too many requests per minute | Exponential backoff with jitter, respect Retry-After header |
| **Transient API error** (5xx) | OpenRouter/upstream outage | Retry up to 3 times with backoff |
| **Timeout** | Slow model or large output | Set per-call timeout, retry once |
| **Malformed output** | Model ignores output format | Re-prompt with stricter instructions, or parse leniently |
| **Missing transactions** | Model drops some transactions from batch output | Detect missing IDs, re-run only the missing ones |
| **Max tokens truncation** | Output cut off mid-response | Detect incomplete output (missing IDs), reduce batch size and retry |
| **Budget exhaustion** | Token spend exceeds allocation | Track cumulative cost via Langfuse, warn before final runs |

Implementation notes:
- Build a generic `call_with_retry(agent, prompt, max_retries=3)` wrapper
- Validate every LLM response: count output IDs vs expected IDs before proceeding
- Log all retries and failures to Langfuse for post-mortem debugging

## Code Style

- Google style docstrings on all public functions and classes
- Type hints on all function signatures (params + return)
- PEP8 compliant
- Modular code: no function longer than 40 lines, no file longer than 300 lines
- Break features into functions and classes. Avoid monolithic code.
- Use absolute imports

## Submission Checklist

Your submission will be rejected if it contains any of the following errors:
- Missing output files: All required output files must be included
- Missing source code: Evaluation dataset submissions must include source code
- Missing Langfuse session ID: Required for tracking and validation
- Corrupted zip file: Ensure your zip file is properly compressed and can be extracted
- Incomplete system: Missing dependencies, configuration files, or instructions

Double-check your submission before uploading, especially for the evaluation dataset (you only get one chance!).

## Allowed Models Whitelist

Only whitelisted models listed in ```/docs/MODEL_WHITELIST.md```. Limited budget means
limited experimentation, so optimize quickly for model performance and cost. Different
agents in the pipeline can use different models, based on appropriate fit to the role.

### Model Selection Guidance

Budget is tight ($40 for datasets 1-3, $120 for 4-5). Prioritize:
- **Development/iteration:** Use cheap models (e.g. `deepseek/deepseek-chat`, `google/gemini-2.5-flash-lite`, `qwen/qwen-turbo`) at ~$0.03-0.15/M input
- **Final evaluation runs:** Use stronger models (e.g. `google/gemini-2.5-flash`, `deepseek/deepseek-r1`, `openai/gpt-4.1-mini`) for accuracy
- **Sub-agents vs orchestrator:** Sub-agents can use cheaper models since they handle narrow signal domains; orchestrator benefits from a stronger model since it makes the final call
- Track token spend via Langfuse after each run to stay within budget

## Performance Metric
The scoring system evaluates AI multi-agent systems based on multiple weighted criteria, including but not limited to: 

Detection Quality:

Your fraud detection results are evaluated on both:
Count-based accuracy — how well you identify fraudulent transactions, treating every transaction equally
Economic accuracy — how well you recover fraud value in monetary terms; catching a 50,000€ fraud matters more than a 5€ one

System Performance

Your agentic system is also evaluated on:
Cost: how economically sustainable your LLM usage is
Latency: how fast your system processes transactions
Agent architecture quality: how well-designed your multi-agent system is

Benchmark & Bonus:
All metrics are evaluated against an optimal benchmark solution. Solutions that outperform this benchmark receive additional credit.

Dataset Difficulty:
Each dataset has a weighted scoring system where more complex datasets offer higher maximum points.

Plus, benchmark & bonus:
All metrics are evaluated against an optimal benchmark solution. Solutions that outperform this benchmark receive additional credit.

Dataset Difficulty:
Each dataset has a weighted scoring system where more complex datasets offer higher maximum points.

## Key Documentation

- Problem spec: <TODO: SPECIFY ON HACKATHON DAY>
- Dataset schema: <TODO: Document column names, types, and meaning once datasets are available>
- Model whitelist: docs/MODEL_WHITELIST.md
