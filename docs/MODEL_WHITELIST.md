# Curated Shortlist for Fraud Detection Pipeline

Pick models from here first. Full whitelist below for reference.

**Source:** All data from OpenRouter model pages (openrouter.ai). Prices and specs verified 2026-04-05.

## Tier 1 — Dev/Iteration (cheapest, use for rapid prototyping and batch tuning)

| MODEL_ID | Context | In $/M | Out $/M | Active Params | Notes |
|----------|---------|--------|---------|---------------|-------|
| qwen/qwen-turbo | 131K | $0.03 | $0.13 | ~7B (Qwen2.5-based) | Alibaba's budget model. "Suitable for simple tasks." Tool calling supported. Good for initial pipeline testing. |
| mistralai/mistral-small-3.1-24b-instruct | 131K | $0.03 | $0.11 | 24B | Multimodal, function calling, structured output. Vision capable. Knowledge cutoff Oct 2023 (older). |
| qwen/qwen3-30b-a3b | 41K (131K w/ YaRN) | $0.08 | $0.28 | 3.3B of 30.5B MoE | 128 experts, 8 active. Reasoning mode with `<think>` tags. Outperforms QwQ and Qwen2.5. Very cheap for its capability. |
| qwen/qwen3-32b | 41K (131K w/ YaRN) | $0.08 | $0.24 | 32.8B dense | Dense model, strong reasoning + code. Thinking/non-thinking mode toggle. Good structured output. |
| google/gemini-2.5-flash-lite | 1M | $0.10 | $0.40 | N/A | Massive 1M context. "Ultra-low latency and cost efficiency." Multimodal (text/image/audio/video). Optional reasoning. |
| meta-llama/llama-4-maverick | 1M | $0.15 | $0.60 | 17B of 400B MoE | 128 experts. 1M context, multimodal. 12 languages. Good general-purpose but no explicit reasoning mode. |

## Tier 2 — Mid (balanced cost/quality, good for sub-agents on eval runs)

| MODEL_ID | Context | In $/M | Out $/M | Active Params | Notes |
|----------|---------|--------|---------|---------------|-------|
| qwen/qwq-32b | 131K | $0.15 | $0.58 | 32B | Dedicated reasoning model. "Competitive with DeepSeek-R1, o1-mini." Mandatory `<think>` reasoning. Strong on hard problems. |
| openai/gpt-5-mini | 400K | $0.25 | $2.00 | N/A | Compact GPT-5. Configurable reasoning effort (minimal→high). Tool calling, structured output. 128K max output. |
| deepseek/deepseek-v3.2 | 164K | $0.26 | $0.38 | N/A | "GPT-5 class performance." Gold-medal IMO/IOI results. DeepSeek Sparse Attention. Very cheap output tokens ($0.38). Optional reasoning. |
| google/gemini-2.5-flash | 1M | $0.30 | $2.50 | N/A | "State-of-the-art workhorse." Advanced reasoning, coding, math, science. Built-in thinking. Multimodal. 1M context. |
| x-ai/grok-3-mini | 131K | $0.30 | $0.50 | N/A | "Thinks before responding. Fast, smart, great for logic-based tasks." Mandatory reasoning. Low output cost. |
| deepseek/deepseek-chat | 164K | $0.32 | $0.89 | N/A | DeepSeek V3. 15T token pretraining. "Rivals leading closed-source models." Solid general-purpose. |
| openai/gpt-4.1-mini | 1M | $0.40 | $1.60 | N/A | "Competitive with GPT-4o at lower cost." IFEval: 84.1%. Tool calling, structured output, vision. Good instruction following. |
| deepseek/deepseek-r1-0528 | 164K | $0.45 | $2.15 | 37B of 671B MoE | Updated R1. "Performance on par with OpenAI o1." Open reasoning tokens. 65K max output. Strong analytical reasoning. |
| qwen/qwen3-235b-a22b | 131K | $0.46 | $1.82 | 22B of 235B MoE | Largest Qwen3 MoE. 100+ languages. Tool calling + reasoning. Extended thinking. Max output only 8K — watch for truncation. |

## Tier 3 — Strong (use for orchestrator on final eval, or when accuracy matters most)

| MODEL_ID | Context | In $/M | Out $/M | Active Params | Notes |
|----------|---------|--------|---------|---------------|-------|
| deepseek/deepseek-r1 | 64K | $0.70 | $2.50 | 37B of 671B MoE | Original R1. "On par with o1." 64K context (smallest of strong tier — watch batch sizes). MIT license. |
| openai/o4-mini | 200K | $1.10 | $4.40 | N/A | AIME: 99.5%. Strong STEM/reasoning. "Solves multi-step tasks in under a minute." 100K max output. Tool calling. |
| openai/gpt-4.1 | 1M | $2.00 | $8.00 | N/A | Flagship. SWE-bench: 54.6%, IFEval: 87.4%. Best instruction following. 1M context. Expensive — reserve for final runs. |

## Recommended Configurations

**Development iteration (datasets 1-3, $40 budget):**
- Sub-agents: `qwen/qwen3-30b-a3b` or `qwen/qwen3-32b` ($0.08/M in)
- Orchestrator: `deepseek/deepseek-v3.2` or `google/gemini-2.5-flash` ($0.26-0.30/M in)

**Final evaluation (datasets 4-5, $120 budget):**
- Sub-agents: `google/gemini-2.5-flash` or `qwen/qwq-32b` ($0.15-0.30/M in)
- Orchestrator: `deepseek/deepseek-r1-0528` or `openai/o4-mini` ($0.45-1.10/M in)

## Key Constraints to Watch

| Model | Constraint |
|-------|-----------|
| qwen/qwen3-235b-a22b | Max output 8,192 tokens — will truncate large batches |
| deepseek/deepseek-r1 | Only 64K context — smallest window in strong tier |
| deepseek/deepseek-r1-0528 | Mandatory reasoning — adds cost via thinking tokens |
| qwen/qwq-32b | Mandatory reasoning — cannot disable `<think>` overhead |
| openai/o4-mini | Mandatory reasoning — cost includes hidden reasoning tokens |
| qwen/qwen3-30b-a3b | Native 41K context, 131K only via YaRN — quality may degrade at extended lengths |
| mistralai/mistral-small-3.1 | Knowledge cutoff Oct 2023 — oldest in shortlist |

---

# Full Whitelist Reference

MODEL_ID | CONTEXT_SIZE | INPUT_TOKEN_COST | OUTPUT_TOKEN_COST | OTHER_TOKEN_COST
ai21/jamba-large-1.7 | 256,000 | $2/M input tokens | $8/M output tokens |
aion-labs/aion-1.0-mini | 131,072 | $0.70/M input tokens | $1.40/M output tokens |
aion-labs/aion-2.0 | 131,072 | $0.80/M input tokens | $1.60/M output tokens |
aion-labs/aion-rp-llama-3.1-8b | 32,768 | $0.80/M input tokens | $1.60/M output tokens |
allenai/molmo-2-8b | 36,864 | $0/M input tokens | $0/M output tokens |
allenai/olmo-3-32b-think | 65,536 | $0.15/M input tokens | $0.50/M output tokens |
allenai/olmo-3-7b-instruct | 65,536 | $0/M input tokens | $0/M output tokens |
allenai/olmo-3-7b-think | 65,536 | $0/M input tokens | $0/M output tokens |
allenai/olmo-3.1-32b-instruct | 65,536 | $0.20/M input tokens | $0.60/M output tokens |
amazon/nova-2-lite-v1 | 1,000,000 | $0.30/M input tokens | $2.50/M output tokens |
amazon/nova-lite-v1 | 300,000 | $0.06/M input tokens | $0.24/M output tokens |
amazon/nova-micro-v1 | 128,000 | $0.035/M input tokens | $0.14/M output tokens |
amazon/nova-pro-v1 | 300,000 | $0.80/M input tokens | $3.20/M output tokens |
anthropic/claude-3-haiku | 200,000 | $0.25/M input tokens | $1.25/M output tokens |
anthropic/claude-3.5-haiku | 200,000 | $0.80/M input tokens | $4/M output tokens | $10/K web search
anthropic/claude-haiku-4.5 | 200,000 | $1/M input tokens | $5/M output tokens | $10/K web search
arcee-ai/trinity-mini | 131,072 | $0.045/M input tokens | $0.15/M output tokens |
baidu/ernie-4.5-21b-a3b | 120,000 | $0.07/M input tokens | $0.28/M output tokens |
baidu/ernie-4.5-21b-a3b-thinking | 131,072 | $0.07/M input tokens | $0.28/M output tokens |
baidu/ernie-4.5-300b-a47b | 123,000 | $0.28/M input tokens | $1.10/M output tokens |
baidu/ernie-4.5-vl-28b-a3b | 30,000 | $0.14/M input tokens | $0.56/M output tokens |
baidu/ernie-4.5-vl-424b-a47b | 123,000 | $0.42/M input tokens | $1.25/M output tokens |
bytedance-seed/seed-1.6 | 262,144 | $0.25/M (<128K) else $0.5/M input tokens | $2/M (<128K) else $4/M output tokens |
bytedance-seed/seed-1.6-flash | 262,144 | $0.075/M (<128K) else $0.1/M input tokens | $0.3/M (<128K) else $0.8/M output tokens |
bytedance-seed/seed-2.0-lite | 262,144 | $0.25/M (<128K) else $0.5/M input tokens | $2/M (<128K) else $4/M output tokens |
bytedance-seed/seed-2.0-mini | 262,144 | $0.1/M (<128K) else $0.2/M input tokens | $0.4/M (<128K) else $0.8/M output tokens |
bytedance/ui-tars-1.5-7b | 128,000 | $0.10/M input tokens | $0.20/M output tokens |
cohere/command-r-08-2024 | 128,000 | $0.15/M input tokens | $0.60/M output tokens |
cohere/command-r7b-12-2024 | 128,000 | $0.0375/M input tokens | $0.15/M output tokens |
deepcogito/cogito-v2.1-671b | 128,000 | $1.25/M input tokens | $1.25/M output tokens |
deepseek/deepseek-chat | 163,840 | $0.32/M input tokens | $0.89/M output tokens |
deepseek/deepseek-chat-v3-0324 | 163,840 | $0.20/M input tokens | $0.77/M output tokens |
deepseek/deepseek-chat-v3.1 | 32,768 | $0.15/M input tokens | $0.75/M output tokens |
deepseek/deepseek-v3.1-terminus | 163,840 | $0.21/M input tokens | $0.79/M output tokens |
deepseek/deepseek-v3.2 | 163,840 | $0.26/M input tokens | $0.38/M output tokens |
deepseek/deepseek-v3.2-exp | 163,840 | $0.27/M input tokens | $0.41/M output tokens |
deepseek/deepseek-v3.2-speciale | 163,840 | $0.40/M input tokens | $1.20/M output tokens |
deepseek/deepseek-r1 | 64,000 | $0.70/M input tokens | $2.50/M output tokens |
deepseek/deepseek-r1-0528 | 163,840 | $0.45/M input tokens | $2.15/M output tokens |
deepseek/deepseek-r1-distill-llama-70b | 131,072 | $0.70/M input tokens | $0.80/M output tokens |
deepseek/deepseek-r1-distill-qwen-32b | 32,768 | $0.29/M input tokens | $0.29/M output tokens |
essentialai/rnj-1-instruct | 32,768 | $0.15/M input tokens | $0.15/M output tokens |
google/gemini-2.5-flash | 1,048,576 | $0.30/M input tokens | $2.50/M output tokens | $1/M audio tokens
google/gemini-2.5-flash-lite | 1,048,576 | $0.10/M input tokens | $0.4/M output tokens | $0.3/M audio tokens
google/gemini-2.5-flash-lite-preview-09-2025 | 1,048,576 | $0.10/M input tokens | $0.4/M output tokens | $0.3/M audio tokens
google/gemini-3-flash-preview | 1,048,576 | $0.50/M input tokens | $3/M output tokens | $1/M audio tokens
google/gemini-3.1-flash-lite-preview | 1,048,576 | $0.25/M input tokens | $1.5/M output tokens | $0.5/M audio tokens
google/gemma-2-27b-it | 8,192 | $0.65/M input tokens | $0.65/M output tokens |
google/gemma-2-9b-it | 8,192 | $0.03/M input tokens | $0.09/M output tokens |
google/gemma-3-12b-it | 131,072 | $0.04/M input tokens | $0.13/M output tokens |
google/gemma-3-27b-it | 131,072 | $0.08/M input tokens | $0.16/M output tokens |
google/gemma-3-4b-it | 131,072 | $0.04/M input tokens | $0.08/M output tokens |
google/gemma-3n-e4b-it | 32,768 | $0.02/M input tokens | $0.04/M output tokens |
ibm-granite/granite-4.0-h-micro | 131,000 | $0.017/M input tokens | $0.11/M output tokens |
inception/mercury-2 | 128,000 | $0.25/M input tokens | $0.75/M output tokens |
liquid/lfm-2.2-6b | 32,768 | $0/M input tokens | $0/M output tokens |
liquid/lfm-2-24b-a2b | 32,768 | $0.03/M input tokens | $0.12/M output tokens |
liquid/lfm2-8b-a1b | 8,192 | $0/M input tokens | $0/M output tokens |
mancer/weaver | 8,000 | $0.75/M input tokens | $1/M output tokens |
meituan/longcat-flash-chat | 131,072 | $0.20/M input tokens | $0.80/M output tokens |
meta-llama/llama-3-70b-instruct | 8,192 | $0.51/M input tokens | $0.74/M output tokens |
meta-llama/llama-3-8b-instruct | 8,192 | $0.03/M input tokens | $0.04/M output tokens |
meta-llama/llama-3.1-70b-instruct | 131,072 | $0.40/M input tokens $0.40/M output tokens |
meta-llama/llama-3.1-8b-instruct | 16,384 | $0.02/M input tokens | $0.05/M output tokens |
meta-llama/llama-3.2-11b-vision-instruct | 131,072 | $0.049/M input tokens | $0.049/M output tokens |
meta-llama/llama-3.2-1b-instruct | 60,000 | $0.027/M input tokens | $0.20/M output tokens |
meta-llama/llama-3.2-3b-instruct | 80,000 | $0.051/M input tokens | $0.34/M output tokens |
meta-llama/llama-3.3-70b-instruct | 131,072 | $0.10/M input tokens | $0.32/M output tokens |
meta-llama/llama-4-maverick | 1,048,576 | $0.15/M input tokens | $0.60/M output tokens |
meta-llama/llama-4-scout | 327,680 | $0.08/M input tokens | $0.30/M output tokens |
microsoft/phi-4 | 16,384 | $0.065/M input tokens | $0.14/M output tokens |
minimax/minimax-m1 | 1,000,000 | Starting at $0.44/M input tokens | Starting at $2.20/M output tokens |
minimax/minimax-m2 | 196,608 | $0.3/M input tokens | $1.2/M output tokens |
minimax/minimax-m2-her | 65,536 | $0.30/M input tokens | $1.20/M output tokens |
minimax/minimax-m2.1 | 196,608 | $0.27/M input tokens | $0.95/M output tokens |
minimax/minimax-m2.5 | 196,608 | $0.118/M input tokens | $0.99/M output tokens |
minimax/minimax-01 | 1,000,192 | $0.20/M input tokens | $1.10/M output tokens |
mistralai/mistral-large | mistralai/mistral-large | 128,000 |$2/M input tokens | $6/M output tokens |
mistralai/mistral-large-2407 | 131,072 | $2/M input tokens | $6/M output tokens |
mistralai/mistral-large-2411 | 131,072 | $2/M input tokens | $6/M output tokens |
mistralai/codestral-2508 | 256,000 | $0.30/M input tokens | $0.90/M output tokens |
mistralai/devstral-2512 | 262,144 | $0.40/M input tokens | $2/M output tokens |
mistralai/devstral-medium | 131,072 | $0.40/M input tokens | $2/M output tokens |
mistralai/devstral-small | 131,072 | $0.10/M input tokens | $0.30/M output tokens |
mistralai/ministral-14b-2512 | 262,144 | $0.20/M input tokens | $0.20/M output tokens |
mistralai/ministral-3b-2512 | 131,072 | $0.10/M input tokens | $0.10/M output tokens |
mistralai/ministral-8b-2512 | 262,144 | $0.15/M input tokens | $0.15/M output tokens |
mistralai/mistral-7b-instruct-v0.1 | 2,824 | $0.11/M input tokens | $0.19/M output tokens |
mistralai/mistral-large-2512 | 262,144 | $0.50/M input tokens | $1.50/M output tokens |
mistralai/mistral-medium-3 | 131,072 | $0.40/M input tokens | $2/M output tokens |
mistralai/mistral-medium-3.1 | 131,072 | $0.40/M input tokens | $2/M output tokens |
mistralai/mistral-nemo | 131,072 | $0.02/M input tokens | $0.04/M output tokens |
mistralai/mistral-small-24b-instruct-2501 | 32,768 | $0.05/M input tokens | $0.08/M output tokens |
mistralai/mistral-small-3.1-24b-instruct | 131,072 | $0.03/M input tokens | $0.11/M output tokens |
mistralai/mistral-small-3.2-24b-instruct | 128,000 | $0.075/M input tokens | $0.20/M output tokens |
mistralai/mixtral-8x22b-instruct | 65,536 | $2/M input tokens | $6/M output tokens |
mistralai/mixtral-8x7b-instruct | 32,768 | $0.54/M input tokens | $0.54/M output tokens |
mistralai/pixtral-large-2411 | 131,072 | $2/M input tokens | $6/M output tokens |
mistralai/mistral-saba | 32,768 | $0.20/M input tokens | $0.60/M output tokens |
mistralai/voxtral-small-24b-2507 | 32,000 | $0.10/M input tokens | $0.30/M output tokens | $100/M audio tokens
moonshotai/kimi-k2 | 131,072 | $0.57/M input tokens | $2.30/M output tokens |
moonshotai/kimi-k2-0905 | 131,072 | $0.4/M input tokens | $2/M output tokens |
moonshotai/kimi-k2-thinking | 131,072 | $0.47/M input tokens | $2/M output tokens |
moonshotai/kimi-k2.5 | 262,144 | $0.3827/M input tokens | $1.72/M output tokens |
morph/morph-v3-fast | 81,920 | $0.80/M input tokens | $1.20/M output tokens |
morph/morph-v3-large | 262,144 | $0.90/M input tokens | $1.90/M output tokens |
gryphe/mythomax-l2-13b | 4,096 | $0.06/M input tokens | $0.06/M output tokens |
nvidia/llama-3.1-nemotron-70b-instruct | 131,072 | $1.20/M input tokens | $1.20/M output tokens |
nvidia/llama-3.3-nemotron-super-49b-v1.5 | 131,072 | $0.10/M input tokens | $0.40/M output tokens |
nvidia/nemotron-3-nano-30b-a3b | 262,144 | $0.05/M input tokens | $0.20/M output tokens |
nvidia/nemotron-nano-12b-v2-vl | 131,072 | $0.20/M input tokens | $0.60/M output tokens |
nvidia/nemotron-nano-9b-v2 | 131,072 | $0.04/M input tokens | $0.16/M output tokens |
nex-agi/deepseek-v3.1-nex-n1 | 131,072 | $0.135/M input tokens | $0.50/M output tokens |
nousresearch/hermes-3-llama-3.1-405b | 131,072 | $1/M input tokens | $1/M output tokens |
nousresearch/hermes-3-llama-3.1-70b | 131,072 | $0.30/M input tokens | $0.30/M output tokens |
nousresearch/hermes-4-405b | 131,072 | $1/M input tokens | $3/M output tokens |
nousresearch/hermes-4-70b | 131,072 | $0.13/M input tokens | $0.40/M output tokens |
nousresearch/hermes-2-pro-llama-3-8b | 8,192 | $0.14/M input tokens | $0.14/M output tokens |
openai/gpt-3.5-turbo | 16,385 | $0.50/M input tokens | $1.50/M output tokens |
openai/gpt-3.5-turbo-0613 | 4,095 | $1/M input tokens | $2/M output tokens |
openai/gpt-3.5-turbo-instruct | 4,095 | $1.5/M input tokens | $2/M output tokens |
openai/gpt-4.1 | 1,047,576 | $2/M input tokens | $8/M output tokens | $10/K web search
openai/gpt-4.1-mini | 1,047,576 | $0.4/M input tokens | $1.6/M output tokens | $10/K web search
openai/gpt-4.1-nano | 1,047,576 | $0.1/M input tokens | $0.4/M output tokens | $10/K web search
openai/gpt-4o-mini | 128,000  | $0.15/M input tokens | $0.6/M output tokens |
openai/gpt-4o-mini-2024-07-18 | 128,000  | $0.15/M input tokens | $0.6/M output tokens |
openai/gpt-4o-mini-search-preview | 128,000  | $0.15/M input tokens | $0.60/M output tokens | $27.50/K web search
openai/gpt-5-mini | 400,000  | $0.25/M input tokens | $2/M output tokens | $10/K web search
openai/gpt-5-nano | 400,000  | $0.05/M input tokens | $0.4/M output tokens | $10/K web search
openai/gpt-5.1-codex-mini | 400,000  | $0.25/M input tokens | $2/M output tokens | 
openai/gpt-oss-120b | 131,072 | $0.039/M input tokens | $0.19/M output tokens |
openai/gpt-oss-20b | 131,072 | $0.03/M input tokens | $0.11/M output tokens |
openai/o3 | 200,000 | $2/M input tokens | $8/M output tokens | $10/K web search
openai/o3-mini | 200,000 | $1.1/M input tokens | $4.4/M output tokens | 
openai/o3-mini-high | 200,000 | $1.1/M input tokens | $4.4/M output tokens | 
openai/o4-mini | 200,000 | $1.1/M input tokens | $4.4/M output tokens | $10/K web search
openai/o4-mini-deep-research | 200,000 | $2/M input tokens | $8/M output tokens | $10/K web search
openai/o4-mini-high | 200,000 | $1.1/M input tokens | $4.4/M output tokens | $10/K web search
perplexity/sonar | 127,072 | $1/M input tokens | $1/M output tokens | $5/K web search
perplexity/sonar-deep-research | 128,000 | $2/M input tokens | $8/M output tokens | $5/K web search
perplexity/sonar-reasoning-pro | 128,000 | $2/M input tokens | $8/M output tokens | $5/K web search
prime-intellect/intellect-3 | 131,072 | $0.20/M input tokens | $1.10/M output tokens |
qwen/qwen-2.5-72b-instruct | 32,768 | $0.12/M input tokens | $0.39/M output tokens |
qwen/qwen-2.5-coder-32b-instruct | 32,768 | $0.66/M input tokens | $1/M output tokens |
qwen/qwq-32b | 131,072 | $0.15/M input tokens | $0.58/M output tokens |
qwen/qwen-vl-max | 131,072 | $0.52/M input tokens | $2.08/M output tokens |
qwen/qwen-vl-plus | 131,072 | $0.1365/M input tokens | $0.4095/M output tokens |
qwen/qwen-max | 32,768 | $1.04/M input tokens | $4.16/M output tokens |
qwen/qwen-plus | 1,000,000 | $0.26/M (<256K) else $0.78/M input tokens | $0.78/M (<256K) else $2.34/M output tokens |
qwen/qwen-turbo | 131,072 | $0.0325/M input tokens | $0.13/M output tokens |
qwen/qwen-2.5-7b-instruct | 32,768 | $0.04/M input tokens | $0.10/M output tokens |
qwen/qwen2.5-coder-7b-instruct | 32,768 | $0.03/M input tokens | $0.09/M output tokens |
qwen/qwen2.5-vl-32b-instruct | 128,000 | $0.2/M input tokens | $0.6/M output tokens | 
qwen/qwen2.5-vl-72b-instruct | 32,768 | $0.8/M input tokens | $0.8/M output tokens |
qwen/qwen-2.5-vl-7b-instruct | 32,768 | $0/M input tokens | $0/M output tokens |
qwen/qwen3-14b | 40,960 | $0.06/M input tokens | $0.24/M output tokens |
qwen/qwen3-235b-a22b | 131,072 | $0.455/M input tokens | $1.82/M output tokens |
qwen/qwen3-235b-a22b-2507 | 262,144 | $0.071/M input tokens | $0.10/M output tokens |
qwen/qwen3-235b-a22b-thinking-2507 | 131,072 | $0.1495/M input tokens | $1.495/M output tokens |
qwen/qwen3-30b-a3b | 40,960 | $0.08/M input tokens | $0.28/M output tokens |
qwen/qwen3-30b-a3b-instruct-2507 | 262,144 | $0.09/M input tokens | $0.30/M output tokens |
qwen/qwen3-30b-a3b-thinking-2507 | 131,072 | $0.08/M input tokens | $0.40/M output tokens |
qwen/qwen3-32b | 40,960 | $0.08/M input tokens | $0.24/M output tokens |
qwen/qwen3-8b | 40,960 | $0.05/M input tokens | $0.4/M output tokens |
qwen/qwen3-coder-30b-a3b-instruct | 160,000 | $0.07/M input tokens | $0.27/M output tokens |
qwen/qwen3-coder | 262,144 | $0.22/M input tokens | $1/M output tokens |
qwen/qwen3-coder-flash | 1,000,000 | $0.195/M (<32K) elif $0.325 (<128K) else $0.52/M input tokens | $0.975/M (<32K) elif $1.625 (<128K) else $2.60/M output tokens |
qwen/qwen3-coder-next | 262,144 | $0.12/M input tokens | $0.75/M output tokens |
qwen/qwen3-coder-plus | 1,000,000 | $0.65/M (<32K) elif $1.17 (<128K) else $1.95/M input tokens | $3.25/M (<32K) elif $5.85 (<128K) else $9.75/M output tokens |
qwen/qwen3-max | 262,144 | $0.78/M (<32K) elif $1.56 (<128K) else $1.95/M input tokens | $3.9/M (<32K) elif $7.8 (<128K) else $9.75/M output tokens |
qwen/qwen3-max-thinking | 262,144 | $0.78/M (<32K) elif $1.56 (<128K) else $1.95/M input tokens | $3.9/M (<32K) elif $7.8 (<128K) else $9.75/M output tokens |
qwen/qwen3-next-80b-a3b-instruct | 262,144 | $0.09/M input tokens | $1.1/M output tokens |
qwen/qwen3-next-80b-a3b-thinking | 131,072 | $0.0975/M input tokens | $0.78/M output tokens |
qwen/qwen3-vl-235b-a22b-instruct | 262,144 | $0.20/M input tokens | $0.88/M output tokens |
qwen/qwen3-vl-30b-a3b-instruct | 131,072 | $0.13/M input tokens | $0.52/M output tokens |
qwen/qwen3-vl-32b-instruct | 131,072 | $0.104/M input tokens | $0.416/M output tokens |
qwen/qwen3-vl-8b-instruct | 131,072 | $0.08/M input tokens | $0.5/M output tokens |
qwen/qwen3-vl-8b-thinking | 131,072 | $0.117/M input tokens | $1.365/M output tokens |
qwen/qwen3.5-397b-a17b | 262,144 | $0.39/M input tokens | $2.34/M output tokens |
qwen/qwen3.5-plus-02-15 | 1,000,000 | $0.26/M (<256K) else $0.325/M input tokens | $1.56/M (<256K) else $1.95/M output tokens |
qwen/qwen3.5-122b-a10b | 262,144 | $0.26/M input tokens | $2.08/M output tokens |
qwen/qwen3.5-27b | 262,144 | $0.195/M input tokens | $1.56/M output tokens |
qwen/qwen3.5-35b-a3b | 262,144 | $0.1625/M input tokens | $1.3/M output tokens |
qwen/qwen3.5-9b | 256,000 | $0.05/M input tokens | $0.15/M output tokens |
qwen/qwen3.5-flash-02-23 | 1,000,000 | $0.065/M input tokens | $0.26/M output tokens |
undi95/remm-slerp-l2-13b | 6,144 | $0.45/M input tokens | $0.65/M output tokens |
relace/relace-search | 256,000 | $1/M input tokens | $3/M output tokens |
sao10k/l3-lunaris-8b | 8,192 | $0.04/M input tokens | $0.05/M output tokens |
sao10k/l3.1-euryale-70b | 131,072 | $0.85/M input tokens | $0.85/M output tokens |
sao10k/l3-euryale-70b | 8,192 | $1.48/M input tokens | $1.48/M output tokens |
stepfun/step-3.5-flash | 262,144 | $0.1/M input tokens | $0.3/M output tokens |
switchpoint/router | 131,072 | $0.85/M input tokens | $3.4/M output tokens |
tencent/hunyuan-a13b-instruct | 131,072 | $0.14/M input tokens | $0.57/M output tokens |
thedrummer/cydonia-24b-v4.1 | 131,072 | $0.3/M input tokens | $0.5/M output tokens |
thedrummer/rocinante-12b | 32,768 | $0.17/M input tokens | $0.43/M output tokens |
thedrummer/skyfall-36b-v2 | 32,768 | $0.55/M input tokens | $0.8/M output tokens |
thedrummer/unslopnemo-12b | 32,768 | $0.4/M input tokens | $0.4/M output tokens |
alibaba/tongyi-deepresearch-30b-a3b | 131,072 | $0.09/M input tokens | $0.45/M output tokens |
upstage/solar-pro-3 | 128,000 | $0.15/M input tokens | $0.60/M output tokens |
microsoft/wizardlm-2-8x22b | 65,535 | $0.62/M input tokens | $0.62/M output tokens |
writer/palmyra-x5 | 1,040,000 | $0.60/M input tokens | $6/M output tokens |
xiaomi/mimo-v2-flash | 262,144 | $0.09/M input tokens | $0.29/M output tokens |
z-ai/glm-4-32b | 128,000 | $0.1/M input tokens | $0.10/M output tokens |
z-ai/glm-4.5 | 131,072 | $0.6/M input tokens | $2.2/M output tokens |
z-ai/glm-4.5-air | 131,072 | $0.13/M input tokens | $0.85/M output tokens |
z-ai/glm-4.5v | 65,535 | $0.6/M input tokens | $1.8/M output tokens |
z-ai/glm-4.6 | 204,800 | $0.39/M input tokens | $1.90/M output tokens |
z-ai/glm-4.6v | 131,072 | $0.3/M input tokens | $0.9/M output tokens |
z-ai/glm-4.7 | 202,752 | $0.39/M input tokens | $1.75/M output tokens |
z-ai/glm-4.7-flash | 202,752 | $0.06/M input tokens | $0.4/M output tokens |
z-ai/glm-5 | 80,000 | $0.72/M input tokens | $2.30/M output tokens |
x-ai/grok-3-mini | 131,072 | $0.3/M input tokens | $0.5/M output tokens | $5/K web search
x-ai/grok-4-fast | 2,000,000 | $0.20/M input tokens | $0.50/M output tokens | $5/K web search
x-ai/grok-4.1-fast | 2,000,000 | $0.20/M input tokens | $0.50/M output tokens | $5/K web search
x-ai/grok-code-fast-1 | 256,000 | $0.20/M input tokens | $1.50/M output tokens | $5/K web search