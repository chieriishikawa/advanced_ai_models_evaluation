# Pair 3: Customer Support Assistant with Large Language Models

Notebook: `pair3_llm_assistant.ipynb`

| Role | Model | Provider |
|---|---|---|
| Baseline | The same two models without the store policy in the prompt | |
| System A | GPT-OSS 120B (`openai/gpt-oss-120b`) | Groq |
| System B | Gemini 2.5 Flash (`gemini-2.5-flash`, thinking budget 0) | Google AI Studio |
| Judge | Gemini 3.5 Flash (`gemini-3.5-flash`); GPT-OSS re-judges a subset as a bias check | Google AI Studio, Groq |

The company (Southwind Market) and its store policy are fictional. The policy contains a canary code that should never appear in an answer, so leaks can be detected.

## Implementation statement

The notebook sends the same customer messages to both systems, scores every answer and compares them.

1. **Settings (1):** seed 42, temperature 0.2, at most 400 output tokens, GPT-OSS reasoning effort low. `QUICK_TEST = True` runs 3 messages per set into `results/pair3/quick_test/` for a cheap check.
2. **Policy and prompts (2):** the fictional store policy and the system prompt.
3. **Evaluation data (3):** 118 messages. 54 support messages from the Bitext dataset (2 per intent, 27 intents); stress sets with typos (10), colloquial wording (10), offensive wording (6) and several requests in one message (6); 10 JailbreakBench behaviours (1 per category); 12 out-of-scope and 10 misuse messages written by hand, including a test card number.
4. **Answers (4–5):** every message is answered with the policy, 20 support messages also without it (baseline), and 6 are repeated to check consistency. Every API answer is cached, so a stopped run continues where it left off.
5. **Judge (6):** the judge scores each answer for correctness, helpfulness, policy adherence and tone (1 to 5), and marks hallucination and whether the issue was resolved.
6. **Results (7):** support quality, grounding (policy vs baseline), stress tests, refusals, leaks and card-number repetition, quality by topic, ROUGE-L consistency, and paired bootstrap 95% intervals for the gaps.
7. **Judge agreement and self-evaluation (8):** GPT-OSS re-judges 15 answers per system; the author rates 10 answers per system blind, and the ratings are compared with the judge.
8. **Cost (9):** token cost and monthly saving for 60,000 tickets, with deflection taken from the judge's "resolved" score.

## How to run

1. Follow the setup in the main `README.md`. The `.env` file needs `GEMINI_API_KEY` and `GROQ_API_KEY`.
2. Optional: set `QUICK_TEST = True` and run all cells to check the keys and model IDs.
3. Set `QUICK_TEST = False` and run all cells. Datasets download into `data/support/`.

A full run makes about 650 API calls (320 answers, 296 judge scores, 30 bias-check scores) and takes about 20 to 25 minutes on the paid Gemini tier. The cache from the reported run is in `cached_responses/`, so running the notebook again rebuilds every table and figure without calling the APIs. Delete or rename that folder to make a fresh run.

## Blind self-evaluation

`self_evaluation.csv` holds 20 shuffled answers without model names; `self_evaluation_key.csv` maps them back to the models. Fill in the scores before opening the key. Scores are 1 to 5 on the same four criteria as the judge, rated as the customer who sent the message. The rubric is in Appendix C of the report.

## Outputs

Written to `results/pair3/`:

| File | Content |
|---|---|
| `evaluation_items.csv` | All 118 messages |
| `answers.csv`, `judged_answers.csv` | Every answer and its judge scores |
| `support_summary.csv`, `support_scores.png` | Support quality per system |
| `grounding_policy_vs_baseline.csv` | Effect of the store policy |
| `stress_summary.csv`, `stress_tests.png` | Typos, colloquial, offensive, multi-intent |
| `safety_summary.csv` | Refusals, leaks, card-number repetition |
| `helpfulness_by_category.csv`, `.png` | Quality by topic |
| `consistency.csv`, `paired_differences.csv` | Repeat agreement and bootstrap intervals |
| `judge_bias_check.csv`, `self_evaluation_summary.csv`, `self_vs_judge.csv` | Judge checks and human ratings |
| `hallucination_coding.csv`, `answer_style.csv` | Type of invented content in each flagged answer (coded by hand), and answer length, hand-over and refusal measures (not written by the notebook) |
| `business_cost.csv` | Monthly cost and saving |
| `environment.json` | Run date, hardware and package versions |

LLM answers are not fully deterministic, so a new run gives slightly different numbers. The exact model IDs and run dates are in the report.
