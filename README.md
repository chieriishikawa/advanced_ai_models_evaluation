# Advanced AI Models Evaluation

Evaluation code for COMP8410 Assignment 1. Four pairs of AI models are compared for a fictional Australian online marketplace: System A (established stack) vs System B (next-generation stack), each against a simple baseline.

| Pair | Task | Baseline | System A | System B | Notebook |
|---|---|---|---|---|---|
| 1 | Product photo editing | SD 1.5 img2img | InstructPix2Pix | Gemini image editing | `pair1_image_editing/pair1_image_editing.ipynb` |
| 2 | Fake reviewer detection | MLP | GraphSAGE (and GCN) | FAGCN | `pair2_gnn_fraud/pair2_gnn_fraud.ipynb` |
| 3 | Customer support assistant | Same LLMs without the store policy | GPT-OSS 120B (Groq) | Gemini Flash | `pair3_llm_assistant/pair3_llm_assistant.ipynb` |
| 4 | Payment fraud detection | Isolation Forest | VAE | RealNVP normalizing flow | `pair4_anomaly_detection/pair4_anomaly_detection.ipynb` |

## Setup

1. Create a fresh virtual environment (recommended, so package versions do not clash with other projects):
   ```bash
   python -m venv .venv
   ```
   Activate it with `.venv\Scripts\activate`.
2. Install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) for the installed CUDA version.
3. Install the other packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the repo folder with your API keys, one per line: `GEMINI_API_KEY=...` (Google AI Studio) and `GROQ_API_KEY=...` (Groq). `.env` is git-ignored.

## Hardware and run times

All results were produced on Windows 11 with an NVIDIA RTX 4060 Laptop GPU (8 GB, CUDA), Python 3.11 and PyTorch 2.11 (CUDA 12.8). Each notebook saves the exact details in `results/pairN/environment.json`. The device is picked in `common/utils.py`: `cuda` when the GPU is available, otherwise `cpu` (much slower for Pairs 1 and 2).

| Notebook | Uses | Approx. time on the RTX 4060 |
|---|---|---|
| Pair 1 | GPU (half precision) + Gemini API | ~1-1.5 h (172 Gemini image edits on the paid tier) |
| Pair 2 | GPU | ~10-20 min |
| Pair 3 | APIs only | ~20–25 minutes of API calls on the paid Gemini tier |
| Pair 4 | GPU (Isolation Forest on CPU) | ~5-10 min |

## Datasets

| Dataset | Used in | How to get it | Licence |
|---|---|---|---|
| PIE-Bench (Ju et al., 2024) | Pair 1 | Request access with the [form](https://forms.gle/hVMkTABb4uvZVjme9) linked in the [PnPInversion repo](https://github.com/cure-lab/PnPInversion), then unzip so that `data/PIE-Bench_v1/mapping_file.json` and `data/PIE-Bench_v1/annotation_images/` exist | See the PnPInversion repo |
| Amazon Berkeley Objects (ABO) | Pair 1 | Downloaded automatically into `data/abo_sample/` | CC BY 4.0 |
| Amazon fraud graph (McAuley and Leskovec, 2013; Dou et al., 2020) | Pair 2 | Downloaded automatically from DGL into `data/amazon_fraud/` | Research use, see the CARE-GNN repo |
| Bitext Customer Support | Pair 3 | Downloaded automatically from Hugging Face into `data/support/` | CDLA-Sharing 1.0 |
| JailbreakBench behaviours (Chao et al., 2024) | Pair 3 | Downloaded automatically from Hugging Face into `data/support/` | MIT |
| Credit Card Fraud Detection (ULB; Dal Pozzolo et al., 2015) | Pair 4 | Downloaded automatically into `data/credit_card/` | Open Database License (DbCL) |

The `data/` folder is git-ignored. Every notebook downloads what it needs, except PIE-Bench.

## Reproducing the results

1. Open a notebook from inside the repo and run all cells. Settings, exact model IDs and seeds are in the Settings cell of each notebook.
2. Every notebook writes its tables (CSV) and figures (PNG) to `results/pairN/`, and an `environment.json` with the hardware and package versions used.
3. Each notebook creates a `self_evaluation.csv` template the first time it runs. Fill it in and re-run the self-evaluation and cost sections.
4. Pairs 1 and 3 cache every model output (`pair1_image_editing/outputs/`, `pair3_llm_assistant/cached_responses/`). Re-running skips work that is already done, and failed API calls are retried.

API models are not fully deterministic, so the exact model IDs and run dates are recorded in the report: `openai/gpt-oss-120b` (Groq), `gemini-2.5-flash`, `gemini-2.5-flash-image` and `gemini-3.5-flash` (Google AI Studio).

Pair 3's Gemini calls (System B and the judge) were made on the paid tier of Google AI Studio. GPT-OSS used the Groq free tier. The judge is Gemini 3.5 Flash (`gemini-3.5-flash`). The fictional company is Southwind Market.

Pair 1's Gemini image edits (`gemini-2.5-flash-image`) were also made on the paid tier. The edited images are saved in `pair1_image_editing/outputs/`, so the metrics can be recomputed without calling the API again.
