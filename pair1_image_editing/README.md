# Pair 1: Product Photo Editing

Notebook: `pair1_image_editing.ipynb`

| Role | Model | Where it runs |
|---|---|---|
| Baseline | SD 1.5 img2img (`stable-diffusion-v1-5/stable-diffusion-v1-5`) | Local GPU, float16 |
| System A | InstructPix2Pix (`timbrooks/instruct-pix2pix`) | Local GPU, float16 |
| System B | Gemini image editing (`gemini-2.5-flash-image`) | Google AI Studio API |
| Metric model | CLIP ViT-L/14 (`openai/clip-vit-large-patch14`) | Local GPU |

## Implementation statement

The notebook edits the same photos with all three models and compares the results.

1. **Settings (1):** seed 42, 512 px images, 20 diffusion steps, 10 PIE-Bench images per edit type, 20 stress images, 5 ABO products, 3 misuse products.
2. **Data (2):** builds a PIE-Bench subset (100 images, 10 edit types) and downloads a sample of Amazon Berkeley Objects (ABO) product photos.
3. **Tasks (3):** 172 edit tasks per model: 100 PIE-Bench edits, 40 stress tests (the same 20 images blurred and at low resolution), 20 marketplace edits on ABO photos (white background, brighter lighting, lifestyle scene, a complex combined edit) and 12 misuse requests (fake brand logo, fake certification, fake claim, hiding damage).
4. **Models and runs (4–5):** every output is saved to `outputs/<model>/<task_id>.png` and every call is logged. Tasks with a saved output are skipped, so a run can be stopped and continued.
5. **Metrics (6–7):** CLIP similarity and directional CLIP for following the instruction; LPIPS, SSIM and PSNR outside the PIE-Bench edit mask for keeping the rest of the image; latency; mean ± std and paired bootstrap 95% intervals; stress tests; misuse outcomes.
6. **Self-evaluation (8):** reads `self_evaluation.csv` (96 rows: 20 ABO and 12 misuse tasks × 3 models).
7. **Cost (9):** monthly cost and saving for the marketplace, GPU energy.

## How to run

1. Follow the setup in the main `README.md` (virtual environment, CUDA PyTorch, `.env` with `GEMINI_API_KEY`).
2. Request PIE-Bench through the form linked in the [PnPInversion repo](https://github.com/cure-lab/PnPInversion) and unzip it so that `data/PIE-Bench_v1/mapping_file.json` and `data/PIE-Bench_v1/annotation_images/` exist. ABO downloads automatically.
3. Optional, to skip all generation and API cost: download the edited images from the Google Drive folder linked in the report and put them in `pair1_image_editing/outputs/` (folders `sd15_img2img/`, `instruct_pix2pix/`, `gemini_image/`, `stress_inputs/`). The notebook then only recomputes the metrics.
4. Run all cells.

Run time on the RTX 4060: about 1 to 1.5 hours from scratch (mostly the 172 Gemini edits), a few minutes with the saved outputs.

## Outputs

Written to `results/pair1/`:

| File | Content |
|---|---|
| `pie_summary.csv`, `pie_spread.csv`, `pie_differences.csv` | PIE-Bench metrics, mean ± std, paired bootstrap differences |
| `consistency_by_edit_type.csv`, `clip_by_edit_type.png`, `lpips_by_edit_type.png` | Results per edit type |
| `speed.csv` | Latency per model |
| `stress_summary.csv`, `stress_spread.csv`, `stress_tests.png` | Blur and low-resolution tests |
| `misuse_status.csv`, `examples_misuse.png` | Misuse requests |
| `abo_summary.csv`, `abo_by_edit.csv`, `examples_abo.png` | Marketplace (ABO) edits |
| `self_evaluation_summary.csv` | Self-evaluation averages, listing-ready and misuse rates |
| `qualitative_coding.csv` | Main failure mode of each marketplace edit, coded by hand from the rating notes (not written by the notebook) |
| `business_cost.csv` | Monthly cost and saving |
| `environment.json` | Run date, hardware and package versions |

## Self-evaluation

Scores are whole numbers from 1 to 5, rated as a seller deciding whether the photo can go on a listing. The rubric is in Appendix C of the report. `listing_ready` of 4 or more counts as ready; `misuse_applied` is `yes` or `no`.

Gemini outputs can differ slightly between runs, so the exact model IDs and run dates are recorded in the report.
