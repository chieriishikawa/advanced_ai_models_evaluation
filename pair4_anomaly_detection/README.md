# Pair 4: Payment Fraud Detection with Generative Models

Notebook: `pair4_anomaly_detection.ipynb`

| Role | Model | Fraud score |
|---|---|---|
| Baseline | Isolation Forest (200 trees, CPU) | Isolation score |
| System A | Variational autoencoder (hidden 64, latent 8) | Reconstruction error |
| System B | RealNVP normalizing flow (6 affine coupling layers) | Negative log-likelihood |

All three models learn from normal payments only and flag payments that look unusual.

## Implementation statement

The notebook trains the three detectors on credit card payments and compares them.

1. **Settings (1):** split seed 42, training seeds 0, 1 and 2, 30 epochs, batch 512, learning rate 1e-3, features clipped to ±10.
2. **Data and splits (2):** downloads the ULB credit card fraud data, uses V1 to V28 plus the log amount, and makes two splits: a random 60/20/20 stratified split and a time-based split (train on earlier payments, test on later ones) for drift. Features are standardised on the training data.
3. **Training (4):** each model is trained on normal training payments; the threshold is chosen on the validation set. Results are mean ± std over three seeds (AUC-PR, AUC-ROC, F1, recall at 1% false-positive rate, training time).
4. **Stress tests (5):** the time split (drift) and Gaussian noise of 0.1, 0.3 and 0.5 std on the test payments. §5.1 retrains the flow with noise of 0.1 std added during training to test a fix for its collapse under noise.
5. **Fairness and explainability (6):** results by payment amount band; VAE per-feature reconstruction error vs flow gradient saliency.
6. **Cost (7):** monthly cost of chargebacks, false declines and reviews for 600,000 payments, a threshold cost curve, and training energy.
7. **Self-evaluation (8):** reads `self_evaluation.csv` (one row per model).

## How to run

1. Follow the setup in the main `README.md`. No API key is needed.
2. Run all cells. The dataset downloads into `data/credit_card/` on the first run.

Run time on the RTX 4060: about 5 to 10 minutes.

## Outputs

Written to `results/pair4/`:

| File | Content |
|---|---|
| `splits.csv` | Split sizes and fraud rates |
| `training_runs.csv`, `technical_summary.csv`, `technical_means.csv` | Results per seed and mean ± std |
| `pr_curves_and_training.png`, `score_distributions.png` | PR curves, training loss, score distributions |
| `stress_results.csv`, `stress_tests.png` | Drift and noise tests |
| `noise_training_fix.csv` | Flow trained with and without noise |
| `fairness_by_amount.csv` | Recall and false-positive rate by amount band |
| `feature_importance.csv`, `feature_importance.png` | Which features drive the score |
| `business_cost.csv`, `threshold_cost_curve.png` | Monthly cost and threshold choice |
| `environment.json` | Run date, hardware and package versions |

## Self-evaluation

Scores are whole numbers from 1 to 5, rated as a fraud analyst who handles blocked payments, on catching costly fraud, explainability, robustness to drift, fairness and ease of running. The rubric is in Appendix C of the report.

GPU results can differ slightly between runs in the last decimal places.
