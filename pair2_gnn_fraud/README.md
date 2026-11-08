# Pair 2: Fake Reviewer Detection with Graph Neural Networks

Notebook: `pair2_gnn_fraud.ipynb`

| Role | Model |
|---|---|
| Baseline | MLP (account features only, no graph) |
| System A | GraphSAGE (mean aggregator), plus GCN as a second established model |
| System B | FAGCN (frequency-adaptive GCN, written by hand in PyTorch) |

All models run on the local GPU. No extra graph library (DGL or PyG) is needed.

## Implementation statement

The notebook trains the four models to flag fake reviewers in the Amazon review graph and compares them.

1. **Settings (1):** split seed 42, training seeds 0, 1 and 2, hidden size 64, dropout 0.5, Adam with learning rate 0.01 and weight decay 5e-4, up to 200 epochs with early stopping (patience 30 on validation AUC-PR), FAGCN epsilon 0.3 and 2 layers.
2. **Data (2):** downloads `FraudAmazon.zip` from DGL, loads the three user relations (U-P-U, U-S-U, U-V-U) and their union, keeps the labelled users, and makes a 40/20/40 stratified split. Features are scaled using the training users only.
3. **Homophily (3):** measures how often fraudsters link to other fraudsters, which explains why a heterophily model (FAGCN) was chosen for System B.
4. **Training (5):** weighted cross-entropy for the class imbalance; results are mean ± std over three seeds (AUC-PR, AUC-ROC, F1, recall at 1% false-positive rate, training time, parameters).
5. **Stress tests (6):** camouflage (fraud–fraud links rewired to genuine users, keeping node degree), feature noise (std 0.5) and few labels (10% of the training users).
6. **Fairness and trust (7):** results by user degree group, calibration (Brier score, ECE, reliability diagram) and permutation feature importance.
7. **Cost (8):** monthly cost of missed fraudsters, wrong suspensions and moderator reviews for 100,000 accounts, and training energy.
8. **Self-evaluation (9):** reads `self_evaluation.csv` (one row per model).

## How to run

1. Follow the setup in the main `README.md`. No API key is needed.
2. Run all cells. The dataset downloads into `data/amazon_fraud/` on the first run.

Run time on the RTX 4060: about 10 to 20 minutes.

## Outputs

Written to `results/pair2/`:

| File | Content |
|---|---|
| `splits.csv`, `homophily.csv` | Split sizes and fraud-neighbour shares |
| `training_runs.csv`, `technical_summary.csv`, `technical_means.csv`, `technical_metrics.png` | Results per seed and mean ± std |
| `stress_results.csv`, `stress_tests.png` | Camouflage, noise and few-label tests |
| `fairness_by_degree.csv` | Recall and false-positive rate by degree group |
| `calibration.csv`, `calibration.png` | Brier score, ECE, reliability diagram |
| `feature_importance.csv`, `feature_importance_summary.csv`, `feature_importance.png` | Which features drive the flags |
| `business_cost.csv` | Monthly cost per model |
| `environment.json` | Run date, hardware and package versions |

## Self-evaluation

Scores are whole numbers from 1 to 5, rated as a moderator who reviews flagged accounts, on usefulness of flags, explainability, robustness, fairness and ease of running. The rubric is in Appendix C of the report.

GPU results can differ slightly from CPU results in the last decimal places.
