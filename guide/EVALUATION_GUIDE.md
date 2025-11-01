# Evaluation Guide

This guide documents how to use `src/evaluate.py` to compute classification and ranking metrics for a trained model, and how the script saves outputs.

## Purpose

Compute model evaluation metrics (classification and ranking) on the test set and produce visualizations.

## Quick usage

```bash
python src/evaluate.py --model_path output/models/best_model.pt --data_dir data/processed --output_dir results
```

## Important arguments

- `--model_path` (required): Path to the PyTorch model checkpoint to evaluate.
- `--data_dir` (default: `data/processed`): Directory containing `test_data.pt` and mappings.
- `--output_dir` (default: `results`): Directory to write results and plots.
- `--batch_size` (default: 1024): Batch size for scoring edges.
- `--num_neg_samples` (default: 1): Number of negative samples per positive sample.
- `--k_values` (default: `10 50`): Hits@K values to compute (e.g., `--k_values 10 50 100`).
- `--device` (auto-detected): Device to use (cuda/cpu); auto-detects GPU if available.

## Outputs

- `results.json`: JSON file with classification and ranking metrics and basic model info.
- `metrics_summary.txt`: Human-readable summary of key metrics.
- `confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png`, `score_distribution.png`: Visualizations.

## Metrics

- Classification metrics: AUC-ROC, AUC-PR, precision, recall, F1
- Ranking metrics: MRR, mean/median rank, Hits@K

## Notes

- The script expects the data in PyTorch format produced by `src/preprocess.py`.
- Ranking metrics require enumerating candidate tails; heavy memory usage may occur for large node counts.
