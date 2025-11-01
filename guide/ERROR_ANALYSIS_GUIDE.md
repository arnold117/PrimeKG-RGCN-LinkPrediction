# Error Analysis Guide

This guide documents `src/error_analysis.py` which inspects false positives and false negatives and summarizes error patterns.

## Purpose

Produce CSVs and visualizations highlighting model errors, confidence distributions, and correlations with graph features.

## Usage

```bash
python src/error_analysis.py --model_path output/models/best_model.pt --output_dir results/error_analysis
```

## Important arguments

- `--model_path` (required): Trained model checkpoint.
- `--data_dir` (default: `data/processed`): Directory containing processed data.
- `--output_dir` (default: `results/error_analysis`): Directory to save error analysis outputs.
- `--threshold` (default: 0.5): Classification threshold for predictions.
- `--top_k` (default: 20): Number of top errors to include in report.
- `--device` (auto-detected): Device to use (cuda/cpu).

## Outputs

- `false_positives.csv`, `false_negatives.csv`: Lists of mispredicted edges with scores.
- `error_report.txt`: Summary of observed error patterns.
- Visualizations: histograms, confusion matrices, and per-node-type breakdowns.

## Notes

- Combine this output with failure analysis and case studies to investigate root causes.
