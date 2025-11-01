# Failure Analysis Guide

This guide documents `src/analyze_failures.py` which performs failure-mode analysis to understand false positives and false negatives.

## Purpose

Identify and characterize failure cases, compare failure subgraphs with success cases, and generate hypotheses for model improvement.

## Usage

```bash
python src/analyze_failures.py --model_path output/models/best_model.pt --data_dir data/processed --output_dir results/failure_analysis
```

## Important arguments

- `--model_path` (default: `output/models/best_model.pt`): Path to model checkpoint.
- `--data_dir` (default: `data/processed`): Directory with processed data.
- `--num_failures` (default: 5): Number of worst failures to analyze.
- `--num_successes` (default: 5): Number of best successes to analyze.
- `--num_samples` (default: 5000): Number of samples to evaluate.
- `--visualize_subgraphs` (flag): Generate subgraph visualizations.
- `--output_dir` (default: `results/failure_analysis`): Directory to write failure analysis outputs.

## Outputs

- `failure_analysis_report.txt`: Summary with categories of failures and suggested fixes.
- `false_negatives.csv`, `false_positives.csv`: Lists of problematic edges.
- Subgraph visualizations for selected failures.

## Notes

- Failure analysis is exploratory; recommended to iterate with different thresholds and filters.
