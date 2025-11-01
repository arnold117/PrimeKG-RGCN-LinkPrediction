# Drug-Disease Case Studies Guide

This guide documents `src/case_studies.py` for disease-specific analyses. It summarizes usage, outputs, and interpretation.

## Overview

Generates reports for a specific disease including top drug predictions, known vs novel labels, connection paths, visualizations, and JSON results for downstream analysis.

## Basic usage

```bash
python src/case_studies.py --disease "diabetes mellitus" --top_k 10
```

## Important arguments

- `--disease` (required): Disease name for analysis (case-insensitive, partial matching supported).
- `--top_k` (default: 10): Number of top predictions to include.
- `--threshold` (default: 0.0): Minimum prediction score to include.
- `--model_path` (default: `output/models/best_model.pt`): Model checkpoint.
- `--data_dir` (default: `data/processed`): Processed data directory.
- `--output_dir` (default: `results/case_studies`): Output directory.

## Outputs per disease

- `case_study_report.txt`: Human-readable report with top predictions and interpretation.
- `prediction_scores.png`: Bar chart of prediction scores.
- `network_paths_top.png`: Network diagram of top connection paths.
- `predictions.json`: Structured results with scores and path details.

## Methodology

- Scores are computed using the trained RGCN encoder and embedding similarity.
- Path finding identifies multi-hop connection paths (up to 4 hops by default) and ranks them.

## Interpretation

- Scores ~0.85-1.00 indicate very high confidence; scores below 0.5 are speculative.
- Known treatments validate model behavior; novel predictions require external validation.

## Notes

- Path-finding and visualization are the most time-consuming steps.
- Ensure the `data/processed` directory exists and the requested model checkpoint is available.
