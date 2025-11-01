# Explain Predictions Guide

This guide documents `src/explain_predictions.py`, which generates path-based explanations and visualizations for individual predictions.

## Purpose

Identify multi-hop connecting paths between drugs and diseases and produce ranked, human-readable explanations with optional network visualizations.

## Usage

```bash
python src/explain_predictions.py --model_path output/models/best_model.pt --prediction_id 12345 --output_dir results/explanations
```

## Important arguments

- `--drug` (required): Drug name to analyze.
- `--disease` (required): Disease name to analyze.
- `--model_path` (default: `output/models/best_model.pt`): Path to model checkpoint.
- `--data_dir` (default: `data/processed`): Directory containing processed data.
- `--top_k` (default: 5): Number of top paths to explain.
- `--output_dir` (default: `results/explanations`): Directory to save explanation JSON and visualizations.

## Outputs

- `explanation_<id>.json`: Structured explanation with paths and scores.
- `explanation_<id>.png`: Optional network visualization for top paths.

## Notes

- Path scoring uses heuristics combining path length, edge weights, and node importance.
- Explanations are intended to aid hypothesis generation and should be reviewed by domain experts.
