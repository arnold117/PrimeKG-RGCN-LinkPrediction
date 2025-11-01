# Run Full Analysis Guide

This guide documents `src/run_full_analysis.py`, the orchestrator script that runs multiple analyses sequentially or selectively.

## Purpose

Run one or more analyses (evaluate, analyze, errors, case_studies, embeddings, explanations, validation, comparison, failures) using a trained model and collect outputs in a target directory.

## Basic usage

```bash
python src/run_full_analysis.py --model_path output/models/best_model.pt --output_base results
```

## Important arguments

- `--model_path` (default: `output/models/best_model.pt`): Model checkpoint used by analyses.
- `--output_dir` (default: `results`): Base directory where per-analysis outputs are written.
- `--data_dir` (default: `data/processed`): Directory containing processed data.
- `--python_path` (default: `./venv/bin/python`): Path to the Python executable to use for subprocess calls.
- `--analyses` (optional): Specific analyses to run. If omitted, runs all available analyses.
- `--skip` (optional): Analyses to skip.
- `--timeout` (default: 300): Maximum time per analysis in seconds.
- `--list` (flag): List all available analyses and exit.

## Notes

- The script invokes other scripts via subprocess; ensure the working environment has required packages installed.
- Check logs produced under the analysis output directories for per-script details.
