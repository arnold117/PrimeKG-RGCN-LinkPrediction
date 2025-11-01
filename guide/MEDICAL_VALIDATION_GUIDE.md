# Medical Validation Guide

This guide documents `src/medical_validation.py` which scores predicted drug-disease links for biological and clinical plausibility.

## Purpose

Provide a reproducible pipeline to assess whether predicted drug-disease associations have supporting evidence from biological overlap, network structure, literature mentions, and clinical trials.

## Usage

```bash
python src/medical_validation.py --model_path output/models/best_model.pt --output_dir results/validation
```

## Important arguments

- `--model_path` (default: `output/models/best_model.pt`): Path to trained model checkpoint.
- `--data_dir` (default: `data/processed`): Processed data directory for mappings and graph info.
- `--top_k` (default: 50): Number of top predictions to validate.
- `--threshold` (default: 0.6): Minimum prediction score threshold.
- `--sample_diseases` (optional): Number of diseases to sample for faster execution.
- `--output_dir` (default: `results/validation`): Directory to save validation reports and CSV.
- `--output_csv` (default: `validation_results.csv`): Output CSV filename.

## Scoring components

The validation score is a weighted combination of multiple signals (default weights used by the repository):

- Prediction score (model confidence): weight 0.25
- Target overlap (shared targets/genetic evidence): weight 0.20
- Network common neighbors / connectivity: weight 0.20
- Literature evidence (co-mentions / abstracts): weight 0.20
- Clinical trials evidence: weight 0.15

These weights are configurable in the script. The final score ranges from 0 to 1.

## Outputs

- `validation_report.txt`: Summary of validation statistics and top candidates.
- `validation_results.csv`: Per-prediction validation scores and component signals.

## Notes and limitations

- Literature and trials checks may rely on local caches or simple heuristics; they are not a substitute for systematic review.
- Validation is intended as triage to prioritize candidates for expert review.
