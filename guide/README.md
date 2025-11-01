# Guide Directory

This directory contains comprehensive usage guides for all scripts in the PrimeKG-RGCN-LinkPrediction project.

## Quick Reference

### Core Pipeline Scripts

| Script | Guide | Purpose |
|--------|-------|---------|
| `src/preprocess.py` | [PREPROCESS_GUIDE.md](PREPROCESS_GUIDE.md) | Convert raw PrimeKG data to PyTorch format |
| `src/train.py` | [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Train RGCN model with checkpointing |
| `src/evaluate.py` | [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) | Compute metrics and generate visualizations |
| `src/run_full_analysis.py` | [RUN_FULL_ANALYSIS_GUIDE.md](RUN_FULL_ANALYSIS_GUIDE.md) | Run complete analysis pipeline |

### Analysis Scripts

| Script | Guide | Purpose |
|--------|-------|---------|
| `src/case_studies.py` | [CASE_STUDIES_GUIDE.md](CASE_STUDIES_GUIDE.md) | Disease-specific drug predictions |
| `src/medical_validation.py` | [MEDICAL_VALIDATION_GUIDE.md](MEDICAL_VALIDATION_GUIDE.md) | Biological plausibility validation |
| `src/compare_methods.py` | [METHOD_COMPARISON_GUIDE.md](METHOD_COMPARISON_GUIDE.md) | Baseline method comparison |
| `src/explain_predictions.py` | [EXPLAIN_PREDICTIONS_GUIDE.md](EXPLAIN_PREDICTIONS_GUIDE.md) | Path-based interpretability |
| `src/error_analysis.py` | [ERROR_ANALYSIS_GUIDE.md](ERROR_ANALYSIS_GUIDE.md) | Error pattern analysis |
| `src/analyze_failures.py` | [ANALYZE_FAILURES_GUIDE.md](ANALYZE_FAILURES_GUIDE.md) | Failure mode investigation |
| `src/visualize_embeddings.py` | [VISUALIZE_EMBEDDINGS_GUIDE.md](VISUALIZE_EMBEDDINGS_GUIDE.md) | Embedding visualization |

### Model Documentation

| Component | Guide | Purpose |
|-----------|-------|---------|
| `src/models/rgcn.py` | [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) | RGCN architecture and implementation |

## Typical Workflows

### First-Time Setup
1. [PREPROCESS_GUIDE.md](PREPROCESS_GUIDE.md) - Prepare data
2. [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Train model
3. [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - Evaluate performance

### Comprehensive Analysis
Use [RUN_FULL_ANALYSIS_GUIDE.md](RUN_FULL_ANALYSIS_GUIDE.md) to run all analyses automatically.

### Drug Repurposing Research
1. [CASE_STUDIES_GUIDE.md](CASE_STUDIES_GUIDE.md) - Identify candidates
2. [EXPLAIN_PREDICTIONS_GUIDE.md](EXPLAIN_PREDICTIONS_GUIDE.md) - Understand mechanisms
3. [MEDICAL_VALIDATION_GUIDE.md](MEDICAL_VALIDATION_GUIDE.md) - Validate findings

### Model Debugging
1. [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - Check overall metrics
2. [ERROR_ANALYSIS_GUIDE.md](ERROR_ANALYSIS_GUIDE.md) - Identify error patterns
3. [ANALYZE_FAILURES_GUIDE.md](ANALYZE_FAILURES_GUIDE.md) - Deep dive failures
4. [VISUALIZE_EMBEDDINGS_GUIDE.md](VISUALIZE_EMBEDDINGS_GUIDE.md) - Inspect embeddings

## Guide Standards

All guides follow a consistent structure:

1. **Purpose** - What the script does
2. **Quick usage** - Minimal example command
3. **Important arguments** - CLI parameters with defaults
4. **Outputs** - Generated files and formats
5. **Notes** - Limitations and tips

## Documentation Maintenance

When modifying scripts:
- Update the corresponding guide with new arguments or defaults
- Keep CLI examples in sync with actual code
- Maintain emoji-free, concise technical style
- Test all example commands before committing

## Additional Resources

- Main project documentation: [../README.md](../README.md)
- Documentation review summary: [../DOCUMENTATION_REVIEW_SUMMARY.md](../DOCUMENTATION_REVIEW_SUMMARY.md)
- Requirements: [../requirements.txt](../requirements.txt)
