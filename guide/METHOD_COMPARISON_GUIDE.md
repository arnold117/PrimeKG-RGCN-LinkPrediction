# Method Comparison Guide

This guide documents `src/compare_methods.py` for running baseline comparisons (Random, Node Degree, TransE) and the RGCN method.

## Usage

```bash
python src/compare_methods.py --model_path output/models/best_model.pt --data_dir data/processed --output_dir results/comparison
```

## Important arguments

- `--model_path` (default: `output/models/best_model.pt`): Path to trained model for the RGCN method.
- `--data_dir` (default: `data/processed`): Directory containing processed dataset.
- `--output_dir` (default: `results/comparison`): Directory to write comparison results.
- `--methods` (default: `random degree rgcn`): Methods to compare (choices: `random`, `degree`, `transe`, `rgcn`).
- `--frequency_analysis` (flag): Analyze performance by disease frequency.
- `--statistical_tests` (flag): Perform statistical significance tests.
- `--transe_epochs` (default: 50): Number of epochs for TransE training.

## Outputs

- `test_results.csv`: Tabular comparison of metrics for each method.
- Plots comparing AUC-ROC, Hits@K, and other metrics across methods.
- Optional LaTeX/Markdown tables for paper inclusion.

## Notes

- Baseline implementations are lightweight and may use simplified training/estimation.
- Statistical significance testing is provided as a placeholder and should be adapted to the experimental design.
