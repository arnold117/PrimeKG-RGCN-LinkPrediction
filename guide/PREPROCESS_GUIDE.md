# Preprocessing Guide

This guide documents `src/preprocess.py` for converting raw PrimeKG data into PyTorch/PyG objects used by the rest of the pipeline.

## Purpose

Create node/relation mappings, filter the graph to the relevant drug-gene-disease subgraph, convert to PyG format, and split edges into train/val/test sets.

## Usage

```bash
python src/preprocess.py --raw-data data/raw/kg.csv --processed-dir data/processed
```

## Important arguments

- `--raw-data` (default: `data/raw/kg.csv`): Path to raw PrimeKG CSV file.
- `--processed-dir` (default: `data/processed`): Directory to write processed `.pt` files.
- `--train-ratio` (default: 0.7): Training set ratio.
- `--val-ratio` (default: 0.15): Validation set ratio.
- `--test-ratio` (default: 0.15): Test set ratio.
- `--seed` (default: 42): Random seed for reproducibility.

## Outputs (written to `data/processed`)

- `full_graph.pt`: Complete graph in PyG format.
- `train_data.pt`, `val_data.pt`, `test_data.pt`: Edge splits.
- `mappings.pt`: Node and relation id mappings.
- `statistics.csv`: Basic dataset statistics.

## Notes

- Preprocessing may take time depending on the raw data size; it is a one-time cost.
- Keep `mappings.pt` and `full_graph.pt` with the model checkpoints to ensure reproducibility.
