# Training Guide

This guide documents `src/train.py` and how to run training with common options and outputs.

## Purpose

Train the RGCN link prediction model with checkpointing, optional gradient accumulation, and memory management utilities.

## Quick usage

```bash
python src/train.py --data_dir data/processed --output_dir output --num_epochs 100
```

## Important arguments

- `--data_dir` (default: `data/processed`): Directory with processed training/validation data.
- `--output_dir` (default: `output`): Directory to save checkpoints and final models.
- `--epochs` (default: 100): Number of training epochs.
- `--batch_size` (default: 1024): Training batch size for positive edges.
- `--num_neg_samples` (default: 1): Number of negative samples per positive edge.
- `--lr` (default: 0.001): Learning rate for optimizer.
- `--optimizer` (default: `adam`, choices: `adam`, `adamw`): Optimizer to use.
- `--gradient_accumulation_steps` (default: 1): Use to simulate larger batch sizes.
- `--embedding_dim` (default: 64): Dimension of node embeddings.
- `--hidden_dim` (default: 128): Dimension of hidden representations.
- `--dropout` (default: 0.5): Dropout rate for RGCN layers.
- `--weight_decay` (default: 0.0): L2 regularization weight decay.
- `--grad_clip` (default: 1.0): Gradient clipping value (0 to disable).

## Checkpointing

- Regular checkpoints are saved under `output/checkpoints/` (e.g., `checkpoint_epoch_10.pt`).
- The best model by validation metric is saved to `output/models/best_model.pt`.
- The final epoch model is saved to `output/models/final_model.pt`.

## Logging

- Training logs are printed to stdout and may be saved to `training.log` depending on how the script is invoked.

## Memory tips

- Reduce `batch_size` or increase `gradient_accumulation_steps` to lower peak memory.
- Use `--clear_cache` after large allocations where supported.
