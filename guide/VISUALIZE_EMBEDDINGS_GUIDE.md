# Embedding Visualization Guide

This guide documents `src/visualize_embeddings.py` for extracting and visualizing learned node embeddings.

## Purpose

Reduce embeddings with t-SNE or UMAP, cluster embeddings, and produce static or interactive visualizations to inspect learned representations.

## Usage

```bash
python src/visualize_embeddings.py --model_path output/models/best_model.pt --output_dir results/embeddings
```

## Important arguments

- `--model_path` (default: `output/models/best_model.pt`): Path to model checkpoint.
- `--data_dir` (default: `data/processed`): Directory containing processed data.
- `--output_dir` (default: `results/embeddings`): Output directory for results.
- `--method` (default: `tsne`, choices: `tsne`, `umap`): Dimensionality reduction method.
- `--sample_size` (optional): Number of nodes to sample for visualization.
- `--query` (optional): Query node name for nearest neighbor analysis.
- `--k_neighbors` (default: 10): Number of nearest neighbors to find.
- `--cluster` (flag): Perform clustering analysis.
- `--n_clusters` (default: 10): Number of clusters for clustering analysis.
- `--skip_interactive` (flag): Skip interactive HTML plot generation.

## Outputs

- `embeddings_tsne.png` or `embeddings_umap.png`: Scatter plot of reduced embeddings.
- `clustering_summary.txt`: Cluster composition summaries.
- `nearest_neighbors.json`: Top-k neighbors for sample nodes.

## Notes

- UMAP typically runs faster than t-SNE on large sample sizes but may produce slightly different structure.
