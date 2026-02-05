"""
Strict Evaluation with Hard Negative Sampling

Issues with original evaluation:
1. Test edges' reverse direction exists in training set (data leakage)
2. Random negative sampling is too easy

This script uses:
1. Only truly unseen edges for testing
2. Hard negative sampling (same node type, similar degree)

Usage:
    python src/strict_evaluation.py --model rgcn
    python src/strict_evaluation.py --model mlp --checkpoint output/mlp/models/best_model.pt
"""

import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import get_encoder, LinkPredictor, list_models


def load_all_data():
    """Load all data and identify truly unseen test edges."""
    print("Loading data...")

    train = torch.load('data/processed/train_data.pt', weights_only=False)
    test = torch.load('data/processed/test_data.pt', weights_only=False)
    graph = torch.load('data/processed/full_graph.pt', weights_only=False)
    mappings = torch.load('data/processed/mappings.pt', weights_only=False)

    # Build set of ALL edges in training (both directions)
    train_edges = set()
    for i in range(train['edge_index'].shape[1]):
        src = train['edge_index'][0, i].item()
        dst = train['edge_index'][1, i].item()
        train_edges.add((src, dst))
        train_edges.add((dst, src))

    # Find truly unseen test edges
    unseen_edges = []
    leaked_edges = []

    for i in range(test['edge_index'].shape[1]):
        src = test['edge_index'][0, i].item()
        dst = test['edge_index'][1, i].item()
        edge_type = test['edge_type'][i].item()

        if (src, dst) not in train_edges and (dst, src) not in train_edges:
            unseen_edges.append((src, dst, edge_type))
        else:
            leaked_edges.append((src, dst, edge_type))

    print(f"\n=== Data Leakage Analysis ===")
    print(f"Total test edges: {test['edge_index'].shape[1]}")
    print(f"Truly unseen: {len(unseen_edges)} ({len(unseen_edges)/test['edge_index'].shape[1]*100:.1f}%)")
    print(f"Leaked (reverse in train): {len(leaked_edges)} ({len(leaked_edges)/test['edge_index'].shape[1]*100:.1f}%)")

    return train, test, graph, mappings, unseen_edges, leaked_edges


def get_node_info(mappings):
    """Get node type and degree information."""
    idx2node = mappings['idx2node']

    node_types = {}
    for idx, (node_id, node_name, node_type) in idx2node.items():
        node_types[idx] = node_type

    return node_types


def generate_hard_negatives(
    positive_edges: list,
    graph: dict,
    mappings: dict,
    num_neg_per_pos: int = 50
):
    """Generate hard negative samples.

    Strategy:
    1. For each positive (drug, gene) edge
    2. Replace gene with another gene that:
       - Is NOT connected to this drug
       - Has similar degree (harder to distinguish)
    """
    print("\nGenerating hard negative samples...")

    edge_index = graph['edge_index']
    num_nodes = graph['num_nodes']
    idx2node = mappings['idx2node']

    # Build adjacency and node info
    adjacency = defaultdict(set)
    node_types = {}
    node_degrees = defaultdict(int)

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        adjacency[src].add(dst)
        adjacency[dst].add(src)
        node_degrees[src] += 1
        node_degrees[dst] += 1

    for idx, (node_id, node_name, node_type) in idx2node.items():
        node_types[idx] = node_type

    # Group nodes by type
    nodes_by_type = defaultdict(list)
    for idx, ntype in node_types.items():
        nodes_by_type[ntype].append(idx)

    # Generate negatives
    negative_edges = []

    for src, dst, edge_type in positive_edges:
        src_type = node_types.get(src, 'unknown')
        dst_type = node_types.get(dst, 'unknown')

        # Find candidate nodes of same type as dst
        candidates = nodes_by_type[dst_type]

        # Filter: not already connected to src
        valid_candidates = [c for c in candidates if c not in adjacency[src]]

        if len(valid_candidates) < num_neg_per_pos:
            # Fall back to random sampling
            sampled = random.sample(valid_candidates, min(len(valid_candidates), num_neg_per_pos))
        else:
            # Prefer nodes with similar degree (harder negatives)
            dst_degree = node_degrees[dst]
            candidates_with_degree = [(c, abs(node_degrees[c] - dst_degree)) for c in valid_candidates]
            candidates_with_degree.sort(key=lambda x: x[1])
            sampled = [c for c, _ in candidates_with_degree[:num_neg_per_pos]]

        for neg_dst in sampled:
            negative_edges.append((src, neg_dst, edge_type))

    print(f"Generated {len(negative_edges)} hard negative samples")
    return negative_edges


def load_model(graph, model_type: str = 'rgcn', checkpoint_path: str = None):
    """Load trained model.

    Args:
        graph: Graph data dict
        model_type: Model type ('rgcn', 'mlp', 'gcn', 'gat')
        checkpoint_path: Path to checkpoint (default: output/models/best_model.pt)
    """
    if checkpoint_path is None:
        checkpoint_path = 'output/models/best_model.pt'

    print(f"Loading {model_type.upper()} from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state_dict = checkpoint.get('model_state_dict', checkpoint)
    has_layer_norm = any('norm1' in k for k in state_dict.keys())

    # Get encoder class based on model type
    encoder_cls = get_encoder(model_type)

    encoder = encoder_cls(
        num_nodes=graph['num_nodes'],
        num_relations=graph['num_relations'],
        embedding_dim=64,
        hidden_dim=128,
        dropout=0.5,
        use_layer_norm=has_layer_norm,
        use_skip_connections=True
    )

    decoder = LinkPredictor(
        num_relations=graph['num_relations'],
        embedding_dim=128,
        dropout=0.0
    )

    class Model(torch.nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder

    model = Model(encoder, decoder)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def evaluate(model, graph, positive_edges, negative_edges, batch_size: int = 4096):
    """Evaluate with strict metrics (GPU accelerated)."""
    device = get_device()
    print(f"\nEvaluating on {device}...")

    model = model.to(device)
    edge_index = graph['edge_index'].to(device)
    edge_type = graph['edge_type'].to(device)

    # Get embeddings once
    with torch.no_grad():
        embeddings = model.encoder(edge_index, edge_type)

    # Convert edges to tensors
    pos_src = torch.tensor([e[0] for e in positive_edges], device=device)
    pos_dst = torch.tensor([e[1] for e in positive_edges], device=device)
    pos_rel = torch.tensor([e[2] for e in positive_edges], device=device)

    neg_src = torch.tensor([e[0] for e in negative_edges], device=device)
    neg_dst = torch.tensor([e[1] for e in negative_edges], device=device)
    neg_rel = torch.tensor([e[2] for e in negative_edges], device=device)

    # Batch scoring for positives
    pos_scores = []
    with torch.no_grad():
        for start in range(0, len(pos_src), batch_size):
            end = min(start + batch_size, len(pos_src))
            src_emb = embeddings[pos_src[start:end]]
            dst_emb = embeddings[pos_dst[start:end]]
            rel = pos_rel[start:end]
            scores = model.decoder(src_emb, dst_emb, rel)
            pos_scores.append(scores.cpu())
    pos_scores = torch.cat(pos_scores).numpy()

    # Batch scoring for negatives
    neg_scores = []
    with torch.no_grad():
        for start in range(0, len(neg_src), batch_size):
            end = min(start + batch_size, len(neg_src))
            src_emb = embeddings[neg_src[start:end]]
            dst_emb = embeddings[neg_dst[start:end]]
            rel = neg_rel[start:end]
            scores = model.decoder(src_emb, dst_emb, rel)
            neg_scores.append(scores.cpu())
    neg_scores = torch.cat(neg_scores).numpy()

    print(f"Scored {len(pos_scores)} positive and {len(neg_scores)} negative edges")

    # Metrics
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)

    # Hits@K and MRR (vectorized)
    neg_per_pos = len(neg_scores) // len(pos_scores)
    neg_scores_matrix = neg_scores.reshape(len(pos_scores), neg_per_pos)

    # Compute ranks: how many negatives score >= positive
    ranks = (neg_scores_matrix >= pos_scores[:, None]).sum(axis=1) + 1

    hits_at_10 = (ranks <= 10).mean()
    hits_at_50 = (ranks <= 50).mean()
    mrr = (1.0 / ranks).mean()

    return {
        'auc_roc': auc,
        'ap': ap,
        'hits_at_10': hits_at_10,
        'hits_at_50': hits_at_50,
        'mrr': mrr,
        'mean_pos_score': pos_scores.mean(),
        'mean_neg_score': neg_scores.mean(),
        'num_pos': len(pos_scores),
        'num_neg': len(neg_scores)
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Strict evaluation with hard negative sampling'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='rgcn',
        choices=list_models(),
        help=f'Model type to evaluate. Available: {list_models()}'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (default: output/models/best_model.pt)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/analysis',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num_neg',
        type=int,
        default=50,
        help='Number of negative samples per positive'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print(f"Strict Evaluation: {args.model.upper()}")
    print("="*60)

    # Load data
    train, test, graph, mappings, unseen_edges, leaked_edges = load_all_data()

    # Load model
    model = load_model(graph, model_type=args.model, checkpoint_path=args.checkpoint)

    # Generate hard negatives for unseen edges
    hard_negatives = generate_hard_negatives(
        unseen_edges, graph, mappings, num_neg_per_pos=args.num_neg
    )

    # Evaluate on truly unseen edges
    results_unseen = evaluate(model, graph, unseen_edges, hard_negatives)
    results_unseen['model'] = args.model

    print("\n" + "="*60)
    print(f"RESULTS: {args.model.upper()} - Truly Unseen Edges + Hard Negatives")
    print("="*60)
    print(f"Positive edges: {results_unseen['num_pos']}")
    print(f"Negative edges: {results_unseen['num_neg']}")
    print(f"AUC-ROC: {results_unseen['auc_roc']:.4f}")
    print(f"AP: {results_unseen['ap']:.4f}")
    print(f"Hits@10: {results_unseen['hits_at_10']:.4f}")
    print(f"Hits@50: {results_unseen['hits_at_50']:.4f}")
    print(f"MRR: {results_unseen['mrr']:.4f}")
    print(f"Mean pos score: {results_unseen['mean_pos_score']:.4f}")
    print(f"Mean neg score: {results_unseen['mean_neg_score']:.4f}")

    # Compare with leaked edges (for reference) - only for first run
    if len(leaked_edges) > 0 and args.model == 'rgcn':
        print("\n" + "="*60)
        print("COMPARISON: Leaked Edges (should be high)")
        print("="*60)
        leaked_negs = generate_hard_negatives(leaked_edges[:1000], graph, mappings, num_neg_per_pos=args.num_neg)
        results_leaked = evaluate(model, graph, leaked_edges[:1000], leaked_negs)
        print(f"AUC-ROC: {results_leaked['auc_roc']:.4f}")
        print(f"Hits@10: {results_leaked['hits_at_10']:.4f}")

    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save with model name in filename
    results_df = pd.DataFrame([results_unseen])
    output_file = output_path / f"strict_evaluation_{args.model}.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nResults saved to {output_file}")

    return results_unseen


if __name__ == "__main__":
    main()
