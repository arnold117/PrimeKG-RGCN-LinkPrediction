"""
Relation-wise Performance Analysis for PrimeKG RGCN Model

Analyzes model performance broken down by:
1. Drug-gene sub-relations: target, enzyme, transporter, carrier
2. Edge statistics and distribution

Usage:
    python src/relation_analysis.py
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.rgcn import DrugDiseaseRGCN, LinkPredictor


def load_raw_drug_gene_edges(raw_path: str = "data/raw/kg.csv") -> pd.DataFrame:
    """Load raw PrimeKG data and extract drug-gene edges with sub-relations."""
    print(f"Loading raw data from {raw_path}...")

    df = pd.read_csv(raw_path)

    # Filter to drug_protein edges only
    drug_gene = df[df['relation'] == 'drug_protein'].copy()

    # Create mapping from (drug_id, gene_id) -> display_relation
    drug_gene['drug_id'] = drug_gene['x_id']
    drug_gene['gene_id'] = drug_gene['y_id'].astype(str)
    drug_gene['sub_relation'] = drug_gene['display_relation']

    print(f"Found {len(drug_gene)} drug-gene edges")
    print("\nSub-relation distribution:")
    print(drug_gene['sub_relation'].value_counts())

    return drug_gene[['drug_id', 'gene_id', 'sub_relation', 'x_name', 'y_name']]


def load_model_and_data():
    """Load trained model and test data."""

    # Load graph data
    print("\nLoading model and data...")
    graph_data = torch.load('data/processed/full_graph.pt', weights_only=False)
    mappings = torch.load('data/processed/mappings.pt', weights_only=False)
    test_data = torch.load('data/processed/test_data.pt', weights_only=False)
    checkpoint = torch.load('output/models/best_model.pt', map_location='cpu', weights_only=False)

    # Detect architecture
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    has_layer_norm = any('norm1' in k for k in state_dict.keys())

    # Create model
    encoder = DrugDiseaseRGCN(
        num_nodes=graph_data['num_nodes'],
        num_relations=graph_data['num_relations'],
        embedding_dim=64,
        hidden_dim=128,
        dropout=0.5,
        use_layer_norm=has_layer_norm,
        use_skip_connections=True
    )

    decoder = LinkPredictor(
        num_relations=graph_data['num_relations'],
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

    return model, graph_data, mappings, test_data


def map_test_edges_to_subrelations(
    test_data: dict,
    mappings: dict,
    raw_edges: pd.DataFrame
) -> dict:
    """Map test edges to their sub-relations using raw data."""

    idx2node = mappings['idx2node']
    edge_index = test_data['edge_index']

    # Create lookup from (drug_id, gene_id) -> sub_relation
    edge_to_subrel = {}
    for _, row in raw_edges.iterrows():
        key = (row['drug_id'], str(row['gene_id']))
        edge_to_subrel[key] = row['sub_relation']

    # Map test edges
    subrel_edges = defaultdict(list)
    unmapped = 0

    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()

        # Get node IDs
        src_id, src_name, src_type = idx2node[src_idx]
        dst_id, dst_name, dst_type = idx2node[dst_idx]

        # Determine drug and gene
        if src_type == 'drug':
            drug_id, gene_id = src_id, str(dst_id)
        else:
            drug_id, gene_id = dst_id, str(src_id)

        # Lookup sub-relation
        key = (drug_id, gene_id)
        if key in edge_to_subrel:
            subrel = edge_to_subrel[key]
            subrel_edges[subrel].append(i)
        else:
            unmapped += 1

    print(f"\nMapped {sum(len(v) for v in subrel_edges.values())} test edges")
    print(f"Unmapped: {unmapped}")

    for subrel, indices in subrel_edges.items():
        print(f"  {subrel}: {len(indices)}")

    return subrel_edges


def evaluate_by_subrelation(
    model,
    graph_data: dict,
    test_data: dict,
    subrel_edges: dict,
    num_neg_samples: int = 100
) -> pd.DataFrame:
    """Evaluate model performance for each sub-relation type."""

    print("\n" + "="*60)
    print("Evaluating by sub-relation type")
    print("="*60)

    # Get embeddings
    with torch.no_grad():
        embeddings = model.encoder(
            graph_data['edge_index'],
            graph_data['edge_type']
        )

    edge_index = test_data['edge_index']
    edge_type = test_data['edge_type']
    num_nodes = graph_data['num_nodes']

    results = []

    for subrel, indices in subrel_edges.items():
        if len(indices) < 10:
            print(f"Skipping {subrel}: too few edges ({len(indices)})")
            continue

        # Get positive edges for this sub-relation
        pos_edges = edge_index[:, indices]
        pos_types = edge_type[indices]

        # Score positive edges
        src_emb = embeddings[pos_edges[0]]
        dst_emb = embeddings[pos_edges[1]]

        with torch.no_grad():
            pos_scores = model.decoder(src_emb, dst_emb, pos_types).squeeze()

        # Generate negative samples
        neg_scores_list = []
        for _ in range(num_neg_samples // len(indices) + 1):
            neg_dst = torch.randint(0, num_nodes, (len(indices),))
            neg_dst_emb = embeddings[neg_dst]
            with torch.no_grad():
                neg_scores = model.decoder(src_emb, neg_dst_emb, pos_types).squeeze()
            neg_scores_list.append(neg_scores)

        neg_scores = torch.cat(neg_scores_list)[:num_neg_samples]

        # Compute metrics
        all_scores = torch.cat([pos_scores, neg_scores]).numpy()
        all_labels = np.concatenate([
            np.ones(len(pos_scores)),
            np.zeros(len(neg_scores))
        ])

        auc = roc_auc_score(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)

        # Hits@K
        all_pos_scores = pos_scores.numpy()
        all_neg_scores = neg_scores.numpy()

        hits_at_10 = 0
        hits_at_50 = 0

        for ps in all_pos_scores:
            rank = (all_neg_scores >= ps).sum() + 1
            if rank <= 10:
                hits_at_10 += 1
            if rank <= 50:
                hits_at_50 += 1

        hits_at_10 /= len(all_pos_scores)
        hits_at_50 /= len(all_pos_scores)

        results.append({
            'sub_relation': subrel,
            'num_edges': len(indices),
            'AUC-ROC': auc,
            'AP': ap,
            'Hits@10': hits_at_10,
            'Hits@50': hits_at_50,
            'mean_pos_score': pos_scores.mean().item(),
            'mean_neg_score': neg_scores.mean().item()
        })

        print(f"\n{subrel} ({len(indices)} edges):")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  AP: {ap:.4f}")
        print(f"  Hits@10: {hits_at_10:.4f}")
        print(f"  Hits@50: {hits_at_50:.4f}")

    return pd.DataFrame(results)


def analyze_edge_properties(
    raw_edges: pd.DataFrame,
    mappings: dict
) -> None:
    """Analyze properties of different edge types."""

    print("\n" + "="*60)
    print("Edge Property Analysis")
    print("="*60)

    # Count unique drugs and genes per sub-relation
    for subrel in raw_edges['sub_relation'].unique():
        subset = raw_edges[raw_edges['sub_relation'] == subrel]
        n_drugs = subset['drug_id'].nunique()
        n_genes = subset['gene_id'].nunique()
        avg_genes_per_drug = len(subset) / n_drugs

        print(f"\n{subrel}:")
        print(f"  Edges: {len(subset):,}")
        print(f"  Unique drugs: {n_drugs:,}")
        print(f"  Unique genes: {n_genes:,}")
        print(f"  Avg genes/drug: {avg_genes_per_drug:.1f}")


def main():
    """Run relation-wise analysis."""

    print("="*60)
    print("Relation-wise Performance Analysis")
    print("="*60)

    # Load raw data with sub-relations
    raw_edges = load_raw_drug_gene_edges()

    # Analyze edge properties
    analyze_edge_properties(raw_edges, None)

    # Load model and test data
    model, graph_data, mappings, test_data = load_model_and_data()

    # Map test edges to sub-relations
    subrel_edges = map_test_edges_to_subrelations(test_data, mappings, raw_edges)

    # Evaluate by sub-relation
    results_df = evaluate_by_subrelation(
        model, graph_data, test_data, subrel_edges
    )

    # Save results
    output_path = Path("output/analysis")
    output_path.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_path / "relation_analysis.csv", index=False)
    print(f"\nResults saved to {output_path / 'relation_analysis.csv'}")

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(results_df.to_string(index=False))

    return results_df


if __name__ == "__main__":
    main()
