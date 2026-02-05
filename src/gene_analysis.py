"""
Gene-Centric Analysis for PrimeKG

Analyzes:
1. Hub genes: genes connected to many drugs and diseases
2. Drug-Gene-Disease pathways
3. Gene importance in link prediction

Usage:
    python src/gene_analysis.py
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_graph_data():
    """Load graph and mappings."""
    print("Loading graph data...")

    graph = torch.load('data/processed/full_graph.pt', weights_only=False)
    mappings = torch.load('data/processed/mappings.pt', weights_only=False)

    print(f"Nodes: {graph['num_nodes']}")
    print(f"Edges: {graph['edge_index'].shape[1]}")
    print(f"Relations: {graph['num_relations']}")

    return graph, mappings


def compute_node_degrees(graph: dict, mappings: dict) -> pd.DataFrame:
    """Compute degree statistics for each node."""
    print("\nComputing node degrees...")

    edge_index = graph['edge_index']
    edge_type = graph['edge_type']
    idx2node = mappings['idx2node']
    idx2rel = mappings['idx2relation']

    # Count degrees by relation type
    node_degrees = defaultdict(lambda: defaultdict(int))

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        rel = idx2rel[edge_type[i].item()]

        node_degrees[src][rel] += 1
        node_degrees[dst][rel] += 1
        node_degrees[src]['total'] += 1
        node_degrees[dst]['total'] += 1

    # Build dataframe
    rows = []
    for idx, degrees in node_degrees.items():
        node_id, node_name, node_type = idx2node[idx]
        rows.append({
            'node_idx': idx,
            'node_id': node_id,
            'node_name': node_name,
            'node_type': node_type,
            'total_degree': degrees['total'],
            'drug_gene_degree': degrees.get('drug-gene', 0),
            'gene_disease_degree': degrees.get('gene-disease', 0),
            'gene_gene_degree': degrees.get('gene-gene', 0)
        })

    df = pd.DataFrame(rows)
    return df


def find_hub_genes(degree_df: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    """Find hub genes (high connectivity to drugs and diseases)."""
    print(f"\nFinding top {top_k} hub genes...")

    genes = degree_df[degree_df['node_type'] == 'gene/protein'].copy()

    # Score: weighted combination of drug and disease connections
    genes['hub_score'] = (
        genes['drug_gene_degree'] * 2 +  # Weight drug connections higher
        genes['gene_disease_degree'] * 1.5 +
        genes['gene_gene_degree'] * 0.5
    )

    genes = genes.sort_values('hub_score', ascending=False)
    hub_genes = genes.head(top_k)

    print("\nTop 20 Hub Genes:")
    print("-" * 80)
    for _, row in hub_genes.head(20).iterrows():
        print(f"  {row['node_name']}: "
              f"drugs={row['drug_gene_degree']}, "
              f"diseases={row['gene_disease_degree']}, "
              f"genes={row['gene_gene_degree']}")

    return hub_genes


def analyze_drug_gene_disease_paths(graph: dict, mappings: dict) -> pd.DataFrame:
    """Analyze drug → gene → disease paths."""
    print("\nAnalyzing drug-gene-disease paths...")

    edge_index = graph['edge_index']
    edge_type = graph['edge_type']
    idx2node = mappings['idx2node']
    idx2rel = mappings['idx2relation']

    # Build adjacency lists
    drug_to_genes = defaultdict(set)
    gene_to_diseases = defaultdict(set)

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        rel = idx2rel[edge_type[i].item()]

        src_info = idx2node[src]
        dst_info = idx2node[dst]

        if rel == 'drug-gene':
            if src_info[2] == 'drug':
                drug_to_genes[src].add(dst)
            else:
                drug_to_genes[dst].add(src)
        elif rel == 'gene-disease':
            if src_info[2] == 'gene/protein':
                gene_to_diseases[src].add(dst)
            else:
                gene_to_diseases[dst].add(src)

    # Count paths through each gene
    gene_path_counts = Counter()
    drug_disease_via_gene = defaultdict(lambda: defaultdict(set))

    for drug_idx, genes in drug_to_genes.items():
        for gene_idx in genes:
            if gene_idx in gene_to_diseases:
                diseases = gene_to_diseases[gene_idx]
                gene_path_counts[gene_idx] += len(diseases)

                for disease_idx in diseases:
                    drug_disease_via_gene[drug_idx][disease_idx].add(gene_idx)

    # Build path statistics
    path_stats = []
    for gene_idx, count in gene_path_counts.most_common(50):
        gene_info = idx2node[gene_idx]
        n_drugs = sum(1 for genes in drug_to_genes.values() if gene_idx in genes)
        n_diseases = len(gene_to_diseases.get(gene_idx, set()))

        path_stats.append({
            'gene_idx': gene_idx,
            'gene_id': gene_info[0],
            'gene_name': gene_info[1],
            'num_paths': count,
            'connected_drugs': n_drugs,
            'connected_diseases': n_diseases
        })

    df = pd.DataFrame(path_stats)

    print("\nTop 20 Genes in Drug-Disease Paths:")
    print("-" * 80)
    for _, row in df.head(20).iterrows():
        print(f"  {row['gene_name']}: "
              f"{row['num_paths']} paths "
              f"(drugs={row['connected_drugs']}, diseases={row['connected_diseases']})")

    return df


def analyze_gene_communities(graph: dict, mappings: dict) -> dict:
    """Analyze gene-gene network communities."""
    print("\nAnalyzing gene-gene network...")

    edge_index = graph['edge_index']
    edge_type = graph['edge_type']
    idx2node = mappings['idx2node']
    idx2rel = mappings['idx2relation']

    # Build gene-gene adjacency
    gene_neighbors = defaultdict(set)

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        rel = idx2rel[edge_type[i].item()]

        if rel == 'gene-gene':
            gene_neighbors[src].add(dst)
            gene_neighbors[dst].add(src)

    # Statistics
    gene_degrees = {g: len(neighbors) for g, neighbors in gene_neighbors.items()}
    degrees = list(gene_degrees.values())

    print(f"\nGene-Gene Network Statistics:")
    print(f"  Genes in network: {len(gene_neighbors)}")
    print(f"  Mean degree: {np.mean(degrees):.1f}")
    print(f"  Max degree: {max(degrees)}")
    print(f"  Median degree: {np.median(degrees):.1f}")

    # Top connected genes in PPI network
    top_ppi = sorted(gene_degrees.items(), key=lambda x: x[1], reverse=True)[:20]

    print("\nTop 20 Genes by PPI Connections:")
    print("-" * 60)
    for gene_idx, degree in top_ppi:
        gene_info = idx2node[gene_idx]
        print(f"  {gene_info[1]}: {degree} connections")

    return {'gene_neighbors': gene_neighbors, 'gene_degrees': gene_degrees}


def main():
    """Run gene-centric analysis."""

    print("="*60)
    print("Gene-Centric Analysis")
    print("="*60)

    # Load data
    graph, mappings = load_graph_data()

    # Compute degrees
    degree_df = compute_node_degrees(graph, mappings)

    # Find hub genes
    hub_genes = find_hub_genes(degree_df, top_k=50)

    # Analyze paths
    path_stats = analyze_drug_gene_disease_paths(graph, mappings)

    # Analyze gene communities
    community_stats = analyze_gene_communities(graph, mappings)

    # Save results
    output_path = Path("output/analysis")
    output_path.mkdir(parents=True, exist_ok=True)

    hub_genes.to_csv(output_path / "hub_genes.csv", index=False)
    path_stats.to_csv(output_path / "gene_path_stats.csv", index=False)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    genes = degree_df[degree_df['node_type'] == 'gene/protein']
    drugs = degree_df[degree_df['node_type'] == 'drug']
    diseases = degree_df[degree_df['node_type'] == 'disease']

    print(f"\nNode Statistics:")
    print(f"  Genes: {len(genes)}")
    print(f"  Drugs: {len(drugs)}")
    print(f"  Diseases: {len(diseases)}")

    print(f"\nGene Degree Statistics:")
    print(f"  Mean drug connections: {genes['drug_gene_degree'].mean():.2f}")
    print(f"  Mean disease connections: {genes['gene_disease_degree'].mean():.2f}")
    print(f"  Mean gene connections: {genes['gene_gene_degree'].mean():.2f}")

    print(f"\nTop Hub Genes by Drug Connections:")
    top_drug_genes = genes.nlargest(5, 'drug_gene_degree')
    for _, row in top_drug_genes.iterrows():
        print(f"  {row['node_name']}: {row['drug_gene_degree']} drugs")

    print(f"\nTop Hub Genes by Disease Connections:")
    top_disease_genes = genes.nlargest(5, 'gene_disease_degree')
    for _, row in top_disease_genes.iterrows():
        print(f"  {row['node_name']}: {row['gene_disease_degree']} diseases")

    print(f"\nResults saved to {output_path}")

    return hub_genes, path_stats


if __name__ == "__main__":
    main()
