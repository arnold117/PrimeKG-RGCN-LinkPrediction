"""
Drug Similarity Analysis using RGCN Embeddings

Analyzes drug embeddings to:
1. Find similar drug pairs based on cosine similarity
2. Validate similarities against shared targets/indications
3. Identify drug repurposing candidates

Usage:
    python src/drug_similarity_analysis.py
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_drug_embeddings():
    """Load drug embeddings and info."""
    print("Loading drug embeddings...")

    emb_data = torch.load('data/processed/drug_embedding_matrix.pt', weights_only=False)
    drug_info = pd.read_csv('data/processed/drug_info.csv')

    drug_ids = emb_data['drug_ids']
    embeddings = emb_data['embeddings'].numpy()

    print(f"Loaded {len(drug_ids)} drug embeddings, dim={embeddings.shape[1]}")

    return drug_ids, embeddings, drug_info


def load_drug_gene_relations():
    """Load drug-gene relationships from raw data."""
    print("\nLoading drug-gene relations...")

    df = pd.read_csv('data/raw/kg.csv', low_memory=False)
    drug_gene = df[df['relation'] == 'drug_protein'].copy()

    # Build drug -> targets mapping
    drug_targets = defaultdict(set)
    for _, row in drug_gene.iterrows():
        drug_id = row['x_id']
        gene_id = str(row['y_id'])
        gene_name = row['y_name']
        drug_targets[drug_id].add((gene_id, gene_name))

    print(f"Built target mappings for {len(drug_targets)} drugs")
    return drug_targets


def load_drug_disease_relations():
    """Load drug-disease relationships (via gene) from mappings."""
    print("\nLoading drug-disease relations...")

    mappings = torch.load('data/processed/mappings.pt', weights_only=False)
    graph = torch.load('data/processed/full_graph.pt', weights_only=False)

    idx2node = mappings['idx2node']
    idx2rel = mappings['idx2relation']
    edge_index = graph['edge_index']
    edge_type = graph['edge_type']

    # Find drug-gene and gene-disease edges
    drug_genes = defaultdict(set)
    gene_diseases = defaultdict(set)

    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i].item()
        dst_idx = edge_index[1, i].item()
        rel_idx = edge_type[i].item()
        rel_name = idx2rel[rel_idx]

        src_info = idx2node[src_idx]
        dst_info = idx2node[dst_idx]

        if rel_name == 'drug-gene':
            if src_info[2] == 'drug':
                drug_genes[src_info[0]].add(dst_info[0])
            else:
                drug_genes[dst_info[0]].add(src_info[0])
        elif rel_name == 'gene-disease':
            if src_info[2] == 'gene/protein':
                gene_diseases[src_info[0]].add(dst_info[0])
            else:
                gene_diseases[dst_info[0]].add(src_info[0])

    # Compute drug -> diseases via shared genes
    drug_diseases = defaultdict(set)
    for drug_id, genes in drug_genes.items():
        for gene in genes:
            if gene in gene_diseases:
                drug_diseases[drug_id].update(gene_diseases[gene])

    print(f"Found indirect disease links for {len(drug_diseases)} drugs")
    return drug_diseases, drug_genes


def compute_similarity_matrix(embeddings: np.ndarray, top_k: int = 10):
    """Compute pairwise cosine similarity."""
    print("\nComputing similarity matrix...")

    sim_matrix = cosine_similarity(embeddings)

    # Zero out diagonal
    np.fill_diagonal(sim_matrix, 0)

    print(f"Similarity matrix shape: {sim_matrix.shape}")
    print(f"Similarity range: [{sim_matrix.min():.4f}, {sim_matrix.max():.4f}]")
    print(f"Mean similarity: {sim_matrix.mean():.4f}")

    return sim_matrix


def find_top_similar_pairs(
    drug_ids: list,
    drug_info: pd.DataFrame,
    sim_matrix: np.ndarray,
    drug_targets: dict,
    top_k: int = 50
) -> pd.DataFrame:
    """Find top similar drug pairs and validate with shared targets."""
    print(f"\nFinding top {top_k} similar drug pairs...")

    # Get top pairs
    n = len(drug_ids)
    pairs = []

    # Flatten upper triangle
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, sim_matrix[i, j]))

    # Sort by similarity
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[:top_k]

    # Build results
    results = []
    id_to_name = dict(zip(drug_info['drug_id'], drug_info['drug_name']))

    for i, j, sim in top_pairs:
        drug1_id = drug_ids[i]
        drug2_id = drug_ids[j]

        drug1_name = id_to_name.get(drug1_id, drug1_id)
        drug2_name = id_to_name.get(drug2_id, drug2_id)

        # Check shared targets
        targets1 = drug_targets.get(drug1_id, set())
        targets2 = drug_targets.get(drug2_id, set())

        shared_targets = targets1 & targets2
        shared_gene_names = [t[1] for t in shared_targets]

        results.append({
            'drug1_id': drug1_id,
            'drug1_name': drug1_name,
            'drug2_id': drug2_id,
            'drug2_name': drug2_name,
            'similarity': sim,
            'num_shared_targets': len(shared_targets),
            'shared_targets': '; '.join(shared_gene_names[:5]),
            'drug1_num_targets': len(targets1),
            'drug2_num_targets': len(targets2)
        })

    df = pd.DataFrame(results)
    return df


def find_repurposing_candidates(
    drug_ids: list,
    drug_info: pd.DataFrame,
    sim_matrix: np.ndarray,
    drug_diseases: dict,
    min_similarity: float = 0.8
) -> pd.DataFrame:
    """Find drug repurposing candidates based on embedding similarity."""
    print(f"\nFinding repurposing candidates (similarity >= {min_similarity})...")

    id_to_name = dict(zip(drug_info['drug_id'], drug_info['drug_name']))
    candidates = []

    n = len(drug_ids)
    for i in range(n):
        drug1_id = drug_ids[i]
        drug1_diseases = drug_diseases.get(drug1_id, set())

        if len(drug1_diseases) == 0:
            continue

        # Find similar drugs
        similar_indices = np.where(sim_matrix[i] >= min_similarity)[0]

        for j in similar_indices:
            if i == j:
                continue

            drug2_id = drug_ids[j]
            drug2_diseases = drug_diseases.get(drug2_id, set())

            # Find diseases unique to drug1 that drug2 might treat
            new_diseases = drug1_diseases - drug2_diseases

            if len(new_diseases) > 0:
                candidates.append({
                    'source_drug_id': drug1_id,
                    'source_drug_name': id_to_name.get(drug1_id, drug1_id),
                    'candidate_drug_id': drug2_id,
                    'candidate_drug_name': id_to_name.get(drug2_id, drug2_id),
                    'similarity': sim_matrix[i, j],
                    'num_new_diseases': len(new_diseases),
                    'source_num_diseases': len(drug1_diseases),
                    'candidate_num_diseases': len(drug2_diseases)
                })

    df = pd.DataFrame(candidates)
    if len(df) > 0:
        df = df.sort_values('similarity', ascending=False)
    print(f"Found {len(df)} repurposing candidates")
    return df


def analyze_clustering(
    drug_ids: list,
    embeddings: np.ndarray,
    drug_info: pd.DataFrame,
    n_clusters: int = 20
):
    """Cluster drugs by embedding and analyze cluster composition."""
    print(f"\nClustering drugs into {n_clusters} clusters...")

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    # Analyze cluster sizes
    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
    print("\nCluster sizes:")
    print(cluster_sizes)

    # Create drug-cluster mapping
    id_to_name = dict(zip(drug_info['drug_id'], drug_info['drug_name']))

    cluster_drugs = defaultdict(list)
    for i, cluster in enumerate(clusters):
        drug_id = drug_ids[i]
        drug_name = id_to_name.get(drug_id, drug_id)
        cluster_drugs[cluster].append((drug_id, drug_name))

    return clusters, cluster_drugs


def main():
    """Run drug similarity analysis."""

    print("="*60)
    print("Drug Similarity Analysis")
    print("="*60)

    # Load data
    drug_ids, embeddings, drug_info = load_drug_embeddings()
    drug_targets = load_drug_gene_relations()
    drug_diseases, _ = load_drug_disease_relations()

    # Compute similarity
    sim_matrix = compute_similarity_matrix(embeddings)

    # Find similar pairs
    similar_pairs = find_top_similar_pairs(
        drug_ids, drug_info, sim_matrix, drug_targets, top_k=50
    )

    # Validate: do similar drugs share targets?
    print("\n" + "="*60)
    print("Validation: Shared Targets Analysis")
    print("="*60)

    pairs_with_shared = similar_pairs[similar_pairs['num_shared_targets'] > 0]
    print(f"Top-50 similar pairs with shared targets: {len(pairs_with_shared)}/50 ({len(pairs_with_shared)/50*100:.0f}%)")

    print("\nTop 10 similar pairs:")
    for _, row in similar_pairs.head(10).iterrows():
        print(f"  {row['drug1_name']} <-> {row['drug2_name']}")
        print(f"    Similarity: {row['similarity']:.4f}, Shared targets: {row['num_shared_targets']}")
        if row['shared_targets']:
            print(f"    Targets: {row['shared_targets']}")

    # Find repurposing candidates
    repurposing = find_repurposing_candidates(
        drug_ids, drug_info, sim_matrix, drug_diseases, min_similarity=0.85
    )

    # Clustering
    clusters, cluster_drugs = analyze_clustering(
        drug_ids, embeddings, drug_info, n_clusters=20
    )

    # Save results
    output_path = Path("output/analysis")
    output_path.mkdir(parents=True, exist_ok=True)

    similar_pairs.to_csv(output_path / "similar_drug_pairs.csv", index=False)
    if len(repurposing) > 0:
        repurposing.head(100).to_csv(output_path / "repurposing_candidates.csv", index=False)

    # Save cluster info
    cluster_df = pd.DataFrame({
        'drug_id': drug_ids,
        'cluster': clusters
    })
    cluster_df.to_csv(output_path / "drug_clusters.csv", index=False)

    print(f"\nResults saved to {output_path}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total drugs analyzed: {len(drug_ids)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Top-50 similar pairs with shared targets: {len(pairs_with_shared)}/50")
    if len(repurposing) > 0:
        print(f"Repurposing candidates (sim >= 0.85): {len(repurposing)}")

    return similar_pairs, repurposing


if __name__ == "__main__":
    main()
