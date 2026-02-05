"""
Export Drug Embeddings from Trained RGCN Model

This script loads the trained RGCN model and exports drug node embeddings
for use in downstream gene expression prediction tasks.

Output:
    - data/processed/drug_embeddings.pt: Dict with drug_id -> embedding mapping
    - data/processed/drug_embeddings.csv: CSV for inspection
"""

import sys
import torch
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.rgcn import DrugDiseaseModel


def load_model_and_data(
    model_path: str = "output/models/best_model.pt",
    graph_path: str = "data/processed/full_graph.pt",
    mappings_path: str = "data/processed/mappings.pt"
):
    """Load trained model, graph structure, and node mappings."""

    # Load graph data
    print(f"Loading graph from {graph_path}...")
    graph_data = torch.load(graph_path, weights_only=False)

    # Load mappings
    print(f"Loading mappings from {mappings_path}...")
    mappings = torch.load(mappings_path, weights_only=False)

    # Load checkpoint first to detect architecture
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Detect if model used layer norm by checking state dict keys
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    has_layer_norm = any('norm1' in k for k in state_dict.keys())

    print(f"  Detected layer norm: {has_layer_norm}")

    # Create model matching the trained architecture
    # We need to create encoder with correct flags
    from src.models.rgcn import DrugDiseaseRGCN, LinkPredictor

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

    # Create a simple container model
    class SimpleModel(torch.nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.num_nodes = encoder.num_nodes
            self.num_relations = encoder.num_relations
            self.hidden_dim = encoder.hidden_dim

        def get_embeddings(self, edge_index, edge_type):
            self.eval()
            with torch.no_grad():
                return self.encoder(edge_index, edge_type)

    model = SimpleModel(encoder, decoder)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model, graph_data, mappings


def extract_drug_embeddings(model, graph_data, mappings):
    """Extract embeddings for all drug nodes."""

    # Get all node embeddings through RGCN
    print("Running RGCN forward pass to get node embeddings...")
    with torch.no_grad():
        all_embeddings = model.get_embeddings(
            graph_data['edge_index'],
            graph_data['edge_type']
        )

    print(f"All embeddings shape: {all_embeddings.shape}")

    # Find drug node indices
    idx2node = mappings['idx2node']

    drug_embeddings = {}
    drug_info = []

    for idx, (node_id, node_name, node_type) in idx2node.items():
        if node_type == 'drug':
            embedding = all_embeddings[idx].numpy()
            drug_embeddings[node_id] = embedding
            drug_info.append({
                'drug_id': node_id,
                'drug_name': node_name,
                'node_idx': idx
            })

    print(f"Extracted embeddings for {len(drug_embeddings)} drugs")

    return drug_embeddings, drug_info, all_embeddings


def save_embeddings(drug_embeddings, drug_info, all_embeddings, output_dir="data/processed"):
    """Save embeddings in multiple formats."""

    output_path = Path(output_dir)

    # Save as PyTorch dict
    pt_path = output_path / "drug_embeddings.pt"
    torch.save({
        'drug_embeddings': drug_embeddings,  # {drug_id: np.array}
        'embedding_dim': 128,
        'num_drugs': len(drug_embeddings)
    }, pt_path)
    print(f"Saved PyTorch embeddings to {pt_path}")

    # Save drug info as CSV for inspection
    df = pd.DataFrame(drug_info)
    csv_path = output_path / "drug_info.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved drug info to {csv_path}")

    # Also save as a matrix for easy loading
    drug_ids = list(drug_embeddings.keys())
    embedding_matrix = torch.tensor([drug_embeddings[did] for did in drug_ids])

    matrix_path = output_path / "drug_embedding_matrix.pt"
    torch.save({
        'drug_ids': drug_ids,
        'embeddings': embedding_matrix  # [num_drugs, 128]
    }, matrix_path)
    print(f"Saved embedding matrix to {matrix_path}")

    return pt_path, csv_path, matrix_path


def analyze_embeddings(drug_embeddings):
    """Basic analysis of drug embeddings."""

    import numpy as np

    embeddings = np.array(list(drug_embeddings.values()))

    print("\n" + "="*50)
    print("Drug Embedding Statistics")
    print("="*50)
    print(f"Number of drugs: {len(drug_embeddings)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Mean: {embeddings.mean():.4f}")
    print(f"Std: {embeddings.std():.4f}")
    print(f"Min: {embeddings.min():.4f}")
    print(f"Max: {embeddings.max():.4f}")

    # Check for any NaN or Inf values
    if np.isnan(embeddings).any():
        print("WARNING: NaN values detected!")
    if np.isinf(embeddings).any():
        print("WARNING: Inf values detected!")

    # L2 norm statistics
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"L2 norm - Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")
    print("="*50 + "\n")


def main():
    """Main function to export drug embeddings."""

    print("="*60)
    print("Exporting Drug Embeddings from Trained RGCN Model")
    print("="*60 + "\n")

    # Load model and data
    model, graph_data, mappings = load_model_and_data()

    # Extract drug embeddings
    drug_embeddings, drug_info, all_embeddings = extract_drug_embeddings(
        model, graph_data, mappings
    )

    # Analyze embeddings
    analyze_embeddings(drug_embeddings)

    # Save embeddings
    save_embeddings(drug_embeddings, drug_info, all_embeddings)

    print("\nDrug embedding export complete!")
    print(f"Total drugs: {len(drug_embeddings)}")
    print(f"Embedding dimension: 128")

    # Show sample drugs
    print("\nSample drugs:")
    for i, info in enumerate(drug_info[:5]):
        print(f"  {info['drug_id']}: {info['drug_name']}")


if __name__ == "__main__":
    main()
