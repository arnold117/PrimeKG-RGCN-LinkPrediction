"""
GCN Encoder for Link Prediction

Standard Graph Convolutional Network that ignores relation types.
Uses the same graph structure as RGCN but treats all edges equally.

This tests whether relation-specific weights (RGCN) are necessary,
or if simple message passing (GCN) is sufficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Optional


class GCNEncoder(nn.Module):
    """
    Standard GCN Encoder - ignores edge types.

    Architecture:
        Input: Learnable node embeddings [num_nodes, embedding_dim]
        GCNConv1: embedding_dim -> hidden_dim
        LayerNorm + ReLU + Dropout
        GCNConv2: hidden_dim -> hidden_dim (with skip connection)

    Key difference from RGCN:
    - Uses same weights for ALL edge types
    - Cannot distinguish drug-gene from gene-disease edges
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,  # Ignored, kept for interface consistency
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        **kwargs  # Ignore extra args
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Learnable node embeddings (same as RGCN)
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

        # GCN layers (no relation-specific weights)
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.node_embeddings.weight)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,  # IGNORED
        node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass - uses graph structure but ignores edge types.

        Args:
            edge_index: Graph edges [2, num_edges]
            edge_type: Edge types [num_edges] - IGNORED
            node_indices: Optional specific nodes to embed

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Get initial embeddings
        if node_indices is None:
            x = self.node_embeddings.weight
        else:
            x = self.node_embeddings(node_indices)

        # GCN Layer 1
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Save for skip connection
        x_skip = x

        # GCN Layer 2 with skip connection
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x + x_skip  # Skip connection

        return x


# ============================================================================
# Test
# ============================================================================

def test_gcn_encoder():
    """Test GCNEncoder."""
    print("Testing GCNEncoder...")

    num_nodes = 100
    num_relations = 3
    embedding_dim = 64
    hidden_dim = 128
    num_edges = 500

    # Create encoder
    encoder = GCNEncoder(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=0.5
    )

    # Create dummy graph data
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))

    # Forward pass
    embeddings = encoder(edge_index, edge_type)

    # Assertions
    assert embeddings.shape == (num_nodes, hidden_dim), \
        f"Expected ({num_nodes}, {hidden_dim}), got {embeddings.shape}"

    # Test that edge_type is truly ignored
    edge_type_2 = torch.randint(0, num_relations, (num_edges,))  # Different types
    embeddings_2 = encoder(edge_index, edge_type_2)

    # Same graph structure -> same embeddings (edge types ignored)
    assert torch.allclose(embeddings, embeddings_2, atol=1e-6), \
        "GCN should produce same output regardless of edge types"

    print(f"  Output shape: {embeddings.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print("  GCN encoder test passed!")


if __name__ == '__main__':
    test_gcn_encoder()
