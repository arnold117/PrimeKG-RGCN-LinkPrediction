"""
MLP Encoder for Link Prediction (Baseline)

This is a baseline model that does NOT use graph structure.
It only learns node embeddings and passes them through MLP layers.

This serves as a lower bound - if RGCN can't beat MLP, the graph
structure is not being utilized effectively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLPEncoder(nn.Module):
    """
    MLP Encoder - No graph convolution, just learned embeddings + MLP.

    Architecture:
        Input: Learnable node embeddings [num_nodes, embedding_dim]
        FC1: embedding_dim -> hidden_dim
        LayerNorm + ReLU + Dropout
        FC2: hidden_dim -> hidden_dim (with skip connection)

    This baseline ignores:
    - Graph structure (edge_index)
    - Relation types (edge_type)

    If RGCN performs similar to MLP, it means the graph structure
    is not being utilized effectively.
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,  # Ignored, kept for interface consistency
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        **kwargs  # Ignore extra args like num_bases, use_layer_norm, etc.
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Learnable node embeddings (same as RGCN)
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

        # MLP layers (replace RGCN convolutions)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

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
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        edge_index: torch.Tensor,  # IGNORED
        edge_type: torch.Tensor,   # IGNORED
        node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass - ignores graph structure.

        Args:
            edge_index: Graph edges [2, num_edges] - IGNORED
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

        # MLP Layer 1
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Save for skip connection
        x_skip = x

        # MLP Layer 2 with skip connection
        x = self.fc2(x)
        x = self.norm2(x)
        x = x + x_skip  # Skip connection

        return x


# ============================================================================
# Test
# ============================================================================

def test_mlp_encoder():
    """Test MLPEncoder."""
    print("Testing MLPEncoder...")

    num_nodes = 100
    num_relations = 3
    embedding_dim = 64
    hidden_dim = 128
    num_edges = 500

    # Create encoder
    encoder = MLPEncoder(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=0.5
    )

    # Create dummy graph data (will be ignored)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))

    # Forward pass
    embeddings = encoder(edge_index, edge_type)

    # Assertions
    assert embeddings.shape == (num_nodes, hidden_dim), \
        f"Expected ({num_nodes}, {hidden_dim}), got {embeddings.shape}"

    # Test that graph structure is truly ignored
    edge_index_2 = torch.randint(0, num_nodes, (2, 100))  # Different graph
    embeddings_2 = encoder(edge_index_2, edge_type[:100])

    # Embeddings should be identical (graph is ignored)
    assert torch.allclose(embeddings, embeddings_2), \
        "MLP should produce same output regardless of graph structure"

    print(f"  Output shape: {embeddings.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print("  MLP encoder test passed!")


if __name__ == '__main__':
    test_mlp_encoder()
