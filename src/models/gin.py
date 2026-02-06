"""
GIN Encoder for Link Prediction

Graph Isomorphism Network - theoretically the most expressive GNN
under the Weisfeiler-Lehman test framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from typing import Optional


class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network Encoder.

    GIN is provably as powerful as the WL graph isomorphism test.
    Uses MLP for neighbor aggregation instead of simple mean/sum.

    Architecture:
        Input: Learnable node embeddings [num_nodes, embedding_dim]
        GINConv1: embedding_dim -> hidden_dim (with MLP)
        LayerNorm + ReLU + Dropout
        GINConv2: hidden_dim -> hidden_dim (with skip connection)
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,  # Ignored
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Learnable node embeddings
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

        # GIN uses MLP for aggregation
        mlp1 = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # GIN layers
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.node_embeddings.weight)

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,  # IGNORED
        node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with expressive aggregation."""
        if node_indices is None:
            x = self.node_embeddings.weight
        else:
            x = self.node_embeddings(node_indices)

        # GIN Layer 1
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Skip connection
        x_skip = x

        # GIN Layer 2
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x + x_skip

        return x


def test_gin_encoder():
    """Test GINEncoder."""
    print("Testing GINEncoder...")

    num_nodes = 100
    num_relations = 3
    embedding_dim = 64
    hidden_dim = 128
    num_edges = 500

    encoder = GINEncoder(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=0.5
    )

    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))

    embeddings = encoder(edge_index, edge_type)

    assert embeddings.shape == (num_nodes, hidden_dim)
    print(f"  Output shape: {embeddings.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print("  GIN encoder test passed!")


if __name__ == '__main__':
    test_gin_encoder()
