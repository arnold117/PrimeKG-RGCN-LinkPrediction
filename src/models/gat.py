"""
GAT Encoder for Link Prediction

Graph Attention Network that uses attention mechanism to weight neighbor contributions.
Ignores relation types, but learns which neighbors are more important.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional


class GATEncoder(nn.Module):
    """
    Graph Attention Network Encoder.

    Uses multi-head attention to learn neighbor importance.
    Ignores edge types but can learn more nuanced aggregation.

    Architecture:
        Input: Learnable node embeddings [num_nodes, embedding_dim]
        GATConv1: embedding_dim -> hidden_dim (multi-head)
        LayerNorm + ELU + Dropout
        GATConv2: hidden_dim -> hidden_dim (with skip connection)
    """

    def __init__(
        self,
        num_nodes: int,
        num_relations: int,  # Ignored
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        heads: int = 4,
        **kwargs
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.heads = heads

        # Learnable node embeddings
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)

        # GAT layers with multi-head attention
        # First layer: embedding_dim -> hidden_dim (using heads)
        self.conv1 = GATConv(
            embedding_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True  # Concatenate head outputs
        )

        # Second layer: hidden_dim -> hidden_dim
        self.conv2 = GATConv(
            hidden_dim,
            hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )

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
        """Forward pass with attention mechanism."""
        if node_indices is None:
            x = self.node_embeddings.weight
        else:
            x = self.node_embeddings(node_indices)

        # GAT Layer 1
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)

        # Skip connection
        x_skip = x

        # GAT Layer 2
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x + x_skip

        return x


def test_gat_encoder():
    """Test GATEncoder."""
    print("Testing GATEncoder...")

    num_nodes = 100
    num_relations = 3
    embedding_dim = 64
    hidden_dim = 128
    num_edges = 500

    encoder = GATEncoder(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=0.5,
        heads=4
    )

    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))

    embeddings = encoder(edge_index, edge_type)

    assert embeddings.shape == (num_nodes, hidden_dim)
    print(f"  Output shape: {embeddings.shape}")
    print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print("  GAT encoder test passed!")


if __name__ == '__main__':
    test_gat_encoder()
