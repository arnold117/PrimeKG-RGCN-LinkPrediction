"""
GraphSAGE Encoder for Link Prediction

GraphSAGE (SAmple and aggreGatE) learns node embeddings by sampling
and aggregating features from a node's local neighborhood.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import Optional


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE Encoder.

    Uses mean aggregation over neighbors (default).
    More scalable than GCN for large graphs due to sampling.

    Architecture:
        Input: Learnable node embeddings [num_nodes, embedding_dim]
        SAGEConv1: embedding_dim -> hidden_dim
        LayerNorm + ReLU + Dropout
        SAGEConv2: hidden_dim -> hidden_dim (with skip connection)
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

        # GraphSAGE layers
        self.conv1 = SAGEConv(embedding_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

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
        """Forward pass with neighborhood aggregation."""
        if node_indices is None:
            x = self.node_embeddings.weight
        else:
            x = self.node_embeddings(node_indices)

        # SAGE Layer 1
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Skip connection
        x_skip = x

        # SAGE Layer 2
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = x + x_skip

        return x


def test_graphsage_encoder():
    """Test GraphSAGEEncoder."""
    print("Testing GraphSAGEEncoder...")

    num_nodes = 100
    num_relations = 3
    embedding_dim = 64
    hidden_dim = 128
    num_edges = 500

    encoder = GraphSAGEEncoder(
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
    print("  GraphSAGE encoder test passed!")


if __name__ == '__main__':
    test_graphsage_encoder()
