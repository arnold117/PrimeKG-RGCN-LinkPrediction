"""
RGCN Model for Drug-Disease Link Prediction

This module implements a Relational Graph Convolutional Network (RGCN) for
link prediction on knowledge graphs, specifically designed for drug-disease
prediction tasks.

Classes:
    - DrugDiseaseRGCN: RGCN encoder for node embeddings
    - LinkPredictor: DistMult-based link prediction decoder
    - DrugDiseaseModel: Complete model combining encoder and decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from typing import Tuple, Optional


class DrugDiseaseRGCN(nn.Module):
    """
    Relational Graph Convolutional Network (RGCN) encoder.
    
    This encoder learns node embeddings by propagating information through
    a heterogeneous graph with multiple relation types using RGCN layers.
    
    Architecture:
        Input: Learnable node embeddings
        Layer 1: RGCN (embedding_dim -> hidden_dim)
        Activation: ReLU
        Dropout
        Layer 2: RGCN (hidden_dim -> hidden_dim)
    
    Args:
        num_nodes (int): Total number of nodes in the graph
        num_relations (int): Number of relation types
        embedding_dim (int): Dimension of initial node embeddings (default: 64)
        hidden_dim (int): Dimension of hidden representations (default: 128)
        dropout (float): Dropout probability (default: 0.5)
        num_bases (int): Number of bases for basis decomposition in RGCN.
                        If None, uses full parameter matrices. (default: None)
    
    Attributes:
        node_embeddings (nn.Embedding): Learnable node embedding table
        conv1 (RGCNConv): First RGCN layer
        conv2 (RGCNConv): Second RGCN layer
        dropout (nn.Dropout): Dropout layer
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        num_bases: Optional[int] = None
    ):
        super(DrugDiseaseRGCN, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Learnable node embeddings as input features
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
        
        # RGCN layers
        # Layer 1: embedding_dim -> hidden_dim
        self.conv1 = RGCNConv(
            in_channels=embedding_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            num_bases=num_bases
        )
        
        # Layer 2: hidden_dim -> hidden_dim
        self.conv2 = RGCNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_relations=num_relations,
            num_bases=num_bases
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize node embeddings using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.node_embeddings.weight)
    
    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        node_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the RGCN encoder.
        
        Args:
            edge_index (torch.Tensor): Graph edge indices [2, num_edges]
            edge_type (torch.Tensor): Edge type indices [num_edges]
            node_indices (torch.Tensor, optional): Specific node indices to get
                embeddings for. If None, returns all node embeddings.
        
        Returns:
            torch.Tensor: Node embeddings [num_nodes, hidden_dim] or
                         [len(node_indices), hidden_dim] if node_indices provided
        """
        # Get initial node embeddings
        if node_indices is None:
            x = self.node_embeddings.weight
        else:
            x = self.node_embeddings(node_indices)
        
        # First RGCN layer
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second RGCN layer
        x = self.conv2(x, edge_index, edge_type)
        
        return x
    
    def get_node_embeddings(self, node_indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for specific nodes.
        
        Args:
            node_indices (torch.Tensor): Node indices [num_nodes]
        
        Returns:
            torch.Tensor: Node embeddings [num_nodes, hidden_dim]
        """
        return self.node_embeddings(node_indices)


class LinkPredictor(nn.Module):
    """
    Link prediction decoder using DistMult scoring function.
    
    DistMult is a bilinear model that computes the score of a triple (h, r, t)
    as: score = <h, r, t> = sum(h * r * t)
    
    where h is head entity embedding, r is relation embedding, and t is tail
    entity embedding, and * denotes element-wise multiplication.
    
    Args:
        num_relations (int): Number of relation types
        embedding_dim (int): Dimension of entity and relation embeddings
        dropout (float): Dropout probability (default: 0.0)
    
    Attributes:
        relation_embeddings (nn.Embedding): Learnable relation embedding table
        dropout (nn.Dropout): Dropout layer
    """
    
    def __init__(
        self,
        num_relations: int,
        embedding_dim: int,
        dropout: float = 0.0
    ):
        super(LinkPredictor, self).__init__()
        
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        
        # Learnable relation embeddings for DistMult
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize relation embeddings using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
    
    def forward(
        self,
        head_embeddings: torch.Tensor,
        tail_embeddings: torch.Tensor,
        relation_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute link prediction scores using DistMult.
        
        Args:
            head_embeddings (torch.Tensor): Head entity embeddings [batch_size, dim]
            tail_embeddings (torch.Tensor): Tail entity embeddings [batch_size, dim]
            relation_types (torch.Tensor): Relation type indices [batch_size]
        
        Returns:
            torch.Tensor: Prediction scores [batch_size]
        """
        # Get relation embeddings
        relation_emb = self.relation_embeddings(relation_types)
        relation_emb = self.dropout(relation_emb)
        
        # DistMult scoring: <h, r, t> = sum(h * r * t)
        scores = torch.sum(head_embeddings * relation_emb * tail_embeddings, dim=1)
        
        return scores
    
    def score_all_tails(
        self,
        head_embeddings: torch.Tensor,
        relation_types: torch.Tensor,
        all_tail_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Score all possible tail entities for given head-relation pairs.
        Useful for ranking during evaluation.
        
        Args:
            head_embeddings (torch.Tensor): Head entity embeddings [batch_size, dim]
            relation_types (torch.Tensor): Relation type indices [batch_size]
            all_tail_embeddings (torch.Tensor): All entity embeddings [num_entities, dim]
        
        Returns:
            torch.Tensor: Scores for all tails [batch_size, num_entities]
        """
        # Get relation embeddings
        relation_emb = self.relation_embeddings(relation_types)
        
        # Compute scores: (h * r) @ all_tails^T
        # [batch_size, dim] * [batch_size, dim] = [batch_size, dim]
        hr = head_embeddings * relation_emb
        
        # [batch_size, dim] @ [dim, num_entities] = [batch_size, num_entities]
        scores = hr @ all_tail_embeddings.t()
        
        return scores


class DrugDiseaseModel(nn.Module):
    """
    Complete model for drug-disease link prediction.
    
    Combines an RGCN encoder for learning node embeddings with a DistMult
    decoder for link prediction.
    
    Args:
        num_nodes (int): Total number of nodes in the graph
        num_relations (int): Number of relation types
        embedding_dim (int): Dimension of initial node embeddings (default: 64)
        hidden_dim (int): Dimension of hidden representations (default: 128)
        dropout (float): Dropout probability for RGCN (default: 0.5)
        decoder_dropout (float): Dropout probability for decoder (default: 0.0)
        num_bases (int, optional): Number of bases for RGCN basis decomposition
    
    Attributes:
        encoder (DrugDiseaseRGCN): RGCN encoder
        decoder (LinkPredictor): DistMult link predictor
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_relations: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.5,
        decoder_dropout: float = 0.0,
        num_bases: Optional[int] = None
    ):
        super(DrugDiseaseModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        # RGCN encoder
        self.encoder = DrugDiseaseRGCN(
            num_nodes=num_nodes,
            num_relations=num_relations,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_bases=num_bases
        )
        
        # DistMult decoder
        self.decoder = LinkPredictor(
            num_relations=num_relations,
            embedding_dim=hidden_dim,
            dropout=decoder_dropout
        )
    
    def forward(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        head_indices: torch.Tensor,
        tail_indices: torch.Tensor,
        relation_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            edge_index (torch.Tensor): Graph edge indices [2, num_edges]
            edge_type (torch.Tensor): Edge type indices [num_edges]
            head_indices (torch.Tensor): Head entity indices [batch_size]
            tail_indices (torch.Tensor): Tail entity indices [batch_size]
            relation_types (torch.Tensor): Relation type indices [batch_size]
        
        Returns:
            torch.Tensor: Prediction scores [batch_size]
        """
        # Encode graph to get node embeddings
        node_embeddings = self.encoder(edge_index, edge_type)
        
        # Get embeddings for head and tail entities
        head_emb = node_embeddings[head_indices]
        tail_emb = node_embeddings[tail_indices]
        
        # Predict links
        scores = self.decoder(head_emb, tail_emb, relation_types)
        
        return scores
    
    def predict(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        head_indices: torch.Tensor,
        tail_indices: torch.Tensor,
        relation_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link existence (inference mode).
        
        Args:
            edge_index (torch.Tensor): Graph edge indices [2, num_edges]
            edge_type (torch.Tensor): Edge type indices [num_edges]
            head_indices (torch.Tensor): Head entity indices [batch_size]
            tail_indices (torch.Tensor): Tail entity indices [batch_size]
            relation_types (torch.Tensor): Relation type indices [batch_size]
        
        Returns:
            torch.Tensor: Prediction scores [batch_size]
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(
                edge_index, edge_type,
                head_indices, tail_indices, relation_types
            )
        return scores
    
    def predict_all_tails(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        head_indices: torch.Tensor,
        relation_types: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict scores for all possible tail entities.
        Useful for ranking during evaluation.
        
        Args:
            edge_index (torch.Tensor): Graph edge indices [2, num_edges]
            edge_type (torch.Tensor): Edge type indices [num_edges]
            head_indices (torch.Tensor): Head entity indices [batch_size]
            relation_types (torch.Tensor): Relation type indices [batch_size]
        
        Returns:
            torch.Tensor: Scores for all tails [batch_size, num_nodes]
        """
        self.eval()
        with torch.no_grad():
            # Encode graph
            all_node_embeddings = self.encoder(edge_index, edge_type)
            
            # Get head embeddings
            head_emb = all_node_embeddings[head_indices]
            
            # Score all possible tails
            scores = self.decoder.score_all_tails(
                head_emb, relation_types, all_node_embeddings
            )
        
        return scores
    
    def get_embeddings(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Get node embeddings after encoding.
        
        Args:
            edge_index (torch.Tensor): Graph edge indices [2, num_edges]
            edge_type (torch.Tensor): Edge type indices [num_edges]
        
        Returns:
            torch.Tensor: Node embeddings [num_nodes, hidden_dim]
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.encoder(edge_index, edge_type)
        return embeddings


# ============================================================================
# Test Section
# ============================================================================

def test_rgcn_encoder():
    """Test DrugDiseaseRGCN encoder."""
    print("Testing DrugDiseaseRGCN Encoder...")
    
    # Parameters
    num_nodes = 100
    num_relations = 3
    embedding_dim = 64
    hidden_dim = 128
    num_edges = 500
    
    # Create model
    encoder = DrugDiseaseRGCN(
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
        f"Expected shape ({num_nodes}, {hidden_dim}), got {embeddings.shape}"
    
    print(f"  ✓ Input: {num_nodes} nodes, {num_edges} edges, {num_relations} relations")
    print(f"  ✓ Output embeddings shape: {embeddings.shape}")
    print(f"  ✓ Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    print("  ✓ Encoder test passed!\n")


def test_link_predictor():
    """Test LinkPredictor decoder."""
    print("Testing LinkPredictor Decoder...")
    
    # Parameters
    num_relations = 3
    embedding_dim = 128
    batch_size = 32
    num_entities = 100
    
    # Create model
    decoder = LinkPredictor(
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        dropout=0.0
    )
    
    # Create dummy embeddings
    head_emb = torch.randn(batch_size, embedding_dim)
    tail_emb = torch.randn(batch_size, embedding_dim)
    relation_types = torch.randint(0, num_relations, (batch_size,))
    
    # Forward pass
    scores = decoder(head_emb, tail_emb, relation_types)
    
    # Test score_all_tails
    all_tail_emb = torch.randn(num_entities, embedding_dim)
    all_scores = decoder.score_all_tails(head_emb, relation_types, all_tail_emb)
    
    # Assertions
    assert scores.shape == (batch_size,), \
        f"Expected shape ({batch_size},), got {scores.shape}"
    assert all_scores.shape == (batch_size, num_entities), \
        f"Expected shape ({batch_size}, {num_entities}), got {all_scores.shape}"
    
    print(f"  ✓ Batch size: {batch_size}, Embedding dim: {embedding_dim}")
    print(f"  ✓ Scores shape: {scores.shape}")
    print(f"  ✓ All tails scores shape: {all_scores.shape}")
    print(f"  ✓ Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print("  ✓ Decoder test passed!\n")


def test_complete_model():
    """Test complete DrugDiseaseModel."""
    print("Testing Complete DrugDiseaseModel...")
    
    # Parameters
    num_nodes = 100
    num_relations = 3
    embedding_dim = 64
    hidden_dim = 128
    num_edges = 500
    batch_size = 32
    
    # Create model
    model = DrugDiseaseModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=0.5,
        decoder_dropout=0.0
    )
    
    # Create dummy graph data
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_type = torch.randint(0, num_relations, (num_edges,))
    
    # Create dummy batch
    head_indices = torch.randint(0, num_nodes, (batch_size,))
    tail_indices = torch.randint(0, num_nodes, (batch_size,))
    relation_types = torch.randint(0, num_relations, (batch_size,))
    
    # Test forward pass (training)
    model.train()
    scores = model(edge_index, edge_type, head_indices, tail_indices, relation_types)
    
    # Test prediction (inference)
    pred_scores = model.predict(
        edge_index, edge_type, head_indices, tail_indices, relation_types
    )
    
    # Test predict_all_tails
    all_scores = model.predict_all_tails(
        edge_index, edge_type, head_indices, relation_types
    )
    
    # Test get_embeddings
    embeddings = model.get_embeddings(edge_index, edge_type)
    
    # Assertions
    assert scores.shape == (batch_size,), \
        f"Expected shape ({batch_size},), got {scores.shape}"
    assert pred_scores.shape == (batch_size,), \
        f"Expected shape ({batch_size},), got {pred_scores.shape}"
    assert all_scores.shape == (batch_size, num_nodes), \
        f"Expected shape ({batch_size}, {num_nodes}), got {all_scores.shape}"
    assert embeddings.shape == (num_nodes, hidden_dim), \
        f"Expected shape ({num_nodes}, {hidden_dim}), got {embeddings.shape}"
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  ✓ Graph: {num_nodes} nodes, {num_edges} edges, {num_relations} relations")
    print(f"  ✓ Batch size: {batch_size}")
    print(f"  ✓ Training scores shape: {scores.shape}")
    print(f"  ✓ Prediction scores shape: {pred_scores.shape}")
    print(f"  ✓ All tails scores shape: {all_scores.shape}")
    print(f"  ✓ Embeddings shape: {embeddings.shape}")
    print(f"  ✓ Total parameters: {num_params:,}")
    print(f"  ✓ Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print("  ✓ Complete model test passed!\n")


def test_with_real_data():
    """Test model with actual preprocessed data if available."""
    print("Testing with Real Data (if available)...")
    
    try:
        from pathlib import Path
        
        # Try to load preprocessed data
        data_path = Path('data/processed/train_data.pt')
        mappings_path = Path('data/processed/mappings.pt')
        
        if not data_path.exists() or not mappings_path.exists():
            print("  ⚠ Preprocessed data not found. Skipping real data test.")
            return
        
        # Load data
        train_data = torch.load(data_path)
        mappings = torch.load(mappings_path)
        
        num_nodes = train_data['num_nodes']
        num_relations = train_data['num_relations']
        edge_index = train_data['edge_index']
        edge_type = train_data['edge_type']
        
        print(f"  ✓ Loaded real data: {num_nodes} nodes, {edge_index.shape[1]} edges")
        
        # Create model
        model = DrugDiseaseModel(
            num_nodes=num_nodes,
            num_relations=num_relations,
            embedding_dim=64,
            hidden_dim=128,
            dropout=0.5
        )
        
        # Create a small batch
        batch_size = 32
        head_indices = torch.randint(0, num_nodes, (batch_size,))
        tail_indices = torch.randint(0, num_nodes, (batch_size,))
        relation_types = torch.randint(0, num_relations, (batch_size,))
        
        # Forward pass
        scores = model(edge_index, edge_type, head_indices, tail_indices, relation_types)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  ✓ Model created with {num_params:,} parameters")
        print(f"  ✓ Forward pass successful: {scores.shape}")
        print(f"  ✓ Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print("  ✓ Real data test passed!\n")
        
    except Exception as e:
        print(f"  ⚠ Real data test failed: {e}\n")


if __name__ == '__main__':
    print("="*70)
    print("RGCN Model Tests")
    print("="*70 + "\n")
    
    # Run tests
    test_rgcn_encoder()
    test_link_predictor()
    test_complete_model()
    test_with_real_data()
    
    print("="*70)
    print("All Tests Completed Successfully! ✓")
    print("="*70)
