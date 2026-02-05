"""
Model Registry for Link Prediction

Available encoders:
- rgcn: Relational GCN (uses relation types)
- mlp: MLP baseline (ignores graph structure)
- gcn: Standard GCN (ignores relation types) [TODO]
- gat: Graph Attention Network [TODO]

All encoders have the same interface:
    encoder(edge_index, edge_type) -> node_embeddings [num_nodes, hidden_dim]
"""

from .rgcn import DrugDiseaseRGCN, LinkPredictor, DrugDiseaseModel
from .mlp import MLPEncoder
from .gcn import GCNEncoder

# Model registry
ENCODER_REGISTRY = {
    'rgcn': DrugDiseaseRGCN,
    'mlp': MLPEncoder,
    'gcn': GCNEncoder,
    # 'gat': GATEncoder,  # TODO
}


def get_encoder(model_type: str):
    """Get encoder class by name."""
    model_type = model_type.lower()
    if model_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(ENCODER_REGISTRY.keys())}"
        )
    return ENCODER_REGISTRY[model_type]


def list_models():
    """List available models."""
    return list(ENCODER_REGISTRY.keys())


__all__ = [
    'DrugDiseaseRGCN',
    'LinkPredictor',
    'DrugDiseaseModel',
    'MLPEncoder',
    'ENCODER_REGISTRY',
    'get_encoder',
    'list_models',
]
