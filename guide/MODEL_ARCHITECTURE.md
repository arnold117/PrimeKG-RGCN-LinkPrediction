# Model Architecture Guide

This guide documents the RGCN model architecture implemented in `src/models/rgcn.py` for drug-disease link prediction.

## Overview

The model combines a Relational Graph Convolutional Network (RGCN) encoder with a DistMult decoder to learn from heterogeneous biomedical knowledge graphs and predict drug-disease associations.

## Architecture Components

### 1. DrugDiseaseRGCN (Encoder)

The encoder learns node embeddings by propagating information through the heterogeneous graph using RGCN layers.

#### Architecture

```
Input: Learnable node embeddings (num_nodes, embedding_dim)
    ↓
RGCN Layer 1: (embedding_dim → hidden_dim)
    ↓
ReLU Activation
    ↓
Dropout (rate: dropout)
    ↓
RGCN Layer 2: (hidden_dim → hidden_dim)
    ↓
Output: Node embeddings (num_nodes, hidden_dim)
```

#### Parameters

- `num_nodes` (int): Total number of nodes in the graph
- `num_relations` (int): Number of relation types
- `embedding_dim` (int, default: 64): Initial node embedding dimension
- `hidden_dim` (int, default: 128): Hidden representation dimension
- `dropout` (float, default: 0.5): Dropout probability for regularization
- `num_bases` (int, optional): Number of bases for RGCN basis decomposition
  - If `None`, uses full parameter matrices (higher capacity, more parameters)
  - If set (e.g., 30), uses basis decomposition (parameter sharing, fewer params)

#### Key Features

- **Learnable node embeddings**: Each node has a trainable embedding vector initialized with Xavier uniform
- **Relation-specific message passing**: RGCN layers aggregate neighbor information based on edge types
- **Two-layer architecture**: Captures multi-hop graph structure (up to 2-hop neighborhoods)
- **Dropout regularization**: Prevents overfitting on training edges

#### Methods

- `forward(edge_index, edge_type, node_indices=None)`: Encode graph and return node embeddings
- `get_node_embeddings(node_indices)`: Get initial embeddings for specific nodes

### 2. LinkPredictor (Decoder)

The decoder uses DistMult scoring to predict link existence between node pairs.

#### Scoring Function

DistMult computes a score for triple (head, relation, tail):

```
score = <h, r, t> = Σ(h ⊙ r ⊙ t)
```

where:
- `h` = head entity embedding (from encoder)
- `r` = learnable relation embedding
- `t` = tail entity embedding (from encoder)
- `⊙` = element-wise multiplication
- `Σ` = sum across dimensions

#### Parameters

- `num_relations` (int): Number of relation types
- `embedding_dim` (int): Dimension of embeddings (matches encoder output)
- `dropout` (float, default: 0.0): Dropout probability for relation embeddings

#### Key Features

- **Learnable relation embeddings**: Each relation type has a trainable embedding
- **Bilinear scoring**: DistMult is a simplified bilinear model (diagonal relation matrix)
- **Fast computation**: Element-wise operations enable efficient batch scoring
- **Symmetric relations**: DistMult assumes symmetric relations (score(h,r,t) = score(t,r,h))

#### Methods

- `forward(head_embeddings, tail_embeddings, relation_types)`: Score specific triples
- `score_all_tails(head_embeddings, relation_types, all_tail_embeddings)`: Score all possible tails for ranking

### 3. DrugDiseaseModel (Complete Model)

The complete model combines encoder and decoder for end-to-end training and inference.

#### Architecture Flow

```
Graph (edge_index, edge_type)
    ↓
DrugDiseaseRGCN Encoder
    ↓
Node Embeddings (num_nodes, hidden_dim)
    ↓
Select head & tail embeddings
    ↓
LinkPredictor Decoder (DistMult)
    ↓
Prediction Scores
```

#### Parameters

All encoder and decoder parameters, plus:
- `decoder_dropout` (float, default: 0.0): Separate dropout for decoder

#### Methods

- `forward(edge_index, edge_type, head_indices, tail_indices, relation_types)`: Training forward pass
- `predict(...)`: Inference mode with no_grad
- `predict_all_tails(edge_index, edge_type, head_indices, relation_types)`: Rank all possible tails
- `get_embeddings(edge_index, edge_type)`: Extract learned node embeddings

## Model Parameters

### Default Configuration

```python
DrugDiseaseModel(
    num_nodes=30926,        # From PrimeKG processed data
    num_relations=3,        # drug-gene, gene-gene, gene-disease
    embedding_dim=64,       # Initial embedding size
    hidden_dim=128,         # RGCN output size
    dropout=0.5,            # RGCN dropout
    decoder_dropout=0.1,    # Decoder dropout
    num_bases=None          # Full parameterization
)
```

### Parameter Count

For PrimeKG dataset (30,926 nodes, 3 relations):

| Component | Parameters | Calculation |
|-----------|-----------|-------------|
| Node embeddings | 1,979,264 | 30,926 × 64 |
| RGCN Layer 1 | ~24,576 | 64 × 128 × 3 (relation-specific) |
| RGCN Layer 2 | ~49,152 | 128 × 128 × 3 |
| Relation embeddings | 384 | 3 × 128 |
| **Total** | **~2,078,208** | Matches checkpoint |

Note: Exact count depends on PyG's RGCNConv implementation details.

## Training

### Loss Function

Binary Cross-Entropy with Logits (BCE):

```python
loss = BCEWithLogitsLoss(scores, labels)
```

where:
- Positive samples: actual edges from training set (label = 1)
- Negative samples: randomly sampled non-edges (label = 0)

### Negative Sampling

- Default: 1 negative per positive edge
- Negatives sampled by corrupting tail entities
- Ensures class balance during training

### Optimization

- Optimizer: Adam (default) or AdamW
- Learning rate: 0.001 (default)
- Weight decay: 0.0 (default, can enable L2 regularization)
- Gradient clipping: 1.0 (prevents exploding gradients)

## Inference

### Link Classification

For a given (drug, relation, disease) triple:

1. Encode graph to get all node embeddings
2. Extract embeddings for drug and disease
3. Compute DistMult score
4. Apply sigmoid: `probability = sigmoid(score)`
5. Threshold (default: 0.5) to classify

### Link Ranking

For drug repurposing (find top diseases for a drug):

1. Encode graph to get all node embeddings
2. Extract embedding for query drug
3. Score against all disease embeddings
4. Rank diseases by score
5. Return top-K candidates

## Implementation Details

### RGCN Specifics

- Uses PyTorch Geometric's `RGCNConv` layer
- Supports basis decomposition for parameter efficiency
- Relation-specific transformation matrices
- Self-loops automatically handled

### DistMult Advantages

- **Efficiency**: Simple element-wise operations
- **Interpretability**: Relation embeddings are directly interpretable
- **Performance**: Empirically strong for knowledge graph completion
- **Symmetry**: Natural for undirected biological relations

### DistMult Limitations

- Cannot model asymmetric relations (e.g., "parent of" vs "child of")
- Cannot model relation composition (e.g., transitivity)
- Assumes relations are independent

## Memory Considerations

### GPU Memory Usage

For 30K nodes, batch_size=1024:

| Operation | Memory | Notes |
|-----------|--------|-------|
| Node embeddings | ~8 MB | 30,926 × 64 × 4 bytes |
| RGCN forward | ~500 MB | Depends on edge count |
| Batch scoring | ~16 MB | 1024 × 128 × 4 bytes × 3 |
| Gradients | ~2× model | During backward pass |

### Optimization Tips

- Use gradient accumulation for larger effective batch sizes
- Enable basis decomposition (`num_bases=30`) to reduce RGCN parameters
- Sample subgraphs for very large graphs (not implemented)

## Extending the Model

### Adding More RGCN Layers

```python
self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations)
```

Trade-off: Deeper models capture longer-range dependencies but are harder to train.

### Alternative Decoders

Replace DistMult with:
- **TransE**: Translation-based scoring (h + r ≈ t)
- **ComplEx**: Complex-valued embeddings (handles asymmetry)
- **ConvE**: Convolutional decoder (more expressive)

### Heterogeneous Node Types

Current implementation treats all nodes uniformly. To add node-type-specific processing:
- Separate embedding tables per node type
- Type-specific transformations before RGCN
- Attention mechanisms for type-aware aggregation

## Model Checkpoints

### Checkpoint Contents

Saved via `torch.save()`:

```python
{
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'best_val_loss': best_val_loss,
    'best_val_acc': best_val_acc,
    # ... training metadata
}
```

### Loading Checkpoints

```python
checkpoint = torch.load('output/models/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Checkpoint Files

- `output/checkpoints/checkpoint_epoch_N.pt`: Regular checkpoints (every 10 epochs)
- `output/models/best_model.pt`: Best validation performance
- `output/models/final_model.pt`: Last epoch (for final evaluation)

## Testing

The model file includes standalone tests:

```bash
python src/models/rgcn.py
```

Tests verify:
- Encoder forward pass and output shapes
- Decoder scoring and output ranges
- Complete model integration
- Gradient flow

## References

### RGCN
Schlichtkrull et al., "Modeling Relational Data with Graph Convolutional Networks" (ESWC 2018)

### DistMult
Yang et al., "Embedding Entities and Relations for Learning and Inference in Knowledge Bases" (ICLR 2015)

### PyTorch Geometric
Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric" (ICLR 2019)

## Related Documentation

- Training guide: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- Evaluation guide: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
- Main README: [../README.md](../README.md)
