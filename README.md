# PrimeKG-RGCN-LinkPrediction
Relational Graph Convolutional Network (RGCN) for predicting drug-disease indications using PrimeKG knowledge graph. Leverages drug-gene and gene-disease relationships for multi-hop reasoning in biomedical link prediction.

## Overview

This project implements a graph neural network approach to predict potential drug-disease relationships by learning from the biomedical knowledge graph structure. The model leverages drug-gene interactions and gene-disease associations to perform multi-hop reasoning for drug repurposing and indication discovery.

### Key Features

- **RGCN Architecture**: Handles heterogeneous edge types in biomedical knowledge graphs
- **PrimeKG Integration**: Uses a high-quality, precision medicine knowledge graph
- **Link Prediction**: Predicts missing drug-disease relationships
- **Multi-hop Reasoning**: Captures indirect relationships through gene intermediates

## Methodology

```
Drug --[interacts]--> Gene --[associated]--> Disease
```

The model learns to predict drug-disease indications by:

1. Encoding drug-gene and gene-disease relationships
2. Learning node embeddings via relational graph convolutions
3. Predicting links between drug and disease nodes

## Dataset

This project uses [PrimeKG](https://github.com/mims-harvard/PrimeKG), a precision medicine knowledge graph containing:

- **4.5 million** relationships
- **20 different** data sources
- Multiple relation types including:
  - Drug-Gene interactions
  - Gene-Gene interactions
  - Gene-Disease associations

**Reference**: Chandak et al., "Building a knowledge graph to enable precision medicine." *Nature Scientific Data* (2023).
