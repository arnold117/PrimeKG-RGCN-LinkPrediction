"""
PrimeKG Data Preprocessing for RGCN Link Prediction

This script processes the PrimeKG knowledge graph to create a drug-gene-disease
subgraph suitable for RGCN-based link prediction tasks.

Steps:
1. Load raw PrimeKG data
2. Filter to drug-gene-disease subgraph
3. Keep only specified relation types
4. Build node and relation mappings
5. Convert to PyTorch Geometric format
6. Split edges for train/val/test
7. Save processed data
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('temp/preprocess.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PrimeKGPreprocessor:
    """Preprocessor for PrimeKG knowledge graph data."""
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize the preprocessor.
        
        Args:
            raw_data_path: Path to raw kg.csv file
            processed_data_path: Directory to save processed data
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        # Node types we want to keep
        self.target_node_types = {'drug', 'gene/protein', 'disease'}
        
        # Relation types we want to keep (standardized names)
        # Based on PrimeKG schema: drug_protein, protein_protein, disease_protein
        self.target_relations = {
            'drug-gene': ['drug_protein'],
            'gene-gene': ['protein_protein'],
            'gene-disease': ['disease_protein']
        }
        
        # Mappings
        self.node2idx = {}  # {(node_id, node_type): sequential_idx}
        self.idx2node = {}  # {sequential_idx: (node_id, node_name, node_type)}
        self.relation2idx = {}  # {relation_name: idx}
        self.idx2relation = {}  # {idx: relation_name}
        
        # Statistics
        self.stats = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load raw PrimeKG data.
        
        Returns:
            DataFrame with raw KG data
        """
        logger.info(f"Loading data from {self.raw_data_path}")
        
        try:
            df = pd.read_csv(self.raw_data_path)
            logger.info(f"Loaded {len(df):,} edges")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            self.stats['total_edges'] = len(df)
            self.stats['total_node_types'] = df['x_type'].nunique()
            self.stats['total_relation_types'] = df['relation'].nunique()
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def filter_subgraph(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to drug-gene-disease subgraph with target relations.
        
        Args:
            df: Raw KG DataFrame
            
        Returns:
            Filtered DataFrame
        """
        logger.info("Filtering to drug-gene-disease subgraph...")
        
        # Filter by node types
        node_type_mask = (
            df['x_type'].isin(self.target_node_types) & 
            df['y_type'].isin(self.target_node_types)
        )
        df_filtered = df[node_type_mask].copy()
        logger.info(f"After node type filtering: {len(df_filtered):,} edges")
        
        # Filter by relation types
        all_target_relations = []
        for relations in self.target_relations.values():
            all_target_relations.extend(relations)
        
        relation_mask = df_filtered['relation'].isin(all_target_relations)
        df_filtered = df_filtered[relation_mask].copy()
        logger.info(f"After relation filtering: {len(df_filtered):,} edges")
        
        # Standardize relation names
        relation_mapping = {}
        for standard_name, variants in self.target_relations.items():
            for variant in variants:
                relation_mapping[variant] = standard_name
        
        df_filtered['relation_standard'] = df_filtered['relation'].map(relation_mapping)
        
        self.stats['filtered_edges'] = len(df_filtered)
        self.stats['filtered_relations'] = df_filtered['relation_standard'].nunique()
        
        return df_filtered
    
    def build_mappings(self, df: pd.DataFrame) -> None:
        """
        Build node and relation index mappings.
        
        Args:
            df: Filtered KG DataFrame
        """
        logger.info("Building node and relation mappings...")
        
        # Collect all unique nodes
        nodes = set()
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting nodes"):
            # Source node (convert ID to string for consistency)
            nodes.add((str(row['x_id']), row['x_name'], row['x_type']))
            # Target node (convert ID to string for consistency)
            nodes.add((str(row['y_id']), row['y_name'], row['y_type']))
        
        # Create node mappings (sorted for reproducibility)
        nodes = sorted(nodes, key=lambda x: (x[2], x[0], x[1]))
        for idx, (node_id, node_name, node_type) in enumerate(nodes):
            key = (str(node_id), node_type)
            self.node2idx[key] = idx
            self.idx2node[idx] = (str(node_id), node_name, node_type)
        
        logger.info(f"Total unique nodes: {len(self.node2idx):,}")
        
        # Create relation mappings
        unique_relations = sorted(df['relation_standard'].unique())
        for idx, relation in enumerate(unique_relations):
            self.relation2idx[relation] = idx
            self.idx2relation[idx] = relation
        
        logger.info(f"Total unique relations: {len(self.relation2idx)}")
        
        # Node type statistics
        node_types = [node_info[2] for node_info in self.idx2node.values()]
        for node_type in self.target_node_types:
            count = node_types.count(node_type)
            logger.info(f"  {node_type}: {count:,} nodes")
            self.stats[f'num_{node_type}_nodes'] = count
        
        # Relation statistics
        for relation, idx in self.relation2idx.items():
            count = (df['relation_standard'] == relation).sum()
            logger.info(f"  {relation}: {count:,} edges")
            self.stats[f'num_{relation}_edges'] = count
    
    def convert_to_pyg_format(self, df: pd.DataFrame) -> Dict:
        """
        Convert to PyTorch Geometric format.
        
        Args:
            df: Filtered KG DataFrame
            
        Returns:
            Dictionary with edge_index and edge_type tensors
        """
        logger.info("Converting to PyTorch Geometric format...")
        
        edge_list = []
        edge_types = []
        num_nodes = len(self.node2idx)
        invalid_edges = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting edges"):
            # Get source and target node indices
            src_key = (str(row['x_id']), row['x_type'])
            tgt_key = (str(row['y_id']), row['y_type'])
            
            # Skip if nodes not in mapping
            if src_key not in self.node2idx or tgt_key not in self.node2idx:
                invalid_edges += 1
                continue
            
            src_idx = self.node2idx[src_key]
            tgt_idx = self.node2idx[tgt_key]
            
            # Validate indices are within bounds
            if src_idx >= num_nodes or tgt_idx >= num_nodes or src_idx < 0 or tgt_idx < 0:
                invalid_edges += 1
                continue
            
            # Get relation index
            rel_idx = self.relation2idx[row['relation_standard']]
            
            # Add edge (bidirectional for undirected graph)
            edge_list.append([src_idx, tgt_idx])
            edge_types.append(rel_idx)
            
            # Add reverse edge
            edge_list.append([tgt_idx, src_idx])
            edge_types.append(rel_idx)
        
        if invalid_edges > 0:
            logger.warning(f"Skipped {invalid_edges} invalid edges during conversion")
        
        # Convert to tensors
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_type = torch.tensor(edge_types, dtype=torch.long)
        
        # Final validation
        max_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
        if max_idx >= num_nodes:
            logger.error(f"Found edge index {max_idx} >= num_nodes {num_nodes}")
            # Filter invalid edges
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, valid_mask]
            edge_type = edge_type[valid_mask]
            logger.warning(f"Filtered to {edge_index.shape[1]} valid edges")
        
        logger.info(f"Created edge_index: {edge_index.shape}")
        logger.info(f"Created edge_type: {edge_type.shape}")
        
        data = {
            'edge_index': edge_index,
            'edge_type': edge_type,
            'num_nodes': num_nodes,
            'num_relations': len(self.relation2idx)
        }
        
        return data
    
    def split_edges(
        self, 
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split drug-disease edges into train/val/test sets.
        
        Note: We split only specific edge types for link prediction tasks,
        while keeping all other edges in the training set for message passing.
        
        Args:
            df: Filtered KG DataFrame
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting edges into train/val/test sets...")
        
        # Identify target edges for link prediction
        # We'll split drug-gene edges since these are crucial for drug discovery
        # and we have a good number of them for meaningful train/val/test splits
        target_relation = 'drug-gene'
        
        target_mask = df['relation_standard'] == target_relation
        target_df = df[target_mask].copy()
        other_df = df[~target_mask].copy()
        
        logger.info(f"Target edges ({target_relation}): {len(target_df):,}")
        logger.info(f"Other edges (kept in train): {len(other_df):,}")
        
        # Check if we have target edges to split
        if len(target_df) == 0:
            logger.warning(f"No {target_relation} edges found! Using gene-disease instead.")
            target_relation = 'gene-disease'
            target_mask = df['relation_standard'] == target_relation
            target_df = df[target_mask].copy()
            other_df = df[~target_mask].copy()
            logger.info(f"Target edges ({target_relation}): {len(target_df):,}")
            logger.info(f"Other edges (kept in train): {len(other_df):,}")
        
        # Split target edges
        np.random.seed(random_seed)
        
        # First split: train vs (val + test)
        train_target, val_test_target = train_test_split(
            target_df,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed
        )
        
        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_target, test_target = train_test_split(
            val_test_target,
            test_size=(1 - val_ratio_adjusted),
            random_state=random_seed
        )
        
        # Training set includes all training target edges + all other edges
        train_df = pd.concat([train_target, other_df], ignore_index=True)
        val_df = val_target.copy()
        test_df = test_target.copy()
        
        logger.info(f"Train edges: {len(train_df):,} ({len(train_target):,} target + {len(other_df):,} other)")
        logger.info(f"Val edges: {len(val_df):,}")
        logger.info(f"Test edges: {len(test_df):,}")
        
        self.stats['train_edges'] = len(train_df)
        self.stats['val_edges'] = len(val_df)
        self.stats['test_edges'] = len(test_df)
        self.stats['train_target_edges'] = len(train_target)
        
        return train_df, val_df, test_df
    
    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_df: pd.DataFrame
    ) -> None:
        """
        Save all processed data to disk.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            full_df: Full DataFrame (for graph structure)
        """
        logger.info(f"Saving processed data to {self.processed_data_path}")
        
        # Convert each split to PyG format
        train_data = self.convert_to_pyg_format(train_df)
        val_data = self.convert_to_pyg_format(val_df)
        test_data = self.convert_to_pyg_format(test_df)
        
        # Also save the full graph structure for message passing during val/test
        logger.info("Converting full graph structure...")
        full_data = self.convert_to_pyg_format(full_df)
        torch.save(full_data, self.processed_data_path / 'full_graph.pt')
        logger.info("Saved full graph structure")
        
        # Save tensors
        torch.save(train_data, self.processed_data_path / 'train_data.pt')
        torch.save(val_data, self.processed_data_path / 'val_data.pt')
        torch.save(test_data, self.processed_data_path / 'test_data.pt')
        logger.info("Saved PyTorch tensors")
        
        # Save mappings
        mappings = {
            'node2idx': self.node2idx,
            'idx2node': self.idx2node,
            'relation2idx': self.relation2idx,
            'idx2relation': self.idx2relation
        }
        torch.save(mappings, self.processed_data_path / 'mappings.pt')
        logger.info("Saved mappings")
        
        # Save raw DataFrames for inspection
        train_df.to_csv(self.processed_data_path / 'train_edges.csv', index=False)
        val_df.to_csv(self.processed_data_path / 'val_edges.csv', index=False)
        test_df.to_csv(self.processed_data_path / 'test_edges.csv', index=False)
        logger.info("Saved CSV files")
        
        # Save statistics
        stats_df = pd.DataFrame([self.stats])
        stats_df.to_csv(self.processed_data_path / 'statistics.csv', index=False)
        logger.info("Saved statistics")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("PREPROCESSING SUMMARY")
        logger.info("="*50)
        for key, value in self.stats.items():
            logger.info(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")
        logger.info("="*50 + "\n")
    
    def process(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> None:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_seed: Random seed for reproducibility
        """
        logger.info("Starting PrimeKG preprocessing pipeline...")
        logger.info(f"Random seed: {random_seed}")
        
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Filter subgraph
        df_filtered = self.filter_subgraph(df)
        
        # Step 3: Build mappings
        self.build_mappings(df_filtered)
        
        # Step 4: Split edges
        train_df, val_df, test_df = self.split_edges(
            df_filtered,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )
        
        # Step 5: Save processed data
        self.save_processed_data(train_df, val_df, test_df, df_filtered)
        
        logger.info("Preprocessing complete!")


def main():
    """Main function to run preprocessing."""
    parser = argparse.ArgumentParser(
        description='Preprocess PrimeKG data for RGCN link prediction'
    )
    parser.add_argument(
        '--raw-data',
        type=str,
        default='data/raw/kg.csv',
        help='Path to raw kg.csv file'
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Directory to save processed data'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, val, and test ratios must sum to 1.0")
    
    # Run preprocessing
    preprocessor = PrimeKGPreprocessor(
        raw_data_path=args.raw_data,
        processed_data_path=args.processed_dir
    )
    
    preprocessor.process(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()
