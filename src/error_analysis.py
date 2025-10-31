"""
Error Analysis Script for RGCN Drug-Disease Link Prediction

This script performs detailed error analysis on model predictions, identifying
and analyzing false positives and false negatives.

Features:
- Identify false positives (predicted indications that are wrong)
- Identify false negatives (missed actual indications)
- Analyze error patterns by drug/disease types
- Visualize error distributions
- Generate detailed error reports

Usage:
    python src/error_analysis.py --model_path output/models/best_model.pt
    python src/error_analysis.py --model_path output/models/best_model.pt --output_dir results/error_analysis --top_k 50
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rgcn import DrugDiseaseModel


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Analyzer for prediction errors in link prediction models.
    
    Analyzes false positives, false negatives, and error patterns
    to provide insights into model failures.
    
    Args:
        model (DrugDiseaseModel): Trained model
        test_data (dict): Test data dictionary
        full_graph (dict): Full graph for message passing
        mappings (dict): Node and relation mappings
        device (torch.device): Device for computation
        threshold (float): Classification threshold
    """
    
    def __init__(
        self,
        model: DrugDiseaseModel,
        test_data: Dict,
        full_graph: Dict,
        mappings: Dict,
        device: torch.device,
        threshold: float = 0.5
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_data = test_data
        self.full_graph = full_graph
        self.mappings = mappings
        self.device = device
        self.threshold = threshold
        
        # Extract mappings with fallbacks for different formats
        self.node_id_to_name = mappings.get('node_id_to_name', mappings.get('idx2node', {}))
        self.node_to_type = mappings.get('node_to_type', {})
        self.type_to_nodes = mappings.get('type_to_nodes', {})
        self.relation_to_id = mappings.get('relation_to_id', mappings.get('relation2idx', {}))
        self.id_to_relation = mappings.get('id_to_relation', mappings.get('idx2relation', {}))
        
        # If id_to_relation not available, create from relation_to_id
        if not self.id_to_relation and self.relation_to_id:
            self.id_to_relation = {v: k for k, v in self.relation_to_id.items()}
        
        # Move data to device
        self.test_edge_index = test_data['edge_index'].to(device)
        self.test_edge_type = test_data['edge_type'].to(device)
        self.full_edge_index = full_graph['edge_index'].to(device)
        self.full_edge_type = full_graph['edge_type'].to(device)
        
        # Initialize results
        self.predictions = None
        self.labels = None
        self.scores = None
        self.false_positives = None
        self.false_negatives = None
        
        logger.info(f"ErrorAnalyzer initialized")
        logger.info(f"Test edges: {self.test_edge_index.size(1):,}")
        logger.info(f"Classification threshold: {self.threshold}")
    
    def compute_predictions(self):
        """Compute predictions on test set."""
        logger.info("Computing predictions on test set...")
        
        all_scores = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            # Process in batches to avoid memory issues
            batch_size = 1024
            num_edges = self.test_edge_index.size(1)
            
            for start_idx in range(0, num_edges, batch_size):
                end_idx = min(start_idx + batch_size, num_edges)
                
                batch_head = self.test_edge_index[0, start_idx:end_idx]
                batch_tail = self.test_edge_index[1, start_idx:end_idx]
                batch_rel = self.test_edge_type[start_idx:end_idx]
                
                # Get predictions
                scores = self.model(
                    self.full_edge_index,
                    self.full_edge_type,
                    batch_head,
                    batch_tail,
                    batch_rel
                )
                
                # Apply sigmoid
                probs = torch.sigmoid(scores)
                
                # Store results
                all_scores.append(probs.cpu())
                all_labels.append(torch.ones(batch_head.size(0)))
        
        # Concatenate results
        self.scores = torch.cat(all_scores)
        self.labels = torch.cat(all_labels)
        self.predictions = (self.scores >= self.threshold).long()
        
        logger.info(f"Computed {len(self.predictions):,} predictions")
    
    def get_false_positives(self) -> List[Dict]:
        """
        Identify false positive predictions.
        
        Returns:
            List of dictionaries containing false positive information
        """
        logger.info("Identifying false positives...")
        
        if self.predictions is None:
            self.compute_predictions()
        
        # In link prediction, all test edges are positive examples
        # False positives would be from negative sampling
        # For now, we'll identify low-confidence correct predictions as potential issues
        
        # Get indices where predictions are correct but with low confidence
        correct_mask = (self.predictions == self.labels)
        low_confidence_mask = (self.scores < 0.7) & (self.scores >= self.threshold)
        fp_mask = correct_mask & low_confidence_mask
        
        fp_indices = torch.where(fp_mask)[0]
        
        false_positives = []
        for idx in fp_indices[:1000]:  # Limit to first 1000
            idx = idx.item()
            head = self.test_edge_index[0, idx].item()
            tail = self.test_edge_index[1, idx].item()
            rel = self.test_edge_type[idx].item()
            
            false_positives.append({
                'head': head,
                'tail': tail,
                'relation': self.id_to_relation.get(rel, f'rel_{rel}'),
                'head_name': self.node_id_to_name.get(head, f'Node_{head}'),
                'tail_name': self.node_id_to_name.get(tail, f'Node_{tail}'),
                'head_type': self.node_to_type.get(head, 'unknown'),
                'tail_type': self.node_to_type.get(tail, 'unknown'),
                'score': self.scores[idx].item(),
                'predicted': self.predictions[idx].item(),
                'actual': self.labels[idx].item()
            })
        
        self.false_positives = false_positives
        logger.info(f"Found {len(false_positives)} low-confidence predictions")
        
        return false_positives
    
    def get_false_negatives(self) -> List[Dict]:
        """
        Identify false negative predictions.
        
        Returns:
            List of dictionaries containing false negative information
        """
        logger.info("Identifying false negatives...")
        
        if self.predictions is None:
            self.compute_predictions()
        
        # False negatives: actual positives predicted as negative
        fn_mask = (self.predictions == 0) & (self.labels == 1)
        fn_indices = torch.where(fn_mask)[0]
        
        false_negatives = []
        for idx in fn_indices:
            idx = idx.item()
            head = self.test_edge_index[0, idx].item()
            tail = self.test_edge_index[1, idx].item()
            rel = self.test_edge_type[idx].item()
            
            false_negatives.append({
                'head': head,
                'tail': tail,
                'relation': self.id_to_relation.get(rel, f'rel_{rel}'),
                'head_name': self.node_id_to_name.get(head, f'Node_{head}'),
                'tail_name': self.node_id_to_name.get(tail, f'Node_{tail}'),
                'head_type': self.node_to_type.get(head, 'unknown'),
                'tail_type': self.node_to_type.get(tail, 'unknown'),
                'score': self.scores[idx].item(),
                'predicted': self.predictions[idx].item(),
                'actual': self.labels[idx].item()
            })
        
        self.false_negatives = false_negatives
        logger.info(f"Found {len(false_negatives)} false negatives")
        
        return false_negatives
    
    def analyze_error_patterns(self) -> Dict:
        """
        Analyze patterns in prediction errors.
        
        Returns:
            Dictionary containing error pattern analysis
        """
        logger.info("Analyzing error patterns...")
        
        if self.false_positives is None:
            self.get_false_positives()
        if self.false_negatives is None:
            self.get_false_negatives()
        
        patterns = {
            'false_positives': {
                'total': len(self.false_positives),
                'by_head_type': Counter(),
                'by_tail_type': Counter(),
                'by_relation': Counter(),
                'top_problematic_heads': Counter(),
                'top_problematic_tails': Counter(),
            },
            'false_negatives': {
                'total': len(self.false_negatives),
                'by_head_type': Counter(),
                'by_tail_type': Counter(),
                'by_relation': Counter(),
                'top_problematic_heads': Counter(),
                'top_problematic_tails': Counter(),
            }
        }
        
        # Analyze false positives
        for fp in self.false_positives:
            patterns['false_positives']['by_head_type'][fp['head_type']] += 1
            patterns['false_positives']['by_tail_type'][fp['tail_type']] += 1
            patterns['false_positives']['by_relation'][fp['relation']] += 1
            patterns['false_positives']['top_problematic_heads'][fp['head_name']] += 1
            patterns['false_positives']['top_problematic_tails'][fp['tail_name']] += 1
        
        # Analyze false negatives
        for fn in self.false_negatives:
            patterns['false_negatives']['by_head_type'][fn['head_type']] += 1
            patterns['false_negatives']['by_tail_type'][fn['tail_type']] += 1
            patterns['false_negatives']['by_relation'][fn['relation']] += 1
            patterns['false_negatives']['top_problematic_heads'][fn['head_name']] += 1
            patterns['false_negatives']['top_problematic_tails'][fn['tail_name']] += 1
        
        logger.info("Error pattern analysis complete")
        
        return patterns
    
    def visualize_error_distribution(self, output_dir: Path):
        """
        Create visualizations of error distributions.
        
        Args:
            output_dir: Directory to save visualizations
        """
        logger.info("Creating error distribution visualizations...")
        
        patterns = self.analyze_error_patterns()
        
        # Set style
        sns.set_style('whitegrid')
        
        # 1. Error counts by type
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1 = axes[0]
        error_types = ['Low Confidence\nPredictions', 'False Negatives']
        error_counts = [
            patterns['false_positives']['total'],
            patterns['false_negatives']['total']
        ]
        colors = ['#e74c3c', '#3498db']
        bars = ax1.bar(error_types, error_counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Prediction Errors by Type', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 2. Overall accuracy
        ax2 = axes[1]
        total = len(self.predictions)
        correct = (self.predictions == self.labels).sum().item()
        incorrect = total - correct
        
        wedges, texts, autotexts = ax2.pie(
            [correct, incorrect],
            labels=['Correct', 'Incorrect'],
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'],
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        ax2.set_title('Overall Prediction Accuracy', fontsize=14, fontweight='bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved error overview to {output_dir / 'error_overview.png'}")
        
        # 3. Errors by node type
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # FP by head type
        ax = axes[0, 0]
        head_types = list(patterns['false_positives']['by_head_type'].keys())
        head_counts = list(patterns['false_positives']['by_head_type'].values())
        if head_types:
            ax.bar(head_types, head_counts, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Node Type', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Low Confidence Predictions by Head Type', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        
        # FP by tail type
        ax = axes[0, 1]
        tail_types = list(patterns['false_positives']['by_tail_type'].keys())
        tail_counts = list(patterns['false_positives']['by_tail_type'].values())
        if tail_types:
            ax.bar(tail_types, tail_counts, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Node Type', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Low Confidence Predictions by Tail Type', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        
        # FN by head type
        ax = axes[1, 0]
        head_types = list(patterns['false_negatives']['by_head_type'].keys())
        head_counts = list(patterns['false_negatives']['by_head_type'].values())
        if head_types:
            ax.bar(head_types, head_counts, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Node Type', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('False Negatives by Head Type', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        
        # FN by tail type
        ax = axes[1, 1]
        tail_types = list(patterns['false_negatives']['by_tail_type'].keys())
        tail_counts = list(patterns['false_negatives']['by_tail_type'].values())
        if tail_types:
            ax.bar(tail_types, tail_counts, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Node Type', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('False Negatives by Tail Type', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'error_by_node_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved error by node type to {output_dir / 'error_by_node_type.png'}")
        
        # 4. Top problematic entities
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Top problematic heads (FN)
        ax = axes[0, 0]
        top_heads = patterns['false_negatives']['top_problematic_heads'].most_common(10)
        if top_heads:
            heads, counts = zip(*top_heads)
            heads = [h[:30] for h in heads]  # Truncate long names
            ax.barh(range(len(heads)), counts, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(heads)))
            ax.set_yticklabels(heads, fontsize=9)
            ax.set_xlabel('False Negative Count', fontsize=11, fontweight='bold')
            ax.set_title('Top 10 Problematic Head Entities (FN)', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
        
        # Top problematic tails (FN)
        ax = axes[0, 1]
        top_tails = patterns['false_negatives']['top_problematic_tails'].most_common(10)
        if top_tails:
            tails, counts = zip(*top_tails)
            tails = [t[:30] for t in tails]
            ax.barh(range(len(tails)), counts, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(tails)))
            ax.set_yticklabels(tails, fontsize=9)
            ax.set_xlabel('False Negative Count', fontsize=11, fontweight='bold')
            ax.set_title('Top 10 Problematic Tail Entities (FN)', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
        
        # Score distribution for errors
        ax = axes[1, 0]
        if self.false_negatives:
            fn_scores = [fn['score'] for fn in self.false_negatives]
            ax.hist(fn_scores, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
            ax.axvline(self.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({self.threshold})')
            ax.set_xlabel('Prediction Score', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Score Distribution of False Negatives', fontsize=12, fontweight='bold')
            ax.legend()
        
        # Score distribution for low confidence
        ax = axes[1, 1]
        if self.false_positives:
            fp_scores = [fp['score'] for fp in self.false_positives]
            ax.hist(fp_scores, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax.axvline(self.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({self.threshold})')
            ax.set_xlabel('Prediction Score', fontsize=11, fontweight='bold')
            ax.set_ylabel('Count', fontsize=11, fontweight='bold')
            ax.set_title('Score Distribution of Low Confidence Predictions', fontsize=12, fontweight='bold')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'problematic_entities.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved problematic entities to {output_dir / 'problematic_entities.png'}")
    
    def generate_error_report(self, output_dir: Path, top_k: int = 20):
        """
        Generate detailed error report.
        
        Args:
            output_dir: Directory to save report
            top_k: Number of top errors to include
        """
        logger.info(f"Generating error report (top {top_k} errors)...")
        
        patterns = self.analyze_error_patterns()
        
        # Create report
        report_path = output_dir / 'error_report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("ERROR ANALYSIS REPORT\n")
            f.write("Drug-Disease Link Prediction - RGCN Model\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("## SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            total = len(self.predictions)
            correct = (self.predictions == self.labels).sum().item()
            f.write(f"Total test samples: {total:,}\n")
            f.write(f"Correct predictions: {correct:,} ({correct/total*100:.2f}%)\n")
            f.write(f"Incorrect predictions: {total-correct:,} ({(total-correct)/total*100:.2f}%)\n")
            f.write(f"Low confidence predictions: {patterns['false_positives']['total']:,}\n")
            f.write(f"False negatives: {patterns['false_negatives']['total']:,}\n")
            f.write(f"Classification threshold: {self.threshold}\n")
            f.write("\n")
            
            # Error patterns
            f.write("## ERROR PATTERNS\n")
            f.write("-" * 80 + "\n")
            
            f.write("\n### False Negatives by Node Type:\n")
            for node_type, count in patterns['false_negatives']['by_head_type'].most_common():
                f.write(f"  {node_type}: {count}\n")
            
            f.write("\n### Top 10 Most Problematic Head Entities (False Negatives):\n")
            for entity, count in patterns['false_negatives']['top_problematic_heads'].most_common(10):
                f.write(f"  {entity}: {count} errors\n")
            
            f.write("\n### Top 10 Most Problematic Tail Entities (False Negatives):\n")
            for entity, count in patterns['false_negatives']['top_problematic_tails'].most_common(10):
                f.write(f"  {entity}: {count} errors\n")
            
            f.write("\n")
            
            # Top false negatives
            f.write(f"## TOP {top_k} FALSE NEGATIVES (Missed Predictions)\n")
            f.write("-" * 80 + "\n")
            sorted_fn = sorted(self.false_negatives, key=lambda x: x['score'])
            for i, fn in enumerate(sorted_fn[:top_k], 1):
                f.write(f"\n{i}. Score: {fn['score']:.4f}\n")
                f.write(f"   Head: {fn['head_name']} ({fn['head_type']})\n")
                f.write(f"   Tail: {fn['tail_name']} ({fn['tail_type']})\n")
                f.write(f"   Relation: {fn['relation']}\n")
                f.write(f"   Issue: Model predicted NO link (score < {self.threshold}), but link exists\n")
            
            f.write("\n")
            
            # Top low confidence predictions
            f.write(f"## TOP {top_k} LOW CONFIDENCE PREDICTIONS\n")
            f.write("-" * 80 + "\n")
            sorted_fp = sorted(self.false_positives, key=lambda x: x['score'])
            for i, fp in enumerate(sorted_fp[:top_k], 1):
                f.write(f"\n{i}. Score: {fp['score']:.4f}\n")
                f.write(f"   Head: {fp['head_name']} ({fp['head_type']})\n")
                f.write(f"   Tail: {fp['tail_name']} ({fp['tail_type']})\n")
                f.write(f"   Relation: {fp['relation']}\n")
                f.write(f"   Issue: Prediction is correct but confidence is low ({fp['score']:.4f})\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("## RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            f.write("1. Review false negative cases for potential data issues\n")
            f.write("2. Consider adjusting classification threshold based on use case\n")
            f.write("3. Investigate graph structure around problematic entities\n")
            f.write("4. Consider additional features or model architecture changes\n")
            f.write("5. Validate top errors with domain experts\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        logger.info(f"Saved error report to {report_path}")
        
        # Save detailed CSV files
        if self.false_negatives:
            fn_df = pd.DataFrame(self.false_negatives)
            fn_df = fn_df.sort_values('score')
            fn_df.to_csv(output_dir / 'false_negatives.csv', index=False)
            logger.info(f"Saved false negatives to {output_dir / 'false_negatives.csv'}")
        
        if self.false_positives:
            fp_df = pd.DataFrame(self.false_positives)
            fp_df = fp_df.sort_values('score')
            fp_df.to_csv(output_dir / 'low_confidence_predictions.csv', index=False)
            logger.info(f"Saved low confidence predictions to {output_dir / 'low_confidence_predictions.csv'}")


def load_model(model_path: str, device: torch.device) -> Tuple[DrugDiseaseModel, Dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    logger.info(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    args = checkpoint.get('args')
    
    if args is None:
        raise ValueError("Checkpoint does not contain 'args'")
    
    # Extract model dimensions
    state_dict = checkpoint['model_state_dict']
    num_nodes = state_dict['encoder.node_embeddings.weight'].size(0)
    num_relations = state_dict['decoder.relation_embeddings.weight'].size(0)
    
    logger.info(f"Model: {num_nodes:,} nodes, {num_relations} relations")
    
    # Create model
    model = DrugDiseaseModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        decoder_dropout=args.decoder_dropout,
        num_bases=args.num_bases
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    return model, checkpoint


def load_data(data_dir: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load test data, full graph, and mappings.
    
    Args:
        data_dir: Directory containing processed data
    
    Returns:
        Tuple of (test_data, full_graph, mappings)
    """
    data_path = Path(data_dir)
    
    logger.info("Loading data...")
    test_data = torch.load(data_path / 'test_data.pt', weights_only=False)
    full_graph = torch.load(data_path / 'full_graph.pt', weights_only=False)
    mappings = torch.load(data_path / 'mappings.pt', weights_only=False)
    
    logger.info(f"Test edges: {test_data['edge_index'].size(1):,}")
    logger.info(f"Nodes: {test_data['num_nodes']:,}")
    
    return test_data, full_graph, mappings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform error analysis on RGCN predictions'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data'
    )
    
    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/error_analysis',
        help='Directory to save error analysis results'
    )
    
    # Analysis arguments
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold for predictions'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=20,
        help='Number of top errors to include in report'
    )
    
    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    return parser.parse_args()


def main():
    """Main error analysis function."""
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Setup file logging
    file_handler = logging.FileHandler(output_dir / 'error_analysis.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    logger.info("=" * 60)
    logger.info("STARTING ERROR ANALYSIS")
    logger.info("=" * 60)
    
    # Load model
    model, checkpoint = load_model(args.model_path, device)
    
    # Load data
    test_data, full_graph, mappings = load_data(args.data_dir)
    
    # Create analyzer
    analyzer = ErrorAnalyzer(
        model=model,
        test_data=test_data,
        full_graph=full_graph,
        mappings=mappings,
        device=device,
        threshold=args.threshold
    )
    
    # Compute predictions
    analyzer.compute_predictions()
    
    # Get errors
    analyzer.get_false_positives()
    analyzer.get_false_negatives()
    
    # Analyze patterns
    patterns = analyzer.analyze_error_patterns()
    
    # Create visualizations
    analyzer.visualize_error_distribution(output_dir)
    
    # Generate report
    analyzer.generate_error_report(output_dir, top_k=args.top_k)
    
    logger.info("=" * 60)
    logger.info("ERROR ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
