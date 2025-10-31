"""
Evaluation Script for RGCN Drug-Disease Link Prediction

This script evaluates a trained DrugDiseaseModel on the test set, computing
various metrics and generating visualizations.

Metrics Computed:
    - Hits@10: Percentage of correct predictions in top 10
    - Hits@50: Percentage of correct predictions in top 50
    - Mean Reciprocal Rank (MRR): Average of 1/rank for correct predictions
    - AUC-ROC: Area Under the ROC Curve

Outputs:
    - results.json: JSON file with all computed metrics
    - confusion_matrix.png: Confusion matrix visualization
    - roc_curve.png: ROC curve visualization
    - metrics_summary.txt: Human-readable summary of metrics

Usage:
    python src/evaluate.py --model_path output/models/best_model.pt
    python src/evaluate.py --model_path output/checkpoints/checkpoint_epoch_100.pt --output_dir custom_results/
    python src/evaluate.py --model_path output/models/best_model.pt --batch_size 512
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

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


class ModelEvaluator:
    """
    Evaluator for link prediction models.
    
    Args:
        model (DrugDiseaseModel): Trained model to evaluate
        test_data (dict): Test data dictionary
        full_graph (dict): Full graph structure for message passing
        device (torch.device): Device to use for evaluation
        batch_size (int): Batch size for evaluation
    """
    
    def __init__(
        self,
        model: DrugDiseaseModel,
        test_data: Dict,
        full_graph: Dict,
        device: torch.device,
        batch_size: int = 1024
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_data = test_data
        self.full_graph = full_graph
        self.device = device
        self.batch_size = batch_size
        
        # Move graph data to device
        self.test_edge_index = test_data['edge_index'].to(device)
        self.test_edge_type = test_data['edge_type'].to(device)
        self.full_edge_index = full_graph['edge_index'].to(device)
        self.full_edge_type = full_graph['edge_type'].to(device)
        
        self.num_nodes = test_data['num_nodes']
        self.num_test_edges = self.test_edge_index.size(1)
        
        logger.info(f"Evaluator initialized with {self.num_test_edges:,} test edges")
        logger.info(f"Number of nodes: {self.num_nodes:,}")
        logger.info(f"Batch size: {self.batch_size}")
    
    def _generate_negative_samples(
        self,
        pos_head: torch.Tensor,
        pos_tail: torch.Tensor,
        pos_rel: torch.Tensor,
        num_neg_samples: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate negative samples for evaluation.
        
        Args:
            pos_head: Positive head entities
            pos_tail: Positive tail entities
            pos_rel: Relation types
            num_neg_samples: Number of negative samples per positive
        
        Returns:
            Tuple of (neg_head, neg_tail, neg_rel)
        """
        batch_size = pos_head.size(0)
        
        # Repeat positive samples
        neg_head = pos_head.repeat_interleave(num_neg_samples)
        neg_tail = pos_tail.repeat_interleave(num_neg_samples)
        neg_rel = pos_rel.repeat_interleave(num_neg_samples)
        
        # Randomly corrupt either head or tail
        total_neg = batch_size * num_neg_samples
        corrupt_head_mask = torch.rand(total_neg, device=self.device) < 0.5
        
        # Generate random entities
        random_entities = torch.randint(
            0, self.num_nodes, (total_neg,), device=self.device
        )
        
        # Apply corruption
        neg_head = torch.where(corrupt_head_mask, random_entities, neg_head)
        neg_tail = torch.where(~corrupt_head_mask, random_entities, neg_tail)
        
        return neg_head, neg_tail, neg_rel
    
    def compute_scores_and_labels(
        self,
        num_neg_samples: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute prediction scores and true labels for all test samples.
        
        Args:
            num_neg_samples: Number of negative samples per positive sample
        
        Returns:
            Tuple of (scores, labels) as numpy arrays
        """
        all_scores = []
        all_labels = []
        
        logger.info("Computing prediction scores...")
        
        # Process test data in batches
        num_batches = (self.num_test_edges + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.num_test_edges)
                
                # Get positive samples
                batch_head = self.test_edge_index[0, start_idx:end_idx]
                batch_tail = self.test_edge_index[1, start_idx:end_idx]
                batch_rel = self.test_edge_type[start_idx:end_idx]
                
                # Generate negative samples
                neg_head, neg_tail, neg_rel = self._generate_negative_samples(
                    batch_head, batch_tail, batch_rel, num_neg_samples
                )
                
                # Combine positive and negative samples
                all_heads = torch.cat([batch_head, neg_head])
                all_tails = torch.cat([batch_tail, neg_tail])
                all_rels = torch.cat([batch_rel, neg_rel])
                
                # Get prediction scores
                scores = self.model(
                    self.full_edge_index,
                    self.full_edge_type,
                    all_heads,
                    all_tails,
                    all_rels
                )
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(scores)
                
                # Create labels (1 for positive, 0 for negative)
                batch_size_current = batch_head.size(0)
                pos_labels = torch.ones(batch_size_current, device=self.device)
                neg_labels = torch.zeros(neg_head.size(0), device=self.device)
                labels = torch.cat([pos_labels, neg_labels])
                
                all_scores.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate all batches
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        
        logger.info(f"Computed scores for {len(scores):,} samples")
        logger.info(f"Positive samples: {labels.sum():.0f}")
        logger.info(f"Negative samples: {(1 - labels).sum():.0f}")
        
        return scores, labels
    
    def compute_ranking_metrics(self, k_values: List[int] = [10, 50]) -> Dict:
        """
        Compute ranking-based metrics: Hits@K and MRR.
        
        For each test triple (h, r, t), we rank t among all possible entities
        and compute metrics based on the rank of the true tail.
        
        Args:
            k_values: List of K values for Hits@K metric
        
        Returns:
            Dictionary with ranking metrics
        """
        logger.info("Computing ranking metrics...")
        
        ranks = []
        reciprocal_ranks = []
        
        # Process test data in batches
        num_batches = (self.num_test_edges + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Computing ranks"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, self.num_test_edges)
                
                # Get test triples
                batch_head = self.test_edge_index[0, start_idx:end_idx]
                batch_tail = self.test_edge_index[1, start_idx:end_idx]
                batch_rel = self.test_edge_type[start_idx:end_idx]
                
                # Encode all nodes
                all_node_embeddings = self.model.encoder(
                    self.full_edge_index,
                    self.full_edge_type
                )
                
                # Get head embeddings
                head_emb = all_node_embeddings[batch_head]
                
                # Score all possible tails
                scores = self.model.decoder.score_all_tails(
                    head_emb, batch_rel, all_node_embeddings
                )
                
                # Get ranks of true tails
                # Rank is 1-indexed (1 = best)
                for i, true_tail in enumerate(batch_tail):
                    # Sort scores in descending order
                    sorted_indices = torch.argsort(scores[i], descending=True)
                    
                    # Find position of true tail (0-indexed)
                    rank_0indexed = (sorted_indices == true_tail).nonzero(as_tuple=True)[0].item()
                    
                    # Convert to 1-indexed rank
                    rank = rank_0indexed + 1
                    ranks.append(rank)
                    reciprocal_ranks.append(1.0 / rank)
        
        ranks = np.array(ranks)
        reciprocal_ranks = np.array(reciprocal_ranks)
        
        # Compute metrics
        metrics = {
            'mrr': float(np.mean(reciprocal_ranks)),
            'mean_rank': float(np.mean(ranks)),
            'median_rank': float(np.median(ranks))
        }
        
        # Compute Hits@K
        for k in k_values:
            hits_at_k = np.mean(ranks <= k)
            metrics[f'hits@{k}'] = float(hits_at_k)
        
        logger.info(f"MRR: {metrics['mrr']:.4f}")
        logger.info(f"Mean Rank: {metrics['mean_rank']:.2f}")
        logger.info(f"Median Rank: {metrics['median_rank']:.2f}")
        for k in k_values:
            logger.info(f"Hits@{k}: {metrics[f'hits@{k}']:.4f}")
        
        return metrics
    
    def compute_classification_metrics(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5
    ) -> Dict:
        """
        Compute classification metrics: AUC-ROC, precision, recall, F1.
        
        Args:
            scores: Prediction scores
            labels: True labels
            threshold: Classification threshold
        
        Returns:
            Dictionary with classification metrics
        """
        logger.info("Computing classification metrics...")
        
        # Make binary predictions
        predictions = (scores >= threshold).astype(int)
        
        # Compute metrics
        metrics = {
            'auc_roc': float(roc_auc_score(labels, scores)),
            'auc_pr': float(average_precision_score(labels, scores)),
            'precision': float(precision_score(labels, predictions)),
            'recall': float(recall_score(labels, predictions)),
            'f1_score': float(f1_score(labels, predictions)),
            'threshold': threshold
        }
        
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"AUC-PR: {metrics['auc_pr']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def evaluate(
        self,
        num_neg_samples: int = 1,
        k_values: List[int] = [10, 50]
    ) -> Dict:
        """
        Run complete evaluation.
        
        Args:
            num_neg_samples: Number of negative samples per positive
            k_values: List of K values for Hits@K
        
        Returns:
            Dictionary with all metrics
        """
        logger.info("=" * 60)
        logger.info("Starting evaluation...")
        logger.info("=" * 60)
        
        # Compute scores and labels
        scores, labels = self.compute_scores_and_labels(num_neg_samples)
        
        # Compute classification metrics
        classification_metrics = self.compute_classification_metrics(scores, labels)
        
        # Compute ranking metrics
        ranking_metrics = self.compute_ranking_metrics(k_values)
        
        # Combine all metrics
        all_metrics = {
            'classification': classification_metrics,
            'ranking': ranking_metrics,
            'test_edges': self.num_test_edges,
            'num_nodes': self.num_nodes
        }
        
        # Store scores and labels for visualization
        self.scores = scores
        self.labels = labels
        
        logger.info("=" * 60)
        logger.info("Evaluation completed!")
        logger.info("=" * 60)
        
        return all_metrics


class ResultsVisualizer:
    """
    Visualizer for evaluation results.
    
    Args:
        scores: Prediction scores
        labels: True labels
        output_dir: Directory to save visualizations
    """
    
    def __init__(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        output_dir: Path
    ):
        self.scores = scores
        self.labels = labels
        self.output_dir = output_dir
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
    
    def plot_confusion_matrix(
        self,
        threshold: float = 0.5,
        filename: str = 'confusion_matrix.png'
    ):
        """
        Plot and save confusion matrix.
        
        Args:
            threshold: Classification threshold
            filename: Output filename
        """
        logger.info("Generating confusion matrix...")
        
        # Make predictions
        predictions = (self.scores >= threshold).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(self.labels, predictions)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.title(f'Confusion Matrix (threshold={threshold})', fontsize=16)
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {filepath}")
    
    def plot_roc_curve(self, filename: str = 'roc_curve.png'):
        """
        Plot and save ROC curve.
        
        Args:
            filename: Output filename
        """
        logger.info("Generating ROC curve...")
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(self.labels, self.scores)
        auc_roc = roc_auc_score(self.labels, self.scores)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(
            fpr, tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {auc_roc:.4f})'
        )
        plt.plot(
            [0, 1], [0, 1],
            color='navy',
            lw=2,
            linestyle='--',
            label='Random classifier'
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ROC curve to {filepath}")
    
    def plot_precision_recall_curve(self, filename: str = 'precision_recall_curve.png'):
        """
        Plot and save Precision-Recall curve.
        
        Args:
            filename: Output filename
        """
        from sklearn.metrics import precision_recall_curve
        
        logger.info("Generating Precision-Recall curve...")
        
        # Compute PR curve
        precision, recall, thresholds = precision_recall_curve(self.labels, self.scores)
        auc_pr = average_precision_score(self.labels, self.scores)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(
            recall, precision,
            color='darkorange',
            lw=2,
            label=f'PR curve (AP = {auc_pr:.4f})'
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=16)
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Precision-Recall curve to {filepath}")
    
    def plot_score_distribution(self, filename: str = 'score_distribution.png'):
        """
        Plot distribution of prediction scores for positive and negative samples.
        
        Args:
            filename: Output filename
        """
        logger.info("Generating score distribution plot...")
        
        # Separate scores by label
        pos_scores = self.scores[self.labels == 1]
        neg_scores = self.scores[self.labels == 0]
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(pos_scores, bins=50, alpha=0.7, color='green', label='Positive')
        plt.hist(neg_scores, bins=50, alpha=0.7, color='red', label='Negative')
        plt.xlabel('Prediction Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Score Distribution', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(
            [pos_scores, neg_scores],
            tick_labels=['Positive', 'Negative'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue')
        )
        plt.ylabel('Prediction Score', fontsize=12)
        plt.title('Score Distribution (Box Plot)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved score distribution to {filepath}")
    
    def generate_all_plots(self):
        """Generate all visualizations."""
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_score_distribution()


def save_results(
    metrics: Dict,
    output_dir: Path,
    model_info: Optional[Dict] = None
):
    """
    Save evaluation results to files.
    
    Args:
        metrics: Dictionary of computed metrics
        output_dir: Output directory
        model_info: Optional model information
    """
    # Save JSON results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        result_dict = {
            'metrics': metrics,
            'model_info': model_info or {}
        }
        json.dump(result_dict, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Save human-readable summary
    summary_path = output_dir / 'metrics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EVALUATION RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        if model_info:
            f.write("Model Information:\n")
            f.write("-" * 60 + "\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        f.write("Dataset Statistics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Test edges: {metrics['test_edges']:,}\n")
        f.write(f"Number of nodes: {metrics['num_nodes']:,}\n")
        f.write("\n")
        
        f.write("Classification Metrics:\n")
        f.write("-" * 60 + "\n")
        for key, value in metrics['classification'].items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Ranking Metrics:\n")
        f.write("-" * 60 + "\n")
        for key, value in metrics['ranking'].items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\n")
        
        f.write("=" * 60 + "\n")
    
    logger.info(f"Saved summary to {summary_path}")


def load_model(
    model_path: str,
    device: torch.device
) -> Tuple[DrugDiseaseModel, Dict]:
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model configuration from args
    args = checkpoint.get('args')
    
    if args is None:
        raise ValueError(
            "Checkpoint does not contain 'args'. "
            "Cannot reconstruct model architecture."
        )
    
    # Extract data info from checkpoint
    # The model state dict contains embedding dimensions
    state_dict = checkpoint['model_state_dict']
    num_nodes = state_dict['encoder.node_embeddings.weight'].size(0)
    num_relations = state_dict['decoder.relation_embeddings.weight'].size(0)
    
    logger.info(f"Model configuration:")
    logger.info(f"  - Nodes: {num_nodes:,}")
    logger.info(f"  - Relations: {num_relations}")
    logger.info(f"  - Embedding dim: {args.embedding_dim}")
    logger.info(f"  - Hidden dim: {args.hidden_dim}")
    logger.info(f"  - Dropout: {args.dropout}")
    
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
    
    # Load state dict
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully!")
    
    # Extract model info
    model_info = {
        'checkpoint_path': str(model_path),
        'epoch': checkpoint.get('epoch', 'unknown'),
        'num_nodes': num_nodes,
        'num_relations': num_relations,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }
    
    if 'best_val_loss' in checkpoint:
        model_info['best_val_loss'] = checkpoint['best_val_loss']
    if 'best_val_acc' in checkpoint:
        model_info['best_val_acc'] = checkpoint['best_val_acc']
    
    return model, model_info


def load_test_data(data_dir: str) -> Tuple[Dict, Dict]:
    """
    Load test data and full graph.
    
    Args:
        data_dir: Directory containing processed data
    
    Returns:
        Tuple of (test_data, full_graph)
    """
    data_path = Path(data_dir)
    
    logger.info("Loading test data...")
    test_data = torch.load(data_path / 'test_data.pt', weights_only=False)
    full_graph = torch.load(data_path / 'full_graph.pt', weights_only=False)
    
    # Filter invalid edges
    num_nodes = test_data['num_nodes']
    
    def filter_edges(data, name):
        edge_index = data['edge_index']
        edge_type = data['edge_type']
        valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
        
        invalid_count = (~valid_mask).sum().item()
        if invalid_count > 0:
            logger.warning(
                f"{name}: Filtered {invalid_count} invalid edges "
                f"({invalid_count / edge_index.size(1) * 100:.2f}%)"
            )
            data['edge_index'] = edge_index[:, valid_mask]
            data['edge_type'] = edge_type[valid_mask]
        
        return data
    
    test_data = filter_edges(test_data, "Test")
    full_graph = filter_edges(full_graph, "Full graph")
    
    logger.info(f"Test edges: {test_data['edge_index'].size(1):,}")
    logger.info(f"Nodes: {num_nodes:,}")
    logger.info(f"Relations: {test_data['num_relations']}")
    
    return test_data, full_graph


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained RGCN model for link prediction'
    )
    
    # Model arguments
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model checkpoint (e.g., output/models/best_model.pt)'
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
        default='results',
        help='Directory to save evaluation results and figures'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--num_neg_samples',
        type=int,
        default=1,
        help='Number of negative samples per positive sample'
    )
    parser.add_argument(
        '--k_values',
        type=int,
        nargs='+',
        default=[10, 50],
        help='K values for Hits@K metric (e.g., --k_values 10 50 100)'
    )
    
    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu). Auto-detects GPU if available.'
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")
    
    # Setup file logging
    file_handler = logging.FileHandler(output_dir / 'evaluation.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)
    
    # Load model
    model, model_info = load_model(args.model_path, device)
    
    # Load test data
    test_data, full_graph = load_test_data(args.data_dir)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_data=test_data,
        full_graph=full_graph,
        device=device,
        batch_size=args.batch_size
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(
        num_neg_samples=args.num_neg_samples,
        k_values=args.k_values
    )
    
    # Save results
    save_results(metrics, output_dir, model_info)
    
    # Generate visualizations
    visualizer = ResultsVisualizer(
        scores=evaluator.scores,
        labels=evaluator.labels,
        output_dir=output_dir
    )
    visualizer.generate_all_plots()
    
    logger.info("=" * 60)
    logger.info("Evaluation completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
