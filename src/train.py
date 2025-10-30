"""
Training Script for RGCN Drug-Disease Link Prediction

This script trains the DrugDiseaseModel on the preprocessed PrimeKG data
using negative sampling and binary cross-entropy loss.

Usage:
    python src/train.py --epochs 100 --lr 0.001 --batch_size 1024
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rgcn import DrugDiseaseModel


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class NegativeSampler:
    """
    Negative sampler for link prediction.
    
    Generates negative samples by corrupting either the head or tail entity
    of positive triples.
    
    Args:
        num_nodes (int): Total number of nodes in the graph
        num_neg_samples (int): Number of negative samples per positive sample
    """
    
    def __init__(self, num_nodes: int, num_neg_samples: int = 1):
        self.num_nodes = num_nodes
        self.num_neg_samples = num_neg_samples
    
    def sample(
        self,
        pos_head: torch.Tensor,
        pos_tail: torch.Tensor,
        pos_rel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate negative samples by corrupting positive triples.
        
        Args:
            pos_head (torch.Tensor): Positive head entities [batch_size]
            pos_tail (torch.Tensor): Positive tail entities [batch_size]
            pos_rel (torch.Tensor): Relation types [batch_size]
        
        Returns:
            Tuple of (neg_head, neg_tail, neg_rel) tensors
        """
        batch_size = pos_head.size(0)
        device = pos_head.device
        
        # Repeat positive samples for each negative sample
        neg_head = pos_head.repeat_interleave(self.num_neg_samples)
        neg_tail = pos_tail.repeat_interleave(self.num_neg_samples)
        neg_rel = pos_rel.repeat_interleave(self.num_neg_samples)
        
        # Randomly corrupt either head or tail
        total_neg = batch_size * self.num_neg_samples
        corrupt_head_mask = torch.rand(total_neg, device=device) < 0.5
        
        # Generate random entities
        random_entities = torch.randint(
            0, self.num_nodes, (total_neg,), device=device
        )
        
        # Apply corruption
        neg_head = torch.where(corrupt_head_mask, random_entities, neg_head)
        neg_tail = torch.where(~corrupt_head_mask, random_entities, neg_tail)
        
        return neg_head, neg_tail, neg_rel


class Trainer:
    """
    Trainer for RGCN link prediction model.
    
    Args:
        model (DrugDiseaseModel): The model to train
        train_data (dict): Training data dictionary
        val_data (dict): Validation data dictionary
        full_graph (dict): Full graph structure for message passing
        device (torch.device): Device to use for training
        args (argparse.Namespace): Training arguments
    """
    
    def __init__(
        self,
        model: DrugDiseaseModel,
        train_data: Dict,
        val_data: Dict,
        full_graph: Dict,
        device: torch.device,
        args: argparse.Namespace
    ):
        self.model = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.full_graph = full_graph
        self.device = device
        self.args = args
        
        # Move graph data to device
        self.train_edge_index = train_data['edge_index'].to(device)
        self.train_edge_type = train_data['edge_type'].to(device)
        self.val_edge_index = val_data['edge_index'].to(device)
        self.val_edge_type = val_data['edge_type'].to(device)
        self.full_edge_index = full_graph['edge_index'].to(device)
        self.full_edge_type = full_graph['edge_type'].to(device)
        
        # Setup optimizer and loss
        self.optimizer = self._setup_optimizer()
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Setup negative sampler
        self.neg_sampler = NegativeSampler(
            num_nodes=train_data['num_nodes'],
            num_neg_samples=args.num_neg_samples
        )
        
        # Tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Checkpoint directory
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with optional weight decay."""
        if self.args.optimizer == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
    
    def _create_batches(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch_size: int,
        shuffle: bool = True
    ) -> list:
        """Create batches from edge data."""
        num_edges = edge_index.size(1)
        indices = torch.randperm(num_edges) if shuffle else torch.arange(num_edges)
        
        batches = []
        for start_idx in range(0, num_edges, batch_size):
            end_idx = min(start_idx + batch_size, num_edges)
            batch_indices = indices[start_idx:end_idx]
            
            batch_head = edge_index[0, batch_indices]
            batch_tail = edge_index[1, batch_indices]
            batch_rel = edge_type[batch_indices]
            
            batches.append((batch_head, batch_tail, batch_rel))
        
        return batches
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Create batches
        batches = self._create_batches(
            self.train_edge_index,
            self.train_edge_type,
            self.args.batch_size,
            shuffle=True
        )
        
        # Training loop
        pbar = tqdm(batches, desc='Training', leave=False)
        for batch_head, batch_tail, batch_rel in pbar:
            # Generate negative samples
            neg_head, neg_tail, neg_rel = self.neg_sampler.sample(
                batch_head, batch_tail, batch_rel
            )
            
            # Combine positive and negative samples
            all_heads = torch.cat([batch_head, neg_head])
            all_tails = torch.cat([batch_tail, neg_tail])
            all_rels = torch.cat([batch_rel, neg_rel])
            
            # Create labels (1 for positive, 0 for negative)
            pos_labels = torch.ones(batch_head.size(0), device=self.device)
            neg_labels = torch.zeros(neg_head.size(0), device=self.device)
            labels = torch.cat([pos_labels, neg_labels])
            
            # Forward pass
            self.optimizer.zero_grad()
            scores = self.model(
                self.train_edge_index,
                self.train_edge_type,
                all_heads,
                all_tails,
                all_rels
            )
            
            # Compute loss
            loss = self.criterion(scores, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.grad_clip
                )
            
            self.optimizer.step()
            
            # Compute accuracy
            predictions = (torch.sigmoid(scores) > 0.5).float()
            correct = (predictions == labels).sum().item()
            
            # Update statistics
            total_loss += loss.item() * labels.size(0)
            total_correct += correct
            total_samples += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct / labels.size(0):.4f}'
            })
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Create batches
        batches = self._create_batches(
            self.val_edge_index,
            self.val_edge_type,
            self.args.batch_size,
            shuffle=False
        )
        
        # Validation loop
        for batch_head, batch_tail, batch_rel in tqdm(batches, desc='Validation', leave=False):
            # Generate negative samples
            neg_head, neg_tail, neg_rel = self.neg_sampler.sample(
                batch_head, batch_tail, batch_rel
            )
            
            # Combine positive and negative samples
            all_heads = torch.cat([batch_head, neg_head])
            all_tails = torch.cat([batch_tail, neg_tail])
            all_rels = torch.cat([batch_rel, neg_rel])
            
            # Create labels
            pos_labels = torch.ones(batch_head.size(0), device=self.device)
            neg_labels = torch.zeros(neg_head.size(0), device=self.device)
            labels = torch.cat([pos_labels, neg_labels])
            
            # Forward pass (use full graph for message passing)
            scores = self.model(
                self.full_edge_index,
                self.full_edge_type,
                all_heads,
                all_tails,
                all_rels
            )
            
            # Compute loss
            loss = self.criterion(scores, labels)
            
            # Compute accuracy
            predictions = (torch.sigmoid(scores) > 0.5).float()
            correct = (predictions == labels).sum().item()
            
            # Update statistics
            total_loss += loss.item() * labels.size(0)
            total_correct += correct
            total_samples += labels.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def save_checkpoint(
        self,
        epoch: int,
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """Save model checkpoint."""
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pt'
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'args': self.args
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training for {self.args.epochs} epochs")
        logger.info(f"Batch size: {self.args.batch_size}")
        logger.info(f"Learning rate: {self.args.lr}")
        logger.info(f"Negative samples: {self.args.num_neg_samples}")
        
        start_time = time.time()
        
        for epoch in range(1, self.args.epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}/{self.args.epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # Save checkpoint
            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            
            if epoch % self.args.save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.args.early_stopping > 0:
                if len(self.val_losses) > self.args.early_stopping:
                    recent_losses = self.val_losses[-self.args.early_stopping:]
                    if all(recent_losses[i] >= recent_losses[0] 
                           for i in range(len(recent_losses))):
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(epoch, filename='final_model.pt')


def load_data(data_dir: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load preprocessed data.
    
    Args:
        data_dir (str): Directory containing processed data
    
    Returns:
        Tuple of (train_data, val_data, full_graph, mappings)
    """
    data_path = Path(data_dir)
    
    logger.info("Loading data...")
    train_data = torch.load(data_path / 'train_data.pt')
    val_data = torch.load(data_path / 'val_data.pt')
    test_data = torch.load(data_path / 'test_data.pt')
    full_graph = torch.load(data_path / 'full_graph.pt')
    mappings = torch.load(data_path / 'mappings.pt')
    
    num_nodes = train_data['num_nodes']
    
    # Filter out invalid edges (indices >= num_nodes)
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
    
    train_data = filter_edges(train_data, "Train")
    val_data = filter_edges(val_data, "Val")
    test_data = filter_edges(test_data, "Test")
    full_graph = filter_edges(full_graph, "Full graph")
    
    logger.info(f"Train edges: {train_data['edge_index'].size(1):,}")
    logger.info(f"Val edges: {val_data['edge_index'].size(1):,}")
    logger.info(f"Test edges: {test_data['edge_index'].size(1):,}")
    logger.info(f"Nodes: {num_nodes:,}")
    logger.info(f"Relations: {train_data['num_relations']}")
    
    return train_data, val_data, full_graph, mappings


def create_model(
    num_nodes: int,
    num_relations: int,
    args: argparse.Namespace
) -> DrugDiseaseModel:
    """
    Create the model.
    
    Args:
        num_nodes (int): Number of nodes
        num_relations (int): Number of relations
        args (argparse.Namespace): Model arguments
    
    Returns:
        DrugDiseaseModel instance
    """
    logger.info("Creating model...")
    model = DrugDiseaseModel(
        num_nodes=num_nodes,
        num_relations=num_relations,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        decoder_dropout=args.decoder_dropout,
        num_bases=args.num_bases
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created with {num_params:,} parameters")
    
    return model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train RGCN model for drug-disease link prediction'
    )
    
    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    
    # Model arguments
    parser.add_argument(
        '--embedding_dim',
        type=int,
        default=64,
        help='Dimension of node embeddings'
    )
    parser.add_argument(
        '--hidden_dim',
        type=int,
        default=128,
        help='Dimension of hidden representations'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout rate for RGCN layers'
    )
    parser.add_argument(
        '--decoder_dropout',
        type=float,
        default=0.1,
        help='Dropout rate for decoder'
    )
    parser.add_argument(
        '--num_bases',
        type=int,
        default=None,
        help='Number of bases for RGCN (None for full parameterization)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='Batch size for training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay (L2 regularization)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'adamw'],
        help='Optimizer to use'
    )
    parser.add_argument(
        '--num_neg_samples',
        type=int,
        default=1,
        help='Number of negative samples per positive sample'
    )
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=1.0,
        help='Gradient clipping value (0 to disable)'
    )
    parser.add_argument(
        '--save_every',
        type=int,
        default=10,
        help='Save checkpoint every N epochs'
    )
    parser.add_argument(
        '--early_stopping',
        type=int,
        default=0,
        help='Early stopping patience (0 to disable)'
    )
    
    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for training (cuda/cpu). Auto-detects GPU if available.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device with detailed logging
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load data
    train_data, val_data, full_graph, mappings = load_data(args.data_dir)
    
    # Create model
    model = create_model(
        num_nodes=train_data['num_nodes'],
        num_relations=train_data['num_relations'],
        args=args
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        full_graph=full_graph,
        device=device,
        args=args
    )
    
    # Train
    trainer.train()
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
