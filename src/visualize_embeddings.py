"""
Node Embedding Visualization for RGCN Model

This script extracts and visualizes learned node embeddings from a trained RGCN model,
providing insights into the learned representations through dimensionality reduction
and clustering analysis.

Features:
- Extract embeddings from trained model
- Dimensionality reduction (t-SNE, UMAP)
- Interactive and static visualizations
- Nearest neighbor analysis
- Clustering analysis by node type

Usage:
    python src/visualize_embeddings.py --model_path output/models/best_model.pt
    python src/visualize_embeddings.py --model_path output/models/best_model.pt --method umap
    python src/visualize_embeddings.py --model_path output/models/best_model.pt --query "Aspirin"
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# Add src to path
sys.path.append(str(Path(__file__).parent))

from models.rgcn import DrugDiseaseModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class EmbeddingVisualizer:
    """
    Visualize and analyze node embeddings from trained RGCN model.
    
    This class extracts embeddings, performs dimensionality reduction,
    and creates various visualizations to understand the learned representations.
    
    Attributes:
        model: Trained DrugDiseaseModel
        embeddings: Node embeddings (num_nodes, hidden_dim)
        embeddings_2d: 2D projected embeddings
        mappings: Node and relation mappings
        node_types: Dictionary mapping node idx to type
        device: Computation device
    """
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = 'data/processed',
        device: Optional[torch.device] = None
    ):
        """
        Initialize embedding visualizer.
        
        Args:
            model_path: Path to trained model checkpoint
            data_dir: Directory containing processed data
            device: Computation device (default: auto-detect)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_dir = Path(data_dir)
        
        logger.info(f"Loading data from {data_dir}...")
        self._load_data()
        
        logger.info(f"Loading model from {model_path}...")
        self._load_model(model_path)
        
        logger.info("Extracting embeddings...")
        self._extract_embeddings()
        
    def _load_data(self):
        """Load processed data and mappings."""
        # Load mappings
        mappings_path = self.data_dir / 'mappings.pt'
        self.mappings = torch.load(mappings_path, weights_only=False)
        
        # Load graph first to get number of nodes
        full_graph = torch.load(self.data_dir / 'full_graph.pt', weights_only=False)
        self.graph_data = full_graph
        
        # Create node type index
        # Note: Only include nodes that exist in the graph
        max_node_idx = self.graph_data['num_nodes'] - 1
        
        self.node_types = {}
        self.disease_indices = []
        self.drug_indices = []
        self.gene_indices = []
        
        for idx, (node_id, name, ntype) in self.mappings['idx2node'].items():
            if idx > max_node_idx:
                continue  # Skip nodes not in graph
            
            self.node_types[idx] = ntype
            if ntype == 'disease':
                self.disease_indices.append(idx)
            elif ntype == 'drug':
                self.drug_indices.append(idx)
            elif ntype == 'gene/protein':
                self.gene_indices.append(idx)
        
        logger.info(f"Loaded {len(self.node_types)} nodes (valid in graph)")
        logger.info(f"  Diseases: {len(self.disease_indices)}")
        logger.info(f"  Drugs: {len(self.drug_indices)}")
        logger.info(f"  Genes/Proteins: {len(self.gene_indices)}")
        
    def _load_model(self, model_path: str):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Extract model parameters
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        else:
            state_dict = checkpoint
            config = {}
        
        # Initialize model
        num_nodes = self.graph_data['num_nodes']
        num_relations = self.graph_data['num_relations']
        
        self.model = DrugDiseaseModel(
            num_nodes=num_nodes,
            num_relations=num_relations,
            embedding_dim=config.get('embedding_dim', 64),
            hidden_dim=config.get('hidden_dim', 128),
            dropout=config.get('dropout', 0.5),
            num_bases=config.get('num_bases', None)
        ).to(self.device)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.hidden_dim = config.get('hidden_dim', 128)
        logger.info(f"Model loaded successfully (hidden_dim={self.hidden_dim})")
        
    def _extract_embeddings(self):
        """Extract node embeddings from model."""
        edge_index = self.graph_data['edge_index'].to(self.device)
        edge_type = self.graph_data['edge_type'].to(self.device)
        
        with torch.no_grad():
            h = self.model.encoder(edge_index, edge_type)
        
        self.embeddings = h.cpu().numpy()
        logger.info(f"Extracted embeddings: {self.embeddings.shape}")
        
    def reduce_dimensions(
        self,
        method: str = 'tsne',
        n_components: int = 2,
        sample_size: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Reduce embeddings to 2D using t-SNE or UMAP.
        
        Args:
            method: 'tsne' or 'umap'
            n_components: Number of dimensions (default: 2)
            sample_size: Number of nodes to sample (None = all)
            random_state: Random seed
        """
        logger.info(f"Reducing dimensions with {method.upper()}...")
        
        # Sample if needed
        if sample_size and sample_size < len(self.embeddings):
            logger.info(f"Sampling {sample_size} nodes for visualization")
            np.random.seed(random_state)
            self.sample_indices = np.random.choice(
                len(self.embeddings), 
                size=sample_size, 
                replace=False
            )
            embeddings_to_reduce = self.embeddings[self.sample_indices]
        else:
            self.sample_indices = np.arange(len(self.embeddings))
            embeddings_to_reduce = self.embeddings
        
        if method.lower() == 'tsne':
            reducer = TSNE(
                n_components=n_components,
                random_state=random_state,
                perplexity=min(30, len(embeddings_to_reduce) - 1),
                max_iter=1000,
                verbose=1
            )
            self.embeddings_2d = reducer.fit_transform(embeddings_to_reduce)
            
        elif method.lower() == 'umap':
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=random_state,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='cosine',
                    verbose=True
                )
                self.embeddings_2d = reducer.fit_transform(embeddings_to_reduce)
            except ImportError:
                logger.error("UMAP not installed. Install with: pip install umap-learn")
                logger.info("Falling back to t-SNE")
                return self.reduce_dimensions(method='tsne', n_components=n_components,
                                            sample_size=sample_size, random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'")
        
        logger.info(f"Reduced to {n_components}D: {self.embeddings_2d.shape}")
        
    def plot_by_node_type(self, output_path: Path):
        """
        Create scatter plot colored by node type.
        
        Args:
            output_path: Path to save the plot
        """
        logger.info("Creating node type visualization...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Prepare data
        node_types_sampled = [self.node_types[idx] for idx in self.sample_indices]
        
        # Color map
        type_colors = {
            'drug': '#2ecc71',      # Green
            'disease': '#e74c3c',   # Red
            'gene/protein': '#3498db'  # Blue
        }
        
        # Plot each type
        for ntype, color in type_colors.items():
            mask = np.array([nt == ntype for nt in node_types_sampled])
            if mask.sum() > 0:
                ax.scatter(
                    self.embeddings_2d[mask, 0],
                    self.embeddings_2d[mask, 1],
                    c=color,
                    label=ntype.replace('gene/protein', 'gene'),
                    alpha=0.6,
                    s=20,
                    edgecolors='none'
                )
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title('Node Embeddings Colored by Type', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, markerscale=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved node type plot to {output_path}")
        
    def plot_interactive(self, output_path: Path, max_points: int = 5000):
        """
        Create interactive HTML plot with hover information.
        
        Args:
            output_path: Path to save the HTML file
            max_points: Maximum points to plot (for performance)
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logger.warning("Plotly not installed. Install with: pip install plotly")
            logger.info("Skipping interactive plot")
            return
        
        logger.info("Creating interactive visualization...")
        
        # Sample if too many points
        if len(self.sample_indices) > max_points:
            logger.info(f"Sampling {max_points} points for interactive plot")
            plot_indices = np.random.choice(
                len(self.sample_indices),
                size=max_points,
                replace=False
            )
        else:
            plot_indices = np.arange(len(self.sample_indices))
        
        # Prepare data
        x = self.embeddings_2d[plot_indices, 0]
        y = self.embeddings_2d[plot_indices, 1]
        
        node_names = []
        node_types_list = []
        node_ids = []
        
        for idx in self.sample_indices[plot_indices]:
            node_id, name, ntype = self.mappings['idx2node'][idx]
            node_names.append(name)
            node_types_list.append(ntype)
            node_ids.append(node_id)
        
        # Create figure
        fig = go.Figure()
        
        # Color map
        type_colors = {
            'drug': '#2ecc71',
            'disease': '#e74c3c',
            'gene/protein': '#3498db'
        }
        
        # Add trace for each type
        for ntype, color in type_colors.items():
            mask = np.array([nt == ntype for nt in node_types_list])
            if mask.sum() > 0:
                fig.add_trace(go.Scatter(
                    x=x[mask],
                    y=y[mask],
                    mode='markers',
                    name=ntype.replace('gene/protein', 'gene'),
                    marker=dict(
                        size=5,
                        color=color,
                        opacity=0.6,
                        line=dict(width=0)
                    ),
                    text=[node_names[i] for i in np.where(mask)[0]],
                    hovertemplate='<b>%{text}</b><br>' +
                                 f'Type: {ntype}<br>' +
                                 'X: %{x:.2f}<br>' +
                                 'Y: %{y:.2f}<br>' +
                                 '<extra></extra>',
                    customdata=[node_ids[i] for i in np.where(mask)[0]]
                ))
        
        fig.update_layout(
            title='Interactive Node Embeddings Visualization',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            hovermode='closest',
            width=1200,
            height=800,
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.write_html(output_path)
        logger.info(f"Saved interactive plot to {output_path}")
        
    def find_nearest_neighbors(
        self,
        query: str,
        k: int = 10,
        node_type: Optional[str] = None
    ) -> List[Tuple[int, str, str, float]]:
        """
        Find k nearest neighbors to a query node.
        
        Args:
            query: Node name to query
            k: Number of neighbors to return
            node_type: Filter by node type (optional)
            
        Returns:
            List of (idx, node_id, name, distance) tuples
        """
        # Find query node
        query_idx = None
        query_lower = query.lower()
        
        for idx, (node_id, name, ntype) in self.mappings['idx2node'].items():
            if name.lower() == query_lower or query_lower in name.lower():
                query_idx = idx
                break
        
        if query_idx is None:
            logger.error(f"Node '{query}' not found")
            return []
        
        query_name = self.mappings['idx2node'][query_idx][1]
        query_type = self.node_types[query_idx]
        logger.info(f"Found query node: {query_name} (type: {query_type})")
        
        # Get query embedding
        query_emb = self.embeddings[query_idx].reshape(1, -1)
        
        # Compute distances
        if node_type:
            # Filter by type
            if node_type == 'drug':
                indices = self.drug_indices
            elif node_type == 'disease':
                indices = self.disease_indices
            elif node_type in ['gene', 'gene/protein']:
                indices = self.gene_indices
            else:
                indices = list(range(len(self.embeddings)))
            
            candidate_embs = self.embeddings[indices]
            distances = cdist(query_emb, candidate_embs, metric='cosine')[0]
            
            # Get top-k (excluding self)
            sorted_idx = np.argsort(distances)
            neighbors = []
            
            for i in sorted_idx:
                if indices[i] == query_idx:
                    continue
                node_id, name, ntype = self.mappings['idx2node'][indices[i]]
                neighbors.append((indices[i], node_id, name, distances[i]))
                if len(neighbors) >= k:
                    break
        else:
            # All nodes
            distances = cdist(query_emb, self.embeddings, metric='cosine')[0]
            sorted_idx = np.argsort(distances)[1:k+1]  # Exclude self
            
            neighbors = []
            for i in sorted_idx:
                node_id, name, ntype = self.mappings['idx2node'][i]
                neighbors.append((i, node_id, name, distances[i]))
        
        return neighbors
    
    def visualize_nearest_neighbors(
        self,
        query: str,
        k: int = 10,
        output_path: Optional[Path] = None
    ):
        """
        Visualize nearest neighbors on 2D plot.
        
        Args:
            query: Node name to query
            k: Number of neighbors to show
            output_path: Path to save plot (optional)
        """
        neighbors = self.find_nearest_neighbors(query, k=k)
        
        if not neighbors:
            return
        
        # Find query in sample
        query_idx = None
        for idx, (node_id, name, ntype) in self.mappings['idx2node'].items():
            if query.lower() in name.lower():
                query_idx = idx
                break
        
        if query_idx not in self.sample_indices:
            logger.warning(f"Query node not in sampled data. Consider re-running without sampling.")
            return
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot all nodes (faded)
        node_types_sampled = [self.node_types[idx] for idx in self.sample_indices]
        ax.scatter(
            self.embeddings_2d[:, 0],
            self.embeddings_2d[:, 1],
            c='lightgray',
            alpha=0.3,
            s=10,
            edgecolors='none'
        )
        
        # Find positions in 2D space
        query_pos = self.embeddings_2d[self.sample_indices == query_idx][0]
        
        # Plot query
        ax.scatter(
            query_pos[0], query_pos[1],
            c='red', s=200, marker='*',
            edgecolors='darkred', linewidths=2,
            label='Query', zorder=5
        )
        
        # Plot neighbors
        neighbor_positions = []
        neighbor_names = []
        
        for neighbor_idx, node_id, name, dist in neighbors:
            if neighbor_idx in self.sample_indices:
                pos_idx = np.where(self.sample_indices == neighbor_idx)[0]
                if len(pos_idx) > 0:
                    pos = self.embeddings_2d[pos_idx[0]]
                    neighbor_positions.append(pos)
                    neighbor_names.append(f"{name[:30]} ({dist:.3f})")
        
        if neighbor_positions:
            neighbor_positions = np.array(neighbor_positions)
            ax.scatter(
                neighbor_positions[:, 0],
                neighbor_positions[:, 1],
                c='blue', s=100, alpha=0.7,
                edgecolors='darkblue', linewidths=1,
                label='Neighbors', zorder=4
            )
            
            # Add lines from query to neighbors
            for pos in neighbor_positions:
                ax.plot(
                    [query_pos[0], pos[0]],
                    [query_pos[1], pos[1]],
                    'b--', alpha=0.3, linewidth=1
                )
            
            # Add labels
            for pos, name in zip(neighbor_positions, neighbor_names):
                ax.annotate(
                    name, xy=pos, xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
                )
        
        # Add query label
        query_name = self.mappings['idx2node'][query_idx][1]
        ax.annotate(
            query_name, xy=query_pos, xytext=(5, 5),
            textcoords='offset points',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3)
        )
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(f'Nearest Neighbors to "{query_name}"', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, markerscale=1.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved nearest neighbors plot to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compute_distance_matrices(self, output_dir: Path):
        """
        Compute and visualize distance matrices between node types.
        
        Args:
            output_dir: Directory to save plots
        """
        logger.info("Computing distance matrices...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample nodes from each type for visualization
        sample_size = min(100, len(self.drug_indices), len(self.disease_indices))
        
        drug_sample = np.random.choice(self.drug_indices, size=sample_size, replace=False)
        disease_sample = np.random.choice(self.disease_indices, size=sample_size, replace=False)
        gene_sample = np.random.choice(
            self.gene_indices, 
            size=min(sample_size, len(self.gene_indices)), 
            replace=False
        )
        
        # Drug-Disease distances
        logger.info("Computing drug-disease distances...")
        drug_embs = self.embeddings[drug_sample]
        disease_embs = self.embeddings[disease_sample]
        
        drug_disease_dist = cdist(drug_embs, disease_embs, metric='cosine')
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(drug_disease_dist, cmap='viridis', aspect='auto')
        ax.set_xlabel('Disease Index', fontsize=12)
        ax.set_ylabel('Drug Index', fontsize=12)
        ax.set_title('Drug-Disease Distance Matrix (Cosine)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Distance')
        plt.tight_layout()
        plt.savefig(output_dir / 'drug_disease_distances.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Mean drug-disease distance: {drug_disease_dist.mean():.4f}")
        logger.info(f"Saved drug-disease distance matrix to {output_dir / 'drug_disease_distances.png'}")
        
        # Drug-Drug distances
        logger.info("Computing drug-drug distances...")
        drug_drug_dist = cdist(drug_embs, drug_embs, metric='cosine')
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(drug_drug_dist, cmap='viridis', aspect='auto')
        ax.set_xlabel('Drug Index', fontsize=12)
        ax.set_ylabel('Drug Index', fontsize=12)
        ax.set_title('Drug-Drug Distance Matrix (Cosine)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Distance')
        plt.tight_layout()
        plt.savefig(output_dir / 'drug_drug_distances.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Mean drug-drug distance: {drug_drug_dist.mean():.4f}")
        
        # Disease-Disease distances
        logger.info("Computing disease-disease distances...")
        disease_disease_dist = cdist(disease_embs, disease_embs, metric='cosine')
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(disease_disease_dist, cmap='viridis', aspect='auto')
        ax.set_xlabel('Disease Index', fontsize=12)
        ax.set_ylabel('Disease Index', fontsize=12)
        ax.set_title('Disease-Disease Distance Matrix (Cosine)', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Distance')
        plt.tight_layout()
        plt.savefig(output_dir / 'disease_disease_distances.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Mean disease-disease distance: {disease_disease_dist.mean():.4f}")
        
    def cluster_analysis(self, output_dir: Path, n_clusters: int = 10):
        """
        Perform clustering analysis on each node type.
        
        Args:
            output_dir: Directory to save results
            n_clusters: Number of clusters for each type
        """
        logger.info(f"Performing clustering analysis with {n_clusters} clusters...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for node_type, indices in [
            ('drug', self.drug_indices),
            ('disease', self.disease_indices),
            ('gene', self.gene_indices)
        ]:
            logger.info(f"\nClustering {node_type}s...")
            
            # Get embeddings for this type
            type_embs = self.embeddings[indices]
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(type_embs)
            
            # Compute silhouette score
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(type_embs, labels)
                logger.info(f"{node_type.capitalize()} silhouette score: {sil_score:.4f}")
            else:
                sil_score = 0.0
            
            results[node_type] = {
                'labels': labels,
                'silhouette': sil_score,
                'cluster_sizes': np.bincount(labels)
            }
            
            # Visualize clusters if we have 2D embeddings
            if hasattr(self, 'embeddings_2d'):
                # Find which sampled nodes are of this type
                type_mask = np.array([self.node_types[idx] == node_type.replace('gene', 'gene/protein') 
                                     for idx in self.sample_indices])
                
                if type_mask.sum() > 0:
                    fig, ax = plt.subplots(figsize=(14, 10))
                    
                    # Get 2D positions for this type
                    type_2d = self.embeddings_2d[type_mask]
                    
                    # Get cluster labels for sampled nodes
                    sampled_type_indices = self.sample_indices[type_mask]
                    # Map back to original indices
                    cluster_labels_sampled = []
                    for idx in sampled_type_indices:
                        pos = indices.index(idx) if idx in indices else -1
                        if pos >= 0:
                            cluster_labels_sampled.append(labels[pos])
                        else:
                            cluster_labels_sampled.append(-1)
                    
                    cluster_labels_sampled = np.array(cluster_labels_sampled)
                    
                    # Plot each cluster
                    scatter = ax.scatter(
                        type_2d[:, 0],
                        type_2d[:, 1],
                        c=cluster_labels_sampled,
                        cmap='tab10',
                        s=30,
                        alpha=0.6,
                        edgecolors='black',
                        linewidths=0.5
                    )
                    
                    ax.set_xlabel('Dimension 1', fontsize=12)
                    ax.set_ylabel('Dimension 2', fontsize=12)
                    ax.set_title(f'{node_type.capitalize()} Clusters (k={n_clusters}, silhouette={sil_score:.3f})',
                               fontsize=14, fontweight='bold')
                    plt.colorbar(scatter, ax=ax, label='Cluster')
                    ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / f'{node_type}_clusters.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Saved {node_type} clusters to {output_dir / f'{node_type}_clusters.png'}")
            
            # Save cluster examples
            with open(output_dir / f'{node_type}_cluster_examples.txt', 'w') as f:
                f.write(f"{node_type.upper()} CLUSTERING RESULTS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Number of clusters: {n_clusters}\n")
                f.write(f"Silhouette score: {sil_score:.4f}\n")
                f.write(f"Total {node_type}s: {len(indices)}\n\n")
                
                for cluster_id in range(n_clusters):
                    cluster_mask = labels == cluster_id
                    cluster_indices = np.array(indices)[cluster_mask]
                    
                    f.write(f"\nCluster {cluster_id} ({cluster_mask.sum()} {node_type}s):\n")
                    f.write("-" * 80 + "\n")
                    
                    # Show first 10 examples
                    for idx in cluster_indices[:10]:
                        node_id, name, ntype = self.mappings['idx2node'][idx]
                        f.write(f"  - {name}\n")
                    
                    if len(cluster_indices) > 10:
                        f.write(f"  ... and {len(cluster_indices) - 10} more\n")
        
        # Save summary
        with open(output_dir / 'clustering_summary.txt', 'w') as f:
            f.write("CLUSTERING ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            for node_type, result in results.items():
                f.write(f"{node_type.upper()}:\n")
                f.write(f"  Silhouette score: {result['silhouette']:.4f}\n")
                f.write(f"  Cluster sizes: {result['cluster_sizes'].tolist()}\n")
                f.write(f"  Mean cluster size: {result['cluster_sizes'].mean():.1f}\n")
                f.write(f"  Std cluster size: {result['cluster_sizes'].std():.1f}\n\n")
        
        logger.info(f"Saved clustering summary to {output_dir / 'clustering_summary.txt'}")
    
    def generate_report(self, output_dir: Path):
        """
        Generate comprehensive analysis report.
        
        Args:
            output_dir: Directory to save report
        """
        report_path = output_dir / 'embedding_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NODE EMBEDDING ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Model: {self.model.__class__.__name__}\n")
            f.write(f"Embedding dimension: {self.hidden_dim}\n")
            f.write(f"Total nodes: {len(self.embeddings)}\n\n")
            
            f.write("Node Type Distribution:\n")
            f.write(f"  Drugs: {len(self.drug_indices)}\n")
            f.write(f"  Diseases: {len(self.disease_indices)}\n")
            f.write(f"  Genes/Proteins: {len(self.gene_indices)}\n\n")
            
            # Embedding statistics
            f.write("Embedding Statistics:\n")
            f.write(f"  Mean: {self.embeddings.mean():.4f}\n")
            f.write(f"  Std: {self.embeddings.std():.4f}\n")
            f.write(f"  Min: {self.embeddings.min():.4f}\n")
            f.write(f"  Max: {self.embeddings.max():.4f}\n\n")
            
            # Norm statistics by type
            f.write("Embedding Norms by Type:\n")
            for node_type, indices in [
                ('Drug', self.drug_indices),
                ('Disease', self.disease_indices),
                ('Gene/Protein', self.gene_indices)
            ]:
                type_embs = self.embeddings[indices]
                norms = np.linalg.norm(type_embs, axis=1)
                f.write(f"  {node_type}:\n")
                f.write(f"    Mean norm: {norms.mean():.4f}\n")
                f.write(f"    Std norm: {norms.std():.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        logger.info(f"Saved analysis report to {report_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize and analyze node embeddings from trained RGCN model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic visualization
    python src/visualize_embeddings.py --model_path output/models/best_model.pt
    
    # Use UMAP instead of t-SNE
    python src/visualize_embeddings.py --model_path output/models/best_model.pt --method umap
    
    # Find nearest neighbors
    python src/visualize_embeddings.py --model_path output/models/best_model.pt --query "Aspirin"
    
    # Full analysis with clustering
    python src/visualize_embeddings.py --model_path output/models/best_model.pt --cluster --n_clusters 15
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='output/models/best_model.pt',
        help='Path to trained model checkpoint (default: output/models/best_model.pt)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data (default: data/processed)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/embeddings',
        help='Output directory for results (default: results/embeddings)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='tsne',
        choices=['tsne', 'umap'],
        help='Dimensionality reduction method (default: tsne)'
    )
    
    parser.add_argument(
        '--sample_size',
        type=int,
        default=None,
        help='Number of nodes to sample for visualization (default: all)'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Query node name for nearest neighbor analysis'
    )
    
    parser.add_argument(
        '--k_neighbors',
        type=int,
        default=10,
        help='Number of nearest neighbors to find (default: 10)'
    )
    
    parser.add_argument(
        '--cluster',
        action='store_true',
        help='Perform clustering analysis'
    )
    
    parser.add_argument(
        '--n_clusters',
        type=int,
        default=10,
        help='Number of clusters for clustering analysis (default: 10)'
    )
    
    parser.add_argument(
        '--skip_interactive',
        action='store_true',
        help='Skip interactive HTML plot generation'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)
    
    try:
        # Initialize visualizer
        visualizer = EmbeddingVisualizer(
            model_path=args.model_path,
            data_dir=args.data_dir
        )
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reduce dimensions
        visualizer.reduce_dimensions(
            method=args.method,
            sample_size=args.sample_size
        )
        
        # Generate visualizations
        logger.info("\n" + "=" * 80)
        logger.info("Generating visualizations...")
        logger.info("=" * 80)
        
        # Node type plot
        visualizer.plot_by_node_type(output_dir / 'node_types.png')
        
        # Interactive plot
        if not args.skip_interactive:
            visualizer.plot_interactive(output_dir / 'interactive_embeddings.html')
        
        # Nearest neighbor analysis
        if args.query:
            logger.info(f"\nFinding nearest neighbors for '{args.query}'...")
            neighbors = visualizer.find_nearest_neighbors(
                args.query,
                k=args.k_neighbors
            )
            
            if neighbors:
                logger.info(f"\nTop {args.k_neighbors} nearest neighbors:")
                for i, (idx, node_id, name, dist) in enumerate(neighbors, 1):
                    ntype = visualizer.node_types[idx]
                    logger.info(f"  {i}. {name} ({ntype}) - distance: {dist:.4f}")
                
                # Visualize
                visualizer.visualize_nearest_neighbors(
                    args.query,
                    k=args.k_neighbors,
                    output_path=output_dir / 'nearest_neighbors.png'
                )
        
        # Distance matrices
        logger.info("\nComputing distance matrices...")
        visualizer.compute_distance_matrices(output_dir)
        
        # Clustering analysis
        if args.cluster:
            logger.info("\nPerforming clustering analysis...")
            visualizer.cluster_analysis(output_dir, n_clusters=args.n_clusters)
        
        # Generate report
        visualizer.generate_report(output_dir)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("VISUALIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Generated files:")
        logger.info(f"  - node_types.png: Nodes colored by type")
        if not args.skip_interactive:
            logger.info(f"  - interactive_embeddings.html: Interactive visualization")
        if args.query:
            logger.info(f"  - nearest_neighbors.png: Nearest neighbor visualization")
        logger.info(f"  - drug_disease_distances.png: Drug-disease distance matrix")
        logger.info(f"  - drug_drug_distances.png: Drug-drug distance matrix")
        logger.info(f"  - disease_disease_distances.png: Disease-disease distance matrix")
        if args.cluster:
            logger.info(f"  - *_clusters.png: Clustering visualizations")
            logger.info(f"  - *_cluster_examples.txt: Cluster member lists")
        logger.info(f"  - embedding_analysis_report.txt: Summary report")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during visualization: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
