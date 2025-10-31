"""
Path-Based Prediction Explanations for Drug-Disease Relationships

This script provides interpretable explanations for drug-disease predictions by:
- Finding connecting paths in the knowledge graph
- Ranking paths by importance
- Generating natural language explanations
- Visualizing key pathways

Usage:
    python src/explain_predictions.py --drug 'Metformin' --disease 'Type 2 Diabetes'
    python src/explain_predictions.py --drug 'Aspirin' --disease 'heart disease' --top_k 5
    python src/explain_predictions.py --compare --prediction_file results/predictions.csv
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from tqdm import tqdm

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


class PredictionExplainer:
    """
    Explain drug-disease predictions using graph paths.
    
    This class finds and analyzes paths connecting drugs and diseases,
    providing interpretable explanations for model predictions.
    
    Attributes:
        model: Trained DrugDiseaseModel
        mappings: Node and relation mappings
        graph: NetworkX graph for path finding
        embeddings: Node embeddings from model
        device: Computation device
    """
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = 'data/processed',
        device: Optional[torch.device] = None
    ):
        """
        Initialize prediction explainer.
        
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
        
        logger.info("Building graph for path finding...")
        self._build_graph()
        
        logger.info("Extracting embeddings...")
        self._extract_embeddings()
        
    def _load_data(self):
        """Load processed data and mappings."""
        # Load mappings
        mappings_path = self.data_dir / 'mappings.pt'
        self.mappings = torch.load(mappings_path, weights_only=False)
        
        # Load graph
        full_graph = torch.load(self.data_dir / 'full_graph.pt', weights_only=False)
        self.graph_data = full_graph
        
        # Create node type index
        max_node_idx = self.graph_data['num_nodes'] - 1
        self.node_types = {}
        self.disease_indices = []
        self.drug_indices = []
        self.gene_indices = []
        
        for idx, (node_id, name, ntype) in self.mappings['idx2node'].items():
            if idx > max_node_idx:
                continue
            self.node_types[idx] = ntype
            if ntype == 'disease':
                self.disease_indices.append(idx)
            elif ntype == 'drug':
                self.drug_indices.append(idx)
            elif ntype == 'gene/protein':
                self.gene_indices.append(idx)
        
        logger.info(f"Loaded {len(self.node_types)} nodes")
        logger.info(f"  Drugs: {len(self.drug_indices)}")
        logger.info(f"  Diseases: {len(self.disease_indices)}")
        logger.info(f"  Genes/Proteins: {len(self.gene_indices)}")
        
    def _load_model(self, model_path: str):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
        else:
            state_dict = checkpoint
            config = {}
        
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
        
        logger.info("Model loaded successfully")
        
    def _build_graph(self):
        """Build NetworkX graph for path finding."""
        self.nx_graph = nx.DiGraph()
        
        edge_index = self.graph_data['edge_index']
        edge_types = self.graph_data['edge_type']
        
        # Add all edges with attributes
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            rel = edge_types[i].item()
            
            self.nx_graph.add_edge(
                src, dst,
                relation=rel,
                relation_name=self.mappings['idx2relation'][rel]
            )
        
        logger.info(f"Built graph: {self.nx_graph.number_of_nodes()} nodes, "
                   f"{self.nx_graph.number_of_edges()} edges")
        
    def _extract_embeddings(self):
        """Extract node embeddings from model."""
        edge_index = self.graph_data['edge_index'].to(self.device)
        edge_type = self.graph_data['edge_type'].to(self.device)
        
        with torch.no_grad():
            h = self.model.encoder(edge_index, edge_type)
        
        self.embeddings = h.cpu().numpy()
        logger.info(f"Extracted embeddings: {self.embeddings.shape}")
        
    def find_node(self, query: str, node_type: Optional[str] = None) -> Optional[int]:
        """
        Find node by name (case-insensitive partial match).
        
        Args:
            query: Node name to search for
            node_type: Filter by type (drug, disease, gene/protein)
            
        Returns:
            Node index if found, None otherwise
        """
        query_lower = query.lower()
        
        # Filter indices by type
        if node_type == 'drug':
            search_indices = self.drug_indices
        elif node_type == 'disease':
            search_indices = self.disease_indices
        elif node_type in ['gene', 'gene/protein']:
            search_indices = self.gene_indices
        else:
            search_indices = list(self.node_types.keys())
        
        # Try exact match first
        for idx in search_indices:
            _, name, _ = self.mappings['idx2node'][idx]
            if name.lower() == query_lower:
                return idx
        
        # Try partial match
        matches = []
        for idx in search_indices:
            _, name, _ = self.mappings['idx2node'][idx]
            if query_lower in name.lower():
                matches.append((idx, name))
        
        if len(matches) == 1:
            logger.info(f"Found: {matches[0][1]}")
            return matches[0][0]
        elif len(matches) > 1:
            logger.warning(f"Multiple matches for '{query}':")
            for idx, name in matches[:5]:
                logger.warning(f"  - {name}")
            logger.info(f"Using first match: {matches[0][1]}")
            return matches[0][0]
        
        return None
    
    def compute_prediction_score(self, drug_idx: int, disease_idx: int) -> float:
        """
        Compute prediction score for drug-disease pair.
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            
        Returns:
            Prediction score (0-1)
        """
        drug_emb = torch.tensor(self.embeddings[drug_idx], device=self.device)
        disease_emb = torch.tensor(self.embeddings[disease_idx], device=self.device)
        
        # Use cosine similarity as proxy
        similarity = F.cosine_similarity(drug_emb.unsqueeze(0), disease_emb.unsqueeze(0))
        score = (similarity.item() + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        return score
    
    def find_paths(
        self,
        drug_idx: int,
        disease_idx: int,
        max_paths: int = 20,
        max_length: int = 4
    ) -> List[List[int]]:
        """
        Find all paths connecting drug and disease.
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            max_paths: Maximum number of paths to return
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is list of node indices)
        """
        try:
            # Use generator and limit collection to avoid memory issues
            paths = []
            path_generator = nx.all_simple_paths(
                self.nx_graph,
                source=drug_idx,
                target=disease_idx,
                cutoff=max_length
            )
            
            # Collect paths with early stopping
            for path in path_generator:
                paths.append(path)
                if len(paths) >= max_paths * 5:  # Collect more than needed
                    break
            
            # Sort by length (prefer shorter paths)
            paths.sort(key=len)
            return paths[:max_paths]
            
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def score_path(self, path: List[int]) -> float:
        """
        Score a path based on embedding similarities along the path.
        
        Args:
            path: List of node indices forming a path
            
        Returns:
            Path score (higher is better)
        """
        if len(path) < 2:
            return 0.0
        
        # Compute average cosine similarity between consecutive nodes
        similarities = []
        for i in range(len(path) - 1):
            emb1 = self.embeddings[path[i]]
            emb2 = self.embeddings[path[i + 1]]
            
            # Cosine similarity
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(sim)
        
        # Average similarity, penalize by length
        avg_sim = np.mean(similarities)
        length_penalty = 1.0 / (1.0 + 0.2 * (len(path) - 2))
        
        return avg_sim * length_penalty
    
    def rank_paths(
        self,
        paths: List[List[int]],
        top_k: int = 5
    ) -> List[Tuple[List[int], float]]:
        """
        Rank paths by importance.
        
        Args:
            paths: List of paths
            top_k: Number of top paths to return
            
        Returns:
            List of (path, score) tuples
        """
        if not paths:
            return []
        
        scored_paths = []
        for path in paths:
            score = self.score_path(path)
            scored_paths.append((path, score))
        
        # Sort by score (descending)
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        
        return scored_paths[:top_k]
    
    def get_path_details(self, path: List[int]) -> Dict:
        """
        Extract detailed information about a path.
        
        Args:
            path: List of node indices
            
        Returns:
            Dictionary with path details
        """
        nodes = []
        relations = []
        
        for i, node_idx in enumerate(path):
            node_id, name, ntype = self.mappings['idx2node'][node_idx]
            nodes.append({
                'idx': node_idx,
                'id': node_id,
                'name': name,
                'type': ntype
            })
            
            # Get edge relation
            if i < len(path) - 1:
                next_idx = path[i + 1]
                edge_data = self.nx_graph.get_edge_data(node_idx, next_idx)
                if edge_data:
                    rel_name = edge_data.get('relation_name', 'unknown')
                    relations.append(rel_name)
                else:
                    relations.append('unknown')
        
        return {
            'path': path,
            'length': len(path) - 1,
            'nodes': nodes,
            'relations': relations
        }
    
    def generate_natural_language_explanation(
        self,
        drug_idx: int,
        disease_idx: int,
        path: List[int],
        score: float
    ) -> str:
        """
        Generate natural language explanation for a path.
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            path: List of node indices
            score: Path importance score
            
        Returns:
            Natural language explanation string
        """
        path_details = self.get_path_details(path)
        nodes = path_details['nodes']
        relations = path_details['relations']
        
        drug_name = nodes[0]['name']
        disease_name = nodes[-1]['name']
        
        # Build explanation based on path length
        if len(path) == 2:
            # Direct connection (shouldn't happen in PrimeKG for drug-disease)
            explanation = (
                f"{drug_name} may treat {disease_name} through a direct relationship "
                f"in the knowledge graph."
            )
        elif len(path) == 3:
            # Drug -> Gene -> Disease
            gene = nodes[1]
            explanation = (
                f"{drug_name} may treat {disease_name} because it {relations[0].replace('-', ' ')} "
                f"{gene['name']}, which {relations[1].replace('-', ' ')} {disease_name}."
            )
        elif len(path) == 4:
            # Drug -> Gene1 -> Gene2 -> Disease
            gene1 = nodes[1]
            gene2 = nodes[2]
            explanation = (
                f"{drug_name} may treat {disease_name} through the following mechanism: "
                f"the drug {relations[0].replace('-', ' ')} {gene1['name']}, "
                f"which {relations[1].replace('-', ' ')} {gene2['name']}, "
                f"and {gene2['name']} {relations[2].replace('-', ' ')} {disease_name}."
            )
        else:
            # Longer path - summarize intermediate steps
            intermediate_genes = [n['name'] for n in nodes[1:-1] if n['type'] == 'gene/protein']
            if intermediate_genes:
                gene_list = ', '.join(intermediate_genes[:3])
                if len(intermediate_genes) > 3:
                    gene_list += f" and {len(intermediate_genes) - 3} other genes"
                
                explanation = (
                    f"{drug_name} may treat {disease_name} through a pathway involving "
                    f"{gene_list}. This connection suggests a {len(path)-1}-step mechanism "
                    f"linking the drug's molecular targets to the disease pathology."
                )
            else:
                explanation = (
                    f"{drug_name} may treat {disease_name} through a complex {len(path)-1}-step "
                    f"pathway in the knowledge graph."
                )
        
        return explanation
    
    def explain_prediction(
        self,
        drug: str,
        disease: str,
        top_k: int = 5,
        output_dir: Optional[Path] = None
    ) -> Dict:
        """
        Generate complete explanation for a drug-disease prediction.
        
        Args:
            drug: Drug name
            disease: Disease name
            top_k: Number of top paths to explain
            output_dir: Output directory for visualizations
            
        Returns:
            Dictionary with explanation details
        """
        logger.info("=" * 80)
        logger.info(f"Explaining prediction: {drug} → {disease}")
        logger.info("=" * 80)
        
        # Find nodes
        drug_idx = self.find_node(drug, node_type='drug')
        if drug_idx is None:
            logger.error(f"Drug '{drug}' not found")
            return {'error': f"Drug '{drug}' not found"}
        
        disease_idx = self.find_node(disease, node_type='disease')
        if disease_idx is None:
            logger.error(f"Disease '{disease}' not found")
            return {'error': f"Disease '{disease}' not found"}
        
        drug_name = self.mappings['idx2node'][drug_idx][1]
        disease_name = self.mappings['idx2node'][disease_idx][1]
        
        logger.info(f"Drug: {drug_name}")
        logger.info(f"Disease: {disease_name}")
        
        # Compute prediction score
        pred_score = self.compute_prediction_score(drug_idx, disease_idx)
        logger.info(f"Prediction score: {pred_score:.4f}")
        
        # Find paths
        logger.info("\nFinding connecting paths...")
        paths = self.find_paths(drug_idx, disease_idx, max_paths=20, max_length=4)
        logger.info(f"Found {len(paths)} paths")
        
        if not paths:
            logger.warning("No paths found connecting drug and disease")
            result = {
                'drug': drug_name,
                'disease': disease_name,
                'prediction_score': pred_score,
                'num_paths': 0,
                'top_paths': []
            }
            
            # Still generate report for no-path case
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                safe_name = f"{drug_name}_{disease_name}".replace('/', '_').replace(' ', '_')[:50]
                report_path = output_dir / f'explanation_{safe_name}.txt'
                self.generate_explanation_report(result, report_path)
            
            return result
        
        # Rank paths
        logger.info("Ranking paths by importance...")
        ranked_paths = self.rank_paths(paths, top_k=top_k)
        
        # Generate explanations
        explanations = []
        for i, (path, path_score) in enumerate(ranked_paths, 1):
            path_details = self.get_path_details(path)
            nl_explanation = self.generate_natural_language_explanation(
                drug_idx, disease_idx, path, path_score
            )
            
            explanations.append({
                'rank': i,
                'path': path,
                'path_score': path_score,
                'details': path_details,
                'explanation': nl_explanation
            })
        
        result = {
            'drug': drug_name,
            'drug_idx': drug_idx,
            'disease': disease_name,
            'disease_idx': disease_idx,
            'prediction_score': pred_score,
            'num_paths': len(paths),
            'top_paths': explanations
        }
        
        # Generate visualizations if output directory provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.visualize_explanation(result, output_dir)
        
        return result
    
    def visualize_explanation(self, explanation: Dict, output_dir: Path):
        """
        Create visualizations for explanation.
        
        Args:
            explanation: Explanation dictionary
            output_dir: Output directory
        """
        drug_name = explanation['drug']
        disease_name = explanation['disease']
        
        # 1. Network visualization of top paths
        self._visualize_paths_network(explanation, output_dir)
        
        # 2. Path comparison bar chart
        self._visualize_path_scores(explanation, output_dir)
        
        # 3. Sankey diagram (if plotly available)
        try:
            self._visualize_sankey(explanation, output_dir)
        except ImportError:
            logger.warning("Plotly not available, skipping Sankey diagram")
    
    def _visualize_paths_network(self, explanation: Dict, output_dir: Path):
        """Visualize paths as network graph."""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create subgraph with all top paths
        G = nx.DiGraph()
        
        drug_name = explanation['drug']
        disease_name = explanation['disease']
        
        # Add all edges from top paths
        edge_importance = defaultdict(float)
        
        for path_info in explanation['top_paths']:
            nodes = path_info['details']['nodes']
            relations = path_info['details']['relations']
            path_score = path_info['path_score']
            
            for i in range(len(nodes) - 1):
                src_name = nodes[i]['name']
                dst_name = nodes[i + 1]['name']
                rel = relations[i]
                
                G.add_edge(src_name, dst_name, relation=rel)
                edge_importance[(src_name, dst_name)] += path_score
        
        # Normalize edge importance
        max_importance = max(edge_importance.values()) if edge_importance else 1.0
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # Draw nodes by type
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node == drug_name:
                node_colors.append('#2ecc71')  # Green for drug
                node_sizes.append(4000)
            elif node == disease_name:
                node_colors.append('#e74c3c')  # Red for disease
                node_sizes.append(4000)
            else:
                node_colors.append('#3498db')  # Blue for genes
                node_sizes.append(2500)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.9, ax=ax)
        
        # Draw edges with varying thickness based on importance
        for (u, v) in G.edges():
            importance = edge_importance.get((u, v), 0.0) / max_importance
            width = 1 + 5 * importance
            alpha = 0.3 + 0.7 * importance
            
            nx.draw_networkx_edges(
                G, pos, [(u, v)],
                width=width,
                alpha=alpha,
                edge_color='gray',
                arrows=True,
                arrowsize=20,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax
            )
        
        # Draw labels
        labels = {node: node[:30] + '...' if len(node) > 30 else node
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9,
                               font_weight='bold', ax=ax)
        
        # Draw edge labels (relations)
        edge_labels = {}
        for (u, v), data in G.edges.items():
            importance = edge_importance.get((u, v), 0.0) / max_importance
            if importance > 0.5:  # Only show labels for important edges
                edge_labels[(u, v)] = data['relation']
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)
        
        ax.set_title(
            f"Top Pathways: {drug_name} → {disease_name}\n"
            f"Prediction Score: {explanation['prediction_score']:.3f} | "
            f"Total Paths: {explanation['num_paths']}",
            fontsize=14, fontweight='bold', pad=20
        )
        ax.axis('off')
        
        plt.tight_layout()
        safe_name = f"{drug_name}_{disease_name}".replace('/', '_').replace(' ', '_')[:50]
        plt.savefig(output_dir / f'paths_network_{safe_name}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved network visualization to {output_dir / f'paths_network_{safe_name}.png'}")
    
    def _visualize_path_scores(self, explanation: Dict, output_dir: Path):
        """Visualize path importance scores."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ranks = [p['rank'] for p in explanation['top_paths']]
        scores = [p['path_score'] for p in explanation['top_paths']]
        lengths = [p['details']['length'] for p in explanation['top_paths']]
        
        # Color by path length
        colors = plt.cm.viridis(np.array(lengths) / max(lengths))
        
        bars = ax.bar(ranks, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Path Rank', fontsize=12)
        ax.set_ylabel('Path Importance Score', fontsize=12)
        ax.set_title(
            f"Path Importance Ranking\n{explanation['drug']} → {explanation['disease']}",
            fontsize=14, fontweight='bold'
        )
        ax.set_xticks(ranks)
        ax.grid(axis='y', alpha=0.3)
        
        # Add colorbar for path length
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(vmin=min(lengths), vmax=max(lengths)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Path Length (# steps)', rotation=270, labelpad=20)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        safe_name = f"{explanation['drug']}_{explanation['disease']}".replace('/', '_').replace(' ', '_')[:50]
        plt.savefig(output_dir / f'path_scores_{safe_name}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved path scores to {output_dir / f'path_scores_{safe_name}.png'}")
    
    def _visualize_sankey(self, explanation: Dict, output_dir: Path):
        """Create Sankey diagram showing all paths."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            return
        
        # Prepare data for Sankey
        node_labels = []
        node_indices = {}
        
        # Collect all unique nodes from top paths
        for path_info in explanation['top_paths']:
            for node in path_info['details']['nodes']:
                if node['name'] not in node_indices:
                    node_indices[node['name']] = len(node_labels)
                    node_labels.append(node['name'])
        
        # Build links
        sources = []
        targets = []
        values = []
        link_labels = []
        
        for path_info in explanation['top_paths']:
            nodes = path_info['details']['nodes']
            relations = path_info['details']['relations']
            path_score = path_info['path_score']
            
            for i in range(len(nodes) - 1):
                src_idx = node_indices[nodes[i]['name']]
                dst_idx = node_indices[nodes[i + 1]['name']]
                
                sources.append(src_idx)
                targets.append(dst_idx)
                values.append(path_score * 10)  # Scale for visibility
                link_labels.append(relations[i])
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color='black', width=0.5),
                label=node_labels,
                color='lightblue'
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                label=link_labels
            )
        )])
        
        fig.update_layout(
            title=f"Pathway Flow: {explanation['drug']} → {explanation['disease']}",
            font_size=10,
            height=600
        )
        
        safe_name = f"{explanation['drug']}_{explanation['disease']}".replace('/', '_').replace(' ', '_')[:50]
        fig.write_html(output_dir / f'sankey_{safe_name}.html')
        
        logger.info(f"Saved Sankey diagram to {output_dir / f'sankey_{safe_name}.html'}")
    
    def generate_explanation_report(
        self,
        explanation: Dict,
        output_path: Path
    ):
        """
        Generate text report with detailed explanation.
        
        Args:
            explanation: Explanation dictionary
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DRUG-DISEASE PREDICTION EXPLANATION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Drug: {explanation['drug']}\n")
            f.write(f"Disease: {explanation['disease']}\n")
            f.write(f"Prediction Score: {explanation['prediction_score']:.4f}\n")
            f.write(f"Total Paths Found: {explanation['num_paths']}\n")
            f.write(f"Top Paths Analyzed: {len(explanation['top_paths'])}\n")
            f.write("\n")
            
            # Summary
            f.write("-" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n\n")
            
            if explanation['prediction_score'] >= 0.7:
                confidence = "high"
            elif explanation['prediction_score'] >= 0.5:
                confidence = "moderate"
            else:
                confidence = "low"
            
            f.write(f"The model predicts a {confidence} confidence relationship between "
                   f"{explanation['drug']} and {explanation['disease']}.\n\n")
            
            if explanation['num_paths'] > 0:
                f.write(f"This prediction is supported by {explanation['num_paths']} "
                       f"connecting pathway(s) in the knowledge graph.\n\n")
            else:
                f.write("Warning: No direct paths found in the knowledge graph. "
                       "The prediction is based solely on embedding similarity.\n\n")
            
            # Top paths with explanations
            f.write("=" * 80 + "\n")
            f.write("TOP PATHWAYS AND EXPLANATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            for path_info in explanation['top_paths']:
                f.write(f"Pathway #{path_info['rank']} (Importance: {path_info['path_score']:.4f})\n")
                f.write("-" * 80 + "\n")
                
                # Visual path representation
                nodes = path_info['details']['nodes']
                relations = path_info['details']['relations']
                
                f.write("Path: ")
                for i, node in enumerate(nodes):
                    f.write(node['name'])
                    if i < len(relations):
                        f.write(f" →[{relations[i]}]→ ")
                f.write("\n\n")
                
                # Detailed node information
                f.write("Detailed Steps:\n")
                for i, node in enumerate(nodes):
                    f.write(f"  {i+1}. {node['name']} ({node['type']})\n")
                    if i < len(relations):
                        f.write(f"     └─ {relations[i]} ─→\n")
                f.write("\n")
                
                # Natural language explanation
                f.write("Explanation:\n")
                f.write(f"  {path_info['explanation']}\n")
                f.write("\n\n")
            
            # Biological interpretation
            f.write("=" * 80 + "\n")
            f.write("BIOLOGICAL INTERPRETATION\n")
            f.write("=" * 80 + "\n\n")
            
            # Extract common genes across paths
            all_genes = []
            for path_info in explanation['top_paths']:
                genes = [n['name'] for n in path_info['details']['nodes']
                        if n['type'] == 'gene/protein']
                all_genes.extend(genes)
            
            if all_genes:
                gene_counts = Counter(all_genes)
                f.write("Key Genes/Proteins Involved:\n")
                for gene, count in gene_counts.most_common(10):
                    f.write(f"  • {gene}: appears in {count} pathway(s)\n")
                f.write("\n")
                
                f.write("These genes/proteins represent potential molecular mechanisms "
                       f"through which {explanation['drug']} may affect {explanation['disease']}.\n\n")
            
            # Recommendations
            f.write("=" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            if explanation['prediction_score'] >= 0.7 and explanation['num_paths'] >= 3:
                f.write("✓ Strong prediction with multiple supporting pathways\n")
                f.write("  → Recommended for further investigation\n")
                f.write("  → Review literature for validation\n")
                f.write("  → Consider for experimental studies\n")
            elif explanation['prediction_score'] >= 0.5:
                f.write("⚠ Moderate prediction\n")
                f.write("  → Requires additional validation\n")
                f.write("  → Check for clinical evidence\n")
                f.write("  → Investigate key genes for biological plausibility\n")
            else:
                f.write("⚠ Low confidence prediction\n")
                f.write("  → Weak or no supporting evidence\n")
                f.write("  → Not recommended without additional data\n")
                f.write("  → May indicate novel mechanism or false positive\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF EXPLANATION\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Saved explanation report to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate interpretable explanations for drug-disease predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Explain a specific drug-disease prediction
    python src/explain_predictions.py --drug 'Metformin' --disease 'Type 2 Diabetes'
    
    # Get more pathways
    python src/explain_predictions.py --drug 'Aspirin' --disease 'heart disease' --top_k 10
    
    # Use specific model
    python src/explain_predictions.py --drug 'Insulin' --disease 'diabetes' \\
        --model_path output/models/final_model.pt
        """
    )
    
    parser.add_argument(
        '--drug',
        type=str,
        required=True,
        help='Drug name to analyze'
    )
    
    parser.add_argument(
        '--disease',
        type=str,
        required=True,
        help='Disease name to analyze'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Number of top paths to explain (default: 5)'
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
        default='results/explanations',
        help='Output directory for results (default: results/explanations)'
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
        # Initialize explainer
        explainer = PredictionExplainer(
            model_path=args.model_path,
            data_dir=args.data_dir
        )
        
        # Generate explanation
        output_dir = Path(args.output_dir)
        explanation = explainer.explain_prediction(
            drug=args.drug,
            disease=args.disease,
            top_k=args.top_k,
            output_dir=output_dir
        )
        
        # Check for errors
        if 'error' in explanation:
            logger.error(explanation['error'])
            sys.exit(1)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EXPLANATION SUMMARY")
        logger.info("=" * 80)
        
        for i, path_info in enumerate(explanation['top_paths'], 1):
            logger.info(f"\nPath {i}: {path_info['explanation']}")
        
        # Generate report
        safe_name = f"{explanation['drug']}_{explanation['disease']}".replace('/', '_').replace(' ', '_')[:50]
        report_path = output_dir / f'explanation_{safe_name}.txt'
        explainer.generate_explanation_report(explanation, report_path)
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPLANATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Prediction score: {explanation['prediction_score']:.4f}")
        logger.info(f"Paths found: {explanation['num_paths']}")
        logger.info(f"Top paths explained: {len(explanation['top_paths'])}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during explanation: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
