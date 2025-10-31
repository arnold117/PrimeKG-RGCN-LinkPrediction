"""
Failure Analysis for Drug-Disease Predictions

This script performs deep analysis of prediction failures to understand:
- Why the model makes incorrect predictions
- What patterns lead to errors
- Structural differences between correct and incorrect predictions
- Potential improvements to the model

Usage:
    python src/analyze_failures.py --model_path output/models/best_model.pt
    python src/analyze_failures.py --num_failures 5 --num_successes 5
    python src/analyze_failures.py --visualize_subgraphs --output_dir results/failure_analysis
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
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
import torch.nn.functional as F
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

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class FailureAnalyzer:
    """
    Analyze prediction failures to understand model limitations.
    
    This class identifies worst predictions, analyzes their graph structure,
    compares with correct predictions, and generates hypotheses about failure modes.
    """
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = 'data/processed',
        device: Optional[torch.device] = None
    ):
        """
        Initialize failure analyzer.
        
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
        
        logger.info("Building graph structure...")
        self._build_graph()
    
    def _load_data(self):
        """Load processed data and mappings."""
        # Load mappings
        mappings_path = self.data_dir / 'mappings.pt'
        self.mappings = torch.load(mappings_path, weights_only=False)
        
        # Load graph
        full_graph = torch.load(self.data_dir / 'full_graph.pt', weights_only=False)
        self.graph_data = full_graph
        
        # Load splits
        train_data = torch.load(self.data_dir / 'train_data.pt', weights_only=False)
        test_data = torch.load(self.data_dir / 'test_data.pt', weights_only=False)
        
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
        
        self.train_data = train_data
        self.test_data = test_data
        
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
    
    def _extract_embeddings(self):
        """Extract node embeddings from model."""
        edge_index = self.graph_data['edge_index'].to(self.device)
        edge_type = self.graph_data['edge_type'].to(self.device)
        
        with torch.no_grad():
            h = self.model.encoder(edge_index, edge_type)
        
        self.embeddings = h.cpu().numpy()
        logger.info(f"Extracted embeddings: {self.embeddings.shape}")
    
    def _build_graph(self):
        """Build NetworkX graph for analysis."""
        self.nx_graph = nx.DiGraph()
        edge_index = self.graph_data['edge_index']
        edge_types = self.graph_data['edge_type']
        
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
    
    def compute_prediction_score(self, drug_idx: int, disease_idx: int) -> float:
        """Compute prediction score for drug-disease pair."""
        drug_emb = self.embeddings[drug_idx]
        disease_emb = self.embeddings[disease_idx]
        
        # Cosine similarity
        similarity = np.dot(drug_emb, disease_emb) / (
            np.linalg.norm(drug_emb) * np.linalg.norm(disease_emb)
        )
        
        return (similarity + 1) / 2  # Scale to [0, 1]
    
    def get_ground_truth_labels(self, num_samples: int = 5000) -> List[Tuple[int, int, int]]:
        """
        Generate ground truth labels for drug-disease pairs.
        
        Since PrimeKG doesn't have direct drug-disease edges, we create proxy labels:
        - Positive: drug and disease are connected through short paths
        - Negative: random pairs with no or very long paths
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            List of (drug_idx, disease_idx, label) tuples
        """
        logger.info("Generating proxy ground truth labels...")
        
        samples = []
        
        # Get genes that are actually in the graph
        genes_in_graph = [g for g in self.gene_indices if g in self.nx_graph]
        
        if not genes_in_graph:
            logger.warning("No genes found in graph, using random pairs only")
            # Generate all as random pairs
            for _ in range(num_samples):
                drug_idx = np.random.choice(self.drug_indices)
                disease_idx = np.random.choice(self.disease_indices)
                samples.append((drug_idx, disease_idx, 0))
            logger.info(f"Generated {len(samples)} labeled samples")
            return samples
        
        # Generate positive samples (connected pairs)
        attempts = 0
        max_attempts = num_samples * 10  # Avoid infinite loop
        
        while len([s for s in samples if s[2] == 1]) < num_samples // 2 and attempts < max_attempts:
            attempts += 1
            
            # Sample a gene that exists in the graph
            gene_idx = np.random.choice(genes_in_graph)
            
            # Find drugs connected to this gene
            try:
                drug_neighbors = [n for n in self.nx_graph.predecessors(gene_idx)
                                if self.node_types.get(n) == 'drug']
            except:
                drug_neighbors = []
            
            # Find diseases connected to this gene
            try:
                disease_neighbors = [n for n in self.nx_graph.successors(gene_idx)
                                   if self.node_types.get(n) == 'disease']
            except:
                disease_neighbors = []
            
            if drug_neighbors and disease_neighbors:
                drug_idx = np.random.choice(drug_neighbors)
                disease_idx = np.random.choice(disease_neighbors)
                samples.append((drug_idx, disease_idx, 1))  # Positive
        
        # Generate negative samples (random pairs)
        for _ in range(num_samples // 2):
            drug_idx = np.random.choice(self.drug_indices)
            disease_idx = np.random.choice(self.disease_indices)
            
            # Check if there's a short path (expensive, so we skip)
            # Assume most random pairs are negative
            samples.append((drug_idx, disease_idx, 0))  # Negative
        
        logger.info(f"Generated {len(samples)} labeled samples")
        return samples
    
    def identify_failures_and_successes(
        self,
        num_failures: int = 5,
        num_successes: int = 5,
        num_samples: int = 5000
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Identify worst failures and best successes.
        
        Args:
            num_failures: Number of worst predictions to analyze
            num_successes: Number of best predictions to analyze
            num_samples: Number of samples to evaluate
            
        Returns:
            Tuple of (failures, successes) lists
        """
        logger.info(f"Identifying failures and successes from {num_samples} samples...")
        
        # Get labeled samples
        samples = self.get_ground_truth_labels(num_samples)
        
        # Compute predictions and errors
        results = []
        
        for drug_idx, disease_idx, label in tqdm(samples, desc="Computing predictions"):
            score = self.compute_prediction_score(drug_idx, disease_idx)
            
            # Error: |prediction - ground_truth|
            error = abs(score - label)
            
            # For failures, we want high confidence wrong predictions
            # Predicted positive (high score) but actually negative
            # Or predicted negative (low score) but actually positive
            if label == 0 and score > 0.7:
                # False positive (high confidence wrong)
                confidence_error = score  # Higher is worse
            elif label == 1 and score < 0.3:
                # False negative (high confidence wrong)
                confidence_error = 1 - score  # Higher is worse
            else:
                confidence_error = 0
            
            drug_name = self.mappings['idx2node'][drug_idx][1]
            disease_name = self.mappings['idx2node'][disease_idx][1]
            
            results.append({
                'drug_idx': drug_idx,
                'disease_idx': disease_idx,
                'drug_name': drug_name,
                'disease_name': disease_name,
                'prediction': score,
                'ground_truth': label,
                'error': error,
                'confidence_error': confidence_error,
                'type': 'FP' if (label == 0 and score > 0.5) else ('FN' if (label == 1 and score <= 0.5) else 'Correct')
            })
        
        # Sort by confidence error (worst first)
        results.sort(key=lambda x: x['confidence_error'], reverse=True)
        
        # Get failures (high confidence wrong)
        failures = [r for r in results if r['type'] in ['FP', 'FN']][:num_failures]
        
        # Get successes (correct predictions with high confidence)
        correct = [r for r in results if r['type'] == 'Correct']
        successes = sorted(correct, key=lambda x: abs(x['prediction'] - 0.5), reverse=True)[:num_successes]
        
        logger.info(f"Identified {len(failures)} failures and {len(successes)} successes")
        
        return failures, successes
    
    def find_paths(
        self,
        drug_idx: int,
        disease_idx: int,
        max_length: int = 4,
        max_paths: int = 20
    ) -> List[List[int]]:
        """Find paths between drug and disease."""
        try:
            paths = []
            for path in nx.all_simple_paths(
                self.nx_graph,
                source=drug_idx,
                target=disease_idx,
                cutoff=max_length
            ):
                paths.append(path)
                if len(paths) >= max_paths:
                    break
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []
    
    def analyze_subgraph(
        self,
        drug_idx: int,
        disease_idx: int,
        radius: int = 2
    ) -> Dict:
        """
        Analyze the local subgraph around drug and disease.
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            radius: Neighborhood radius
            
        Returns:
            Dictionary with subgraph statistics
        """
        # Get neighborhoods
        drug_neighbors = set()
        disease_neighbors = set()
        
        # BFS from drug
        if drug_idx in self.nx_graph:
            drug_neighbors.add(drug_idx)
            for _ in range(radius):
                new_neighbors = set()
                for node in list(drug_neighbors):
                    if node in self.nx_graph:
                        new_neighbors.update(self.nx_graph.successors(node))
                        new_neighbors.update(self.nx_graph.predecessors(node))
                drug_neighbors.update(new_neighbors)
        
        # BFS from disease
        if disease_idx in self.nx_graph:
            disease_neighbors.add(disease_idx)
            for _ in range(radius):
                new_neighbors = set()
                for node in list(disease_neighbors):
                    if node in self.nx_graph:
                        new_neighbors.update(self.nx_graph.successors(node))
                        new_neighbors.update(self.nx_graph.predecessors(node))
                disease_neighbors.update(new_neighbors)
        
        # Common neighbors
        common_neighbors = drug_neighbors & disease_neighbors
        
        # Get subgraph
        all_nodes = drug_neighbors | disease_neighbors
        subgraph = self.nx_graph.subgraph(all_nodes)
        
        # Analyze node types in subgraph
        node_type_counts = Counter()
        for node in all_nodes:
            ntype = self.node_types.get(node, 'unknown')
            node_type_counts[ntype] += 1
        
        # Analyze paths
        paths = self.find_paths(drug_idx, disease_idx)
        
        return {
            'drug_neighborhood_size': len(drug_neighbors),
            'disease_neighborhood_size': len(disease_neighbors),
            'common_neighbors': len(common_neighbors),
            'subgraph_nodes': len(all_nodes),
            'subgraph_edges': subgraph.number_of_edges(),
            'node_type_counts': dict(node_type_counts),
            'num_paths': len(paths),
            'shortest_path_length': len(paths[0]) - 1 if paths else None,
            'paths': paths
        }
    
    def compare_failure_with_success(
        self,
        failure: Dict,
        success: Dict
    ) -> Dict:
        """
        Compare structural differences between failure and success.
        
        Args:
            failure: Failure case dictionary
            success: Success case dictionary
            
        Returns:
            Dictionary with comparison results
        """
        # Analyze both subgraphs
        failure_subgraph = self.analyze_subgraph(
            failure['drug_idx'],
            failure['disease_idx']
        )
        
        success_subgraph = self.analyze_subgraph(
            success['drug_idx'],
            success['disease_idx']
        )
        
        # Compute differences
        comparison = {
            'failure': failure_subgraph,
            'success': success_subgraph,
            'differences': {
                'drug_neighborhood_diff': (
                    success_subgraph['drug_neighborhood_size'] - 
                    failure_subgraph['drug_neighborhood_size']
                ),
                'disease_neighborhood_diff': (
                    success_subgraph['disease_neighborhood_size'] - 
                    failure_subgraph['disease_neighborhood_size']
                ),
                'common_neighbors_diff': (
                    success_subgraph['common_neighbors'] - 
                    failure_subgraph['common_neighbors']
                ),
                'path_count_diff': (
                    success_subgraph['num_paths'] - 
                    failure_subgraph['num_paths']
                )
            }
        }
        
        return comparison
    
    def visualize_subgraph(
        self,
        drug_idx: int,
        disease_idx: int,
        output_path: Path,
        title: str = "Subgraph"
    ):
        """
        Visualize subgraph around drug-disease pair.
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            output_path: Output file path
            title: Plot title
        """
        # Get 2-hop neighborhood
        nodes = {drug_idx, disease_idx}
        
        for node in [drug_idx, disease_idx]:
            if node in self.nx_graph:
                # 1-hop
                nodes.update(self.nx_graph.successors(node))
                nodes.update(self.nx_graph.predecessors(node))
                
                # 2-hop (limited)
                for neighbor in list(nodes):
                    if neighbor in self.nx_graph:
                        successors = list(self.nx_graph.successors(neighbor))[:5]
                        predecessors = list(self.nx_graph.predecessors(neighbor))[:5]
                        nodes.update(successors)
                        nodes.update(predecessors)
        
        # Limit size
        if len(nodes) > 50:
            # Keep only closest nodes
            nodes = set(list(nodes)[:50])
        
        subgraph = self.nx_graph.subgraph(nodes)
        
        # Create plot
        plt.figure(figsize=(14, 10))
        
        # Layout
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        # Color nodes by type
        node_colors = []
        node_sizes = []
        
        for node in subgraph.nodes():
            ntype = self.node_types.get(node, 'unknown')
            
            if node == drug_idx:
                node_colors.append('red')
                node_sizes.append(1000)
            elif node == disease_idx:
                node_colors.append('blue')
                node_sizes.append(1000)
            elif ntype == 'drug':
                node_colors.append('lightcoral')
                node_sizes.append(300)
            elif ntype == 'disease':
                node_colors.append('lightblue')
                node_sizes.append(300)
            elif ntype == 'gene/protein':
                node_colors.append('lightgreen')
                node_sizes.append(200)
            else:
                node_colors.append('gray')
                node_sizes.append(100)
        
        # Draw
        nx.draw_networkx_nodes(
            subgraph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7
        )
        
        nx.draw_networkx_edges(
            subgraph, pos,
            edge_color='gray',
            alpha=0.3,
            arrows=True,
            arrowsize=10
        )
        
        # Labels for drug and disease only
        labels = {}
        for node in [drug_idx, disease_idx]:
            if node in subgraph:
                labels[node] = self.mappings['idx2node'][node][1][:20]
        
        nx.draw_networkx_labels(
            subgraph, pos,
            labels=labels,
            font_size=8
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Query Drug'),
            Patch(facecolor='blue', label='Query Disease'),
            Patch(facecolor='lightcoral', label='Other Drugs'),
            Patch(facecolor='lightblue', label='Other Diseases'),
            Patch(facecolor='lightgreen', label='Genes/Proteins')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved subgraph visualization to {output_path}")
    
    def generate_failure_hypotheses(
        self,
        failures: List[Dict],
        comparisons: List[Dict]
    ) -> List[str]:
        """
        Generate hypotheses about failure modes from analysis.
        
        Args:
            failures: List of failure cases
            comparisons: List of comparison results
            
        Returns:
            List of hypothesis strings
        """
        hypotheses = []
        
        # Analyze patterns in failures
        failure_types = Counter([f['type'] for f in failures])
        
        # Path analysis
        path_counts = [c['failure']['num_paths'] for c in comparisons if c['failure']['num_paths'] is not None]
        avg_failure_paths = np.mean(path_counts) if path_counts else 0
        
        success_path_counts = [c['success']['num_paths'] for c in comparisons if c['success']['num_paths'] is not None]
        avg_success_paths = np.mean(success_path_counts) if success_path_counts else 0
        
        # Hypothesis 1: Path-based
        if avg_success_paths > avg_failure_paths * 1.5:
            hypotheses.append(
                f"Model fails when there are FEW CONNECTING PATHS between drug and disease "
                f"(failures: {avg_failure_paths:.1f} paths, successes: {avg_success_paths:.1f} paths). "
                "Suggests model relies heavily on graph connectivity."
            )
        elif avg_failure_paths > avg_success_paths * 1.5:
            hypotheses.append(
                f"Model fails when there are TOO MANY PATHS (failures: {avg_failure_paths:.1f}, "
                f"successes: {avg_success_paths:.1f}). May indicate noisy connections or "
                "misleading intermediate nodes."
            )
        
        # Hypothesis 2: Neighborhood size
        failure_neighborhoods = [c['failure']['common_neighbors'] for c in comparisons]
        success_neighborhoods = [c['success']['common_neighbors'] for c in comparisons]
        
        avg_failure_common = np.mean(failure_neighborhoods) if failure_neighborhoods else 0
        avg_success_common = np.mean(success_neighborhoods) if success_neighborhoods else 0
        
        if avg_success_common > avg_failure_common * 1.3:
            hypotheses.append(
                f"Model fails when drug and disease have FEW COMMON NEIGHBORS "
                f"(failures: {avg_failure_common:.1f}, successes: {avg_success_common:.1f}). "
                "Suggests common neighbors provide strong signal for prediction."
            )
        
        # Hypothesis 3: Error types
        if failure_types['FP'] > failure_types['FN'] * 1.5:
            hypotheses.append(
                f"Model makes more FALSE POSITIVES ({failure_types['FP']}) than false negatives "
                f"({failure_types['FN']}). May be over-predicting connections, possibly due to "
                "high-degree nodes creating spurious associations."
            )
        elif failure_types['FN'] > failure_types['FP'] * 1.5:
            hypotheses.append(
                f"Model makes more FALSE NEGATIVES ({failure_types['FN']}) than false positives "
                f"({failure_types['FP']}). May be under-predicting, possibly missing subtle "
                "connections or being too conservative."
            )
        
        # Hypothesis 4: Graph structure
        subgraph_size_failures = [c['failure']['subgraph_nodes'] for c in comparisons]
        subgraph_size_successes = [c['success']['subgraph_nodes'] for c in comparisons]
        
        avg_failure_size = np.mean(subgraph_size_failures) if subgraph_size_failures else 0
        avg_success_size = np.mean(subgraph_size_successes) if subgraph_size_successes else 0
        
        if avg_failure_size < avg_success_size * 0.7:
            hypotheses.append(
                f"Failures occur in SPARSE NEIGHBORHOODS (avg {avg_failure_size:.0f} nodes) "
                f"compared to successes ({avg_success_size:.0f} nodes). "
                "Model may need more context to make accurate predictions."
            )
        
        # Default hypothesis if no clear patterns
        if not hypotheses:
            hypotheses.append(
                "No clear structural patterns differentiate failures from successes. "
                "Errors may be due to: (1) data quality issues, (2) complex non-structural factors, "
                "or (3) limitations of the evaluation protocol."
            )
        
        return hypotheses
    
    def suggest_improvements(self, hypotheses: List[str]) -> List[str]:
        """
        Suggest improvements based on failure hypotheses.
        
        Args:
            hypotheses: List of failure hypotheses
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        for hypothesis in hypotheses:
            if "FEW CONNECTING PATHS" in hypothesis:
                suggestions.append(
                    "• Add higher-order path features to capture multi-hop relationships"
                )
                suggestions.append(
                    "• Increase number of GCN layers to expand receptive field"
                )
                suggestions.append(
                    "• Consider attention mechanisms to focus on relevant paths"
                )
            
            elif "TOO MANY PATHS" in hypothesis:
                suggestions.append(
                    "• Implement path filtering to remove noisy connections"
                )
                suggestions.append(
                    "• Add path-level attention to weight important paths higher"
                )
                suggestions.append(
                    "• Use graph regularization to reduce spurious edges"
                )
            
            elif "FEW COMMON NEIGHBORS" in hypothesis:
                suggestions.append(
                    "• Add common neighbor features explicitly to the model"
                )
                suggestions.append(
                    "• Use graph neural network with higher-order aggregation"
                )
                suggestions.append(
                    "• Consider GraphSAINT or similar sampling for better neighborhood coverage"
                )
            
            elif "FALSE POSITIVES" in hypothesis:
                suggestions.append(
                    "• Increase regularization (dropout, weight decay) to reduce overfitting"
                )
                suggestions.append(
                    "• Add negative sampling during training with hard negatives"
                )
                suggestions.append(
                    "• Use degree-normalized aggregation to reduce bias toward high-degree nodes"
                )
            
            elif "FALSE NEGATIVES" in hypothesis:
                suggestions.append(
                    "• Reduce regularization to allow model to capture more connections"
                )
                suggestions.append(
                    "• Add positional encoding to capture long-range dependencies"
                )
                suggestions.append(
                    "• Increase model capacity (more layers, wider hidden dimensions)"
                )
            
            elif "SPARSE NEIGHBORHOODS" in hypothesis:
                suggestions.append(
                    "• Augment graph with inferred edges (e.g., from similarity)"
                )
                suggestions.append(
                    "• Use graph completion techniques to densify neighborhoods"
                )
                suggestions.append(
                    "• Add auxiliary node features (e.g., from external databases)"
                )
        
        # General improvements
        if not suggestions:
            suggestions = [
                "• Collect more data to improve graph coverage",
                "• Add domain-specific features (drug properties, disease symptoms)",
                "• Try ensemble methods combining multiple models",
                "• Implement uncertainty estimation to flag low-confidence predictions",
                "• Use transfer learning from related tasks or datasets"
            ]
        
        return list(set(suggestions))  # Remove duplicates
    
    def generate_report(
        self,
        failures: List[Dict],
        successes: List[Dict],
        comparisons: List[Dict],
        hypotheses: List[str],
        suggestions: List[str],
        output_path: Path
    ):
        """Generate comprehensive failure analysis report."""
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FAILURE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Analyzed {len(failures)} failure cases and {len(successes)} success cases\n")
            f.write(f"Identified {len(hypotheses)} key failure modes\n")
            f.write(f"Generated {len(suggestions)} improvement suggestions\n\n")
            
            # Failure Cases
            f.write("=" * 80 + "\n")
            f.write("WORST PREDICTION FAILURES\n")
            f.write("=" * 80 + "\n\n")
            
            for i, failure in enumerate(failures, 1):
                f.write(f"{i}. {failure['drug_name']} → {failure['disease_name']}\n")
                f.write(f"   Type: {failure['type']} (")
                if failure['type'] == 'FP':
                    f.write("Predicted positive but actually negative)\n")
                else:
                    f.write("Predicted negative but actually positive)\n")
                f.write(f"   Prediction Score: {failure['prediction']:.4f}\n")
                f.write(f"   Ground Truth: {failure['ground_truth']}\n")
                f.write(f"   Confidence Error: {failure['confidence_error']:.4f}\n")
                
                # Get comparison
                if i <= len(comparisons):
                    comp = comparisons[i-1]
                    f.write("\n   Structural Analysis:\n")
                    f.write(f"     - Drug neighborhood: {comp['failure']['drug_neighborhood_size']} nodes\n")
                    f.write(f"     - Disease neighborhood: {comp['failure']['disease_neighborhood_size']} nodes\n")
                    f.write(f"     - Common neighbors: {comp['failure']['common_neighbors']}\n")
                    f.write(f"     - Connecting paths: {comp['failure']['num_paths']}\n")
                    
                    if comp['failure']['num_paths'] > 0:
                        f.write(f"     - Shortest path length: {comp['failure']['shortest_path_length']}\n")
                    
                    f.write("\n   Potential Reasons for Failure:\n")
                    if comp['failure']['num_paths'] == 0:
                        f.write("     ✗ No direct paths connecting drug and disease\n")
                    elif comp['failure']['num_paths'] < 3:
                        f.write("     ⚠ Very few connecting paths (weak connection)\n")
                    
                    if comp['failure']['common_neighbors'] == 0:
                        f.write("     ✗ No common neighbors between drug and disease\n")
                    elif comp['failure']['common_neighbors'] < 2:
                        f.write("     ⚠ Very few common neighbors\n")
                    
                    if comp['failure']['drug_neighborhood_size'] < 10:
                        f.write("     ⚠ Sparse drug neighborhood (limited information)\n")
                    
                    if comp['failure']['disease_neighborhood_size'] < 10:
                        f.write("     ⚠ Sparse disease neighborhood (limited information)\n")
                
                f.write("\n")
            
            # Success Cases
            f.write("=" * 80 + "\n")
            f.write("SUCCESSFUL PREDICTIONS (For Comparison)\n")
            f.write("=" * 80 + "\n\n")
            
            for i, success in enumerate(successes, 1):
                f.write(f"{i}. {success['drug_name']} → {success['disease_name']}\n")
                f.write(f"   Prediction Score: {success['prediction']:.4f}\n")
                f.write(f"   Ground Truth: {success['ground_truth']}\n")
                
                if i <= len(comparisons):
                    comp = comparisons[i-1]
                    f.write(f"   Paths: {comp['success']['num_paths']}, ")
                    f.write(f"Common neighbors: {comp['success']['common_neighbors']}\n")
                
                f.write("\n")
            
            # Structural Comparison
            f.write("=" * 80 + "\n")
            f.write("STRUCTURAL COMPARISON: FAILURES VS SUCCESSES\n")
            f.write("=" * 80 + "\n\n")
            
            # Aggregate statistics
            failure_stats = {
                'paths': np.mean([c['failure']['num_paths'] for c in comparisons]),
                'common': np.mean([c['failure']['common_neighbors'] for c in comparisons]),
                'drug_nbrs': np.mean([c['failure']['drug_neighborhood_size'] for c in comparisons]),
                'disease_nbrs': np.mean([c['failure']['disease_neighborhood_size'] for c in comparisons])
            }
            
            success_stats = {
                'paths': np.mean([c['success']['num_paths'] for c in comparisons]),
                'common': np.mean([c['success']['common_neighbors'] for c in comparisons]),
                'drug_nbrs': np.mean([c['success']['drug_neighborhood_size'] for c in comparisons]),
                'disease_nbrs': np.mean([c['success']['disease_neighborhood_size'] for c in comparisons])
            }
            
            f.write(f"Average Connecting Paths:\n")
            f.write(f"  Failures: {failure_stats['paths']:.2f}\n")
            f.write(f"  Successes: {success_stats['paths']:.2f}\n")
            f.write(f"  Difference: {success_stats['paths'] - failure_stats['paths']:.2f}\n\n")
            
            f.write(f"Average Common Neighbors:\n")
            f.write(f"  Failures: {failure_stats['common']:.2f}\n")
            f.write(f"  Successes: {success_stats['common']:.2f}\n")
            f.write(f"  Difference: {success_stats['common'] - failure_stats['common']:.2f}\n\n")
            
            f.write(f"Average Drug Neighborhood Size:\n")
            f.write(f"  Failures: {failure_stats['drug_nbrs']:.0f}\n")
            f.write(f"  Successes: {success_stats['drug_nbrs']:.0f}\n")
            f.write(f"  Difference: {success_stats['drug_nbrs'] - failure_stats['drug_nbrs']:.0f}\n\n")
            
            f.write(f"Average Disease Neighborhood Size:\n")
            f.write(f"  Failures: {failure_stats['disease_nbrs']:.0f}\n")
            f.write(f"  Successes: {success_stats['disease_nbrs']:.0f}\n")
            f.write(f"  Difference: {success_stats['disease_nbrs'] - failure_stats['disease_nbrs']:.0f}\n\n")
            
            # Failure Hypotheses
            f.write("=" * 80 + "\n")
            f.write("FAILURE MODE HYPOTHESES\n")
            f.write("=" * 80 + "\n\n")
            
            for i, hypothesis in enumerate(hypotheses, 1):
                f.write(f"{i}. {hypothesis}\n\n")
            
            # Improvement Suggestions
            f.write("=" * 80 + "\n")
            f.write("SUGGESTED IMPROVEMENTS\n")
            f.write("=" * 80 + "\n\n")
            
            for suggestion in suggestions:
                f.write(f"{suggestion}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Saved failure analysis report to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze prediction failures to understand model limitations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic failure analysis
    python src/analyze_failures.py
    
    # Analyze more cases
    python src/analyze_failures.py --num_failures 10 --num_successes 10
    
    # With subgraph visualizations
    python src/analyze_failures.py --visualize_subgraphs
    
    # More samples for better statistics
    python src/analyze_failures.py --num_samples 10000
        """
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='output/models/best_model.pt',
        help='Path to trained model (default: output/models/best_model.pt)'
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data (default: data/processed)'
    )
    
    parser.add_argument(
        '--num_failures',
        type=int,
        default=5,
        help='Number of worst failures to analyze (default: 5)'
    )
    
    parser.add_argument(
        '--num_successes',
        type=int,
        default=5,
        help='Number of best successes to analyze (default: 5)'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5000,
        help='Number of samples to evaluate (default: 5000)'
    )
    
    parser.add_argument(
        '--visualize_subgraphs',
        action='store_true',
        help='Generate subgraph visualizations'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/failure_analysis',
        help='Output directory (default: results/failure_analysis)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize analyzer
        analyzer = FailureAnalyzer(
            model_path=args.model_path,
            data_dir=args.data_dir
        )
        
        # Identify failures and successes
        failures, successes = analyzer.identify_failures_and_successes(
            num_failures=args.num_failures,
            num_successes=args.num_successes,
            num_samples=args.num_samples
        )
        
        if not failures:
            logger.warning("No failures identified")
            sys.exit(0)
        
        # Compare failures with successes
        logger.info("Comparing failures with successes...")
        comparisons = []
        
        for i, (failure, success) in enumerate(zip(failures, successes)):
            comp = analyzer.compare_failure_with_success(failure, success)
            comparisons.append(comp)
        
        # Visualize subgraphs if requested
        if args.visualize_subgraphs:
            logger.info("Generating subgraph visualizations...")
            
            for i, failure in enumerate(failures, 1):
                output_path = output_dir / f'failure_{i}_subgraph.png'
                title = f"Failure {i}: {failure['drug_name']} → {failure['disease_name']}"
                analyzer.visualize_subgraph(
                    failure['drug_idx'],
                    failure['disease_idx'],
                    output_path,
                    title
                )
            
            for i, success in enumerate(successes, 1):
                output_path = output_dir / f'success_{i}_subgraph.png'
                title = f"Success {i}: {success['drug_name']} → {success['disease_name']}"
                analyzer.visualize_subgraph(
                    success['drug_idx'],
                    success['disease_idx'],
                    output_path,
                    title
                )
        
        # Generate hypotheses
        logger.info("Generating failure hypotheses...")
        hypotheses = analyzer.generate_failure_hypotheses(failures, comparisons)
        
        # Generate improvement suggestions
        logger.info("Generating improvement suggestions...")
        suggestions = analyzer.suggest_improvements(hypotheses)
        
        # Generate report
        report_path = output_dir / 'failure_analysis_report.txt'
        analyzer.generate_report(
            failures,
            successes,
            comparisons,
            hypotheses,
            suggestions,
            report_path
        )
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("FAILURE ANALYSIS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Analyzed {len(failures)} failures and {len(successes)} successes")
        logger.info(f"\nKey Findings:")
        for hypothesis in hypotheses:
            logger.info(f"  • {hypothesis}")
        logger.info(f"\nTop Improvements:")
        for suggestion in suggestions[:3]:
            logger.info(f"  {suggestion}")
        logger.info(f"\nFull report saved to: {report_path}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
