"""
Drug-Disease Prediction Case Studies

This script provides detailed analysis of drug-disease predictions for specific
diseases, including:
- Top-k drug predictions with confidence scores
- Known vs novel predictions
- Graph paths connecting drugs to diseases
- Network visualizations
- Medical interpretation reports

Usage:
    python src/case_studies.py --disease "Type 2 Diabetes" --top_k 10
    python src/case_studies.py --disease "Alzheimer" --top_k 20 --threshold 0.5
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import numpy as np
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


class DrugDiseaseCaseStudy:
    """
    Analyze drug-disease predictions for a specific disease.
    
    This class performs comprehensive analysis of model predictions for a given
    disease, including identifying top drug candidates, analyzing prediction paths,
    and generating visualizations and reports.
    
    Attributes:
        model: Trained DrugDiseaseModel
        mappings: Node and relation index mappings
        train_edges: Training set edges
        test_edges: Test set edges
        graph_data: Full graph structure
        device: Computation device (CPU/GPU)
    """
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = 'data/processed',
        device: Optional[torch.device] = None
    ):
        """
        Initialize case study analyzer.
        
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
        
        logger.info("Pre-computing embeddings on GPU for fast inference...")
        self._precompute_embeddings()
        
        logger.info("Building graph for path finding...")
        self._build_graph()
        
    def _load_data(self):
        """Load processed data and mappings."""
        # Load mappings
        mappings_path = self.data_dir / 'mappings.pt'
        self.mappings = torch.load(mappings_path, weights_only=False)
        
        # Create node type index
        self.node_types = {}  # idx -> type
        self.disease_indices = []
        self.drug_indices = []
        self.gene_indices = []
        
        for idx, (node_id, name, ntype) in self.mappings['idx2node'].items():
            self.node_types[idx] = ntype
            if ntype == 'disease':
                self.disease_indices.append(idx)
            elif ntype == 'drug':
                self.drug_indices.append(idx)
            elif ntype == 'gene/protein':
                self.gene_indices.append(idx)
        
        logger.info(f"Loaded {len(self.mappings['idx2node'])} nodes")
        logger.info(f"  Diseases: {len(self.disease_indices)}")
        logger.info(f"  Drugs: {len(self.drug_indices)}")
        logger.info(f"  Genes/Proteins: {len(self.gene_indices)}")
        
        # Load train/test edges
        train_data = torch.load(self.data_dir / 'train_data.pt', weights_only=False)
        test_data = torch.load(self.data_dir / 'test_data.pt', weights_only=False)
        
        self.train_edges = train_data['edge_index']
        self.train_edge_types = train_data['edge_type']
        self.test_edges = test_data['edge_index']
        self.test_edge_types = test_data['edge_type']
        
        # Load full graph
        full_graph = torch.load(self.data_dir / 'full_graph.pt', weights_only=False)
        self.graph_data = full_graph
        
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
        
        logger.info("Model loaded successfully")
    
    def _precompute_embeddings(self):
        """Pre-compute all node embeddings for fast inference."""
        edge_index = self.graph_data['edge_index'].to(self.device)
        edge_type = self.graph_data['edge_type'].to(self.device)
        
        with torch.no_grad():
            h = self.model.encoder(edge_index, edge_type)
        
        # Keep embeddings on GPU for fast access
        self.embeddings_gpu = h
        
        # Also store normalized versions for similarity computation
        self.embeddings_gpu_norm = F.normalize(h, p=2, dim=1)
        
        logger.info(f"Pre-computed embeddings: {h.shape}, device: {h.device}")
    
    def _build_graph(self):
        """Build NetworkX graph for path finding."""
        self.nx_graph = nx.MultiDiGraph()
        
        edge_index = self.graph_data['edge_index']
        edge_types = self.graph_data['edge_type']
        
        # Add all edges
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            rel = edge_types[i].item()
            
            self.nx_graph.add_edge(
                src, dst,
                relation=rel,
                relation_name=self.mappings['idx2relation'][rel]
            )
        
        logger.info(f"Built graph with {self.nx_graph.number_of_nodes()} nodes "
                   f"and {self.nx_graph.number_of_edges()} edges")
        
    def find_disease(self, disease_name: str) -> Optional[int]:
        """
        Find disease node index by name (case-insensitive partial match).
        
        Args:
            disease_name: Disease name to search for
            
        Returns:
            Disease node index if found, None otherwise
        """
        disease_name_lower = disease_name.lower()
        
        # First try exact match
        for idx in self.disease_indices:
            _, name, _ = self.mappings['idx2node'][idx]
            if name.lower() == disease_name_lower:
                return idx
        
        # Then try partial match
        matches = []
        for idx in self.disease_indices:
            _, name, _ = self.mappings['idx2node'][idx]
            if disease_name_lower in name.lower():
                matches.append((idx, name))
        
        if len(matches) == 1:
            logger.info(f"Found disease: {matches[0][1]}")
            return matches[0][0]
        elif len(matches) > 1:
            logger.warning(f"Multiple diseases match '{disease_name}':")
            for idx, name in matches[:5]:
                logger.warning(f"  - {name}")
            logger.info(f"Using first match: {matches[0][1]}")
            return matches[0][0]
        
        return None
    
    def predict_top_drugs(
        self,
        disease_idx: int,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        Predict top-k drugs for a given disease.
        
        Uses graph-based scoring: computes similarity between drug and disease
        embeddings learned by the RGCN model. Since PrimeKG doesn't have direct
        drug-disease edges, we use embedding similarity as a proxy for potential
        therapeutic relationships.
        
        Args:
            disease_idx: Disease node index
            top_k: Number of top predictions to return
            threshold: Minimum score threshold
            
        Returns:
            List of (drug_idx, score) tuples sorted by score
        """
        logger.info(f"Computing predictions for all {len(self.drug_indices)} drugs...")
        
        # Use pre-computed embeddings (already on GPU)
        disease_emb_norm = self.embeddings_gpu_norm[disease_idx]  # (hidden_dim,)
        
        # Get drug embeddings (already normalized)
        drug_indices_tensor = torch.tensor(self.drug_indices, device=self.device)
        drug_embs_norm = self.embeddings_gpu_norm[drug_indices_tensor]  # (num_drugs, hidden_dim)
        
        # Compute cosine similarity (all on GPU)
        similarity = torch.mv(drug_embs_norm, disease_emb_norm)
        
        # Convert to numpy
        scores = similarity.cpu().numpy()
        
        # Convert from [-1, 1] to [0, 1]
        scores = (scores + 1) / 2
        
        logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        logger.info(f"Mean score: {scores.mean():.4f}")
        
        # Filter by threshold and get top-k
        predictions = [(self.drug_indices[i], scores[i]) 
                      for i in range(len(scores)) if scores[i] >= threshold]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:top_k]
    
    def check_known_associations(
        self,
        disease_idx: int,
        drug_indices: List[int]
    ) -> Dict[int, bool]:
        """
        Check which drugs have known associations with the disease in training data.
        
        Args:
            disease_idx: Disease node index
            drug_indices: List of drug indices to check
            
        Returns:
            Dictionary mapping drug_idx to is_known (True/False)
        """
        known = {drug_idx: False for drug_idx in drug_indices}
        
        # Check training edges for drug-disease paths
        # This includes indirect connections through genes
        drug_set = set(drug_indices)
        
        for i in range(self.train_edges.shape[1]):
            src = self.train_edges[0, i].item()
            dst = self.train_edges[1, i].item()
            
            # Check if edge connects drug to disease (or vice versa)
            if src in drug_set and dst == disease_idx:
                known[src] = True
            elif src == disease_idx and dst in drug_set:
                known[dst] = True
        
        return known
    
    def find_paths(
        self,
        drug_idx: int,
        disease_idx: int,
        max_paths: int = 5,
        max_length: int = 4
    ) -> List[List[int]]:
        """
        Find paths between drug and disease in the graph.
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            max_paths: Maximum number of paths to return
            max_length: Maximum path length
            
        Returns:
            List of paths (each path is a list of node indices)
        """
        try:
            # Find all simple paths up to max_length
            paths = list(nx.all_simple_paths(
                self.nx_graph,
                source=drug_idx,
                target=disease_idx,
                cutoff=max_length
            ))
            
            # Sort by length and return top paths
            paths.sort(key=len)
            return paths[:max_paths]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def analyze_prediction(
        self,
        drug_idx: int,
        disease_idx: int,
        score: float,
        is_known: bool
    ) -> Dict:
        """
        Perform detailed analysis of a single drug-disease prediction.
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            score: Prediction score
            is_known: Whether this is a known association
            
        Returns:
            Dictionary with analysis results
        """
        drug_id, drug_name, drug_type = self.mappings['idx2node'][drug_idx]
        disease_id, disease_name, disease_type = self.mappings['idx2node'][disease_idx]
        
        # Find paths
        paths = self.find_paths(drug_idx, disease_idx, max_paths=5)
        
        # Analyze paths
        path_details = []
        intermediate_genes = set()
        
        for path in paths:
            path_info = {
                'length': len(path) - 1,
                'nodes': [],
                'relations': []
            }
            
            for i, node_idx in enumerate(path):
                node_id, node_name, node_type = self.mappings['idx2node'][node_idx]
                path_info['nodes'].append({
                    'idx': node_idx,
                    'name': node_name,
                    'type': node_type
                })
                
                if node_type == 'gene/protein' and node_idx not in [drug_idx, disease_idx]:
                    intermediate_genes.add((node_idx, node_name))
                
                # Get edge relation if not last node
                if i < len(path) - 1:
                    next_node = path[i + 1]
                    # Find edge relation
                    edge_data = self.nx_graph.get_edge_data(node_idx, next_node)
                    if edge_data:
                        rel_name = edge_data[0].get('relation_name', 'unknown')
                        path_info['relations'].append(rel_name)
            
            path_details.append(path_info)
        
        return {
            'drug': {
                'idx': drug_idx,
                'id': drug_id,
                'name': drug_name,
                'type': drug_type
            },
            'disease': {
                'idx': disease_idx,
                'id': disease_id,
                'name': disease_name,
                'type': disease_type
            },
            'score': float(score),
            'is_known': is_known,
            'status': 'Known Treatment' if is_known else 'Novel Prediction',
            'num_paths': len(paths),
            'paths': path_details,
            'intermediate_genes': sorted(list(intermediate_genes), key=lambda x: x[1])
        }
    
    def visualize_predictions(
        self,
        disease_name: str,
        predictions: List[Dict],
        output_dir: Path
    ):
        """
        Create visualizations for predictions.
        
        Args:
            disease_name: Disease name for titles
            predictions: List of prediction analysis dictionaries
            output_dir: Output directory for plots
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Bar chart of prediction scores
        fig, ax = plt.subplots(figsize=(12, 6))
        
        drug_names = [p['drug']['name'][:30] for p in predictions]
        scores = [p['score'] for p in predictions]
        colors = ['green' if p['is_known'] else 'blue' for p in predictions]
        
        bars = ax.barh(drug_names, scores, color=colors, alpha=0.7)
        ax.set_xlabel('Prediction Score', fontsize=12)
        ax.set_ylabel('Drug', fontsize=12)
        ax.set_title(f'Top Drug Predictions for {disease_name}', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Known Treatment'),
            Patch(facecolor='blue', alpha=0.7, label='Novel Prediction')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{score:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'prediction_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved prediction scores plot to {output_dir / 'prediction_scores.png'}")
        
        # 2. Network diagram for top prediction
        if predictions and predictions[0]['paths']:
            self._visualize_network(predictions[0], output_dir)
    
    def _visualize_network(self, prediction: Dict, output_dir: Path):
        """
        Visualize network paths for a single prediction.
        
        Args:
            prediction: Prediction analysis dictionary
            output_dir: Output directory
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create subgraph with all paths
        G = nx.DiGraph()
        
        drug_name = prediction['drug']['name']
        disease_name = prediction['disease']['name']
        
        for path_info in prediction['paths'][:3]:  # Show top 3 paths
            nodes = path_info['nodes']
            for i in range(len(nodes) - 1):
                src_name = nodes[i]['name']
                dst_name = nodes[i + 1]['name']
                relation = path_info['relations'][i] if i < len(path_info['relations']) else ''
                
                G.add_edge(src_name, dst_name, relation=relation)
        
        # Set up layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Draw nodes by type
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node == drug_name:
                node_colors.append('lightgreen')
                node_sizes.append(3000)
            elif node == disease_name:
                node_colors.append('lightcoral')
                node_sizes.append(3000)
            else:
                node_colors.append('lightblue')
                node_sizes.append(2000)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              arrows=True, arrowsize=20, 
                              arrowstyle='->', width=2, alpha=0.6, ax=ax)
        
        # Draw labels
        labels = {node: node[:25] + '...' if len(node) > 25 else node 
                 for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=9, 
                               font_weight='bold', ax=ax)
        
        # Draw edge labels (relations)
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                     font_size=7, ax=ax)
        
        ax.set_title(f'Drug-Disease Connection Paths\n{drug_name} → {disease_name}\n'
                    f'Score: {prediction["score"]:.3f} ({prediction["status"]})',
                    fontsize=12, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'network_paths_top.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved network diagram to {output_dir / 'network_paths_top.png'}")
    
    def generate_report(
        self,
        disease_name: str,
        disease_idx: int,
        predictions: List[Dict],
        output_path: Path
    ):
        """
        Generate comprehensive text report.
        
        Args:
            disease_name: Disease name
            disease_idx: Disease node index
            predictions: List of prediction analysis dictionaries
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"DRUG-DISEASE PREDICTION CASE STUDY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Disease: {disease_name}\n")
            f.write(f"Disease ID: {self.mappings['idx2node'][disease_idx][0]}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model.__class__.__name__}\n")
            f.write("\n")
            
            # Summary statistics
            f.write("-" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total predictions analyzed: {len(predictions)}\n")
            
            known_count = sum(1 for p in predictions if p['is_known'])
            novel_count = len(predictions) - known_count
            f.write(f"Known treatments: {known_count}\n")
            f.write(f"Novel predictions: {novel_count}\n")
            
            avg_score = np.mean([p['score'] for p in predictions])
            f.write(f"Average prediction score: {avg_score:.4f}\n")
            
            total_paths = sum(p['num_paths'] for p in predictions)
            f.write(f"Total graph paths found: {total_paths}\n")
            f.write("\n")
            
            # Top predictions
            f.write("=" * 80 + "\n")
            f.write("TOP DRUG PREDICTIONS\n")
            f.write("=" * 80 + "\n\n")
            
            for i, pred in enumerate(predictions, 1):
                f.write(f"{i}. {pred['drug']['name']}\n")
                f.write(f"   Status: {pred['status']}\n")
                f.write(f"   Prediction Score: {pred['score']:.4f}\n")
                f.write(f"   Drug ID: {pred['drug']['id']}\n")
                
                # Intermediate genes
                if pred['intermediate_genes']:
                    f.write(f"   Key Genes/Proteins ({len(pred['intermediate_genes'])}):\n")
                    for gene_idx, gene_name in pred['intermediate_genes'][:5]:
                        f.write(f"      - {gene_name}\n")
                    if len(pred['intermediate_genes']) > 5:
                        f.write(f"      ... and {len(pred['intermediate_genes']) - 5} more\n")
                
                # Paths
                if pred['paths']:
                    f.write(f"   Connection Paths ({pred['num_paths']} found):\n")
                    for j, path_info in enumerate(pred['paths'][:2], 1):
                        f.write(f"      Path {j} ({path_info['length']} steps):\n")
                        path_str = " → ".join([
                            f"{node['name'][:30]}" 
                            for node in path_info['nodes']
                        ])
                        f.write(f"         {path_str}\n")
                else:
                    f.write("   No direct paths found in graph\n")
                
                f.write("\n")
            
            # Medical interpretation
            f.write("=" * 80 + "\n")
            f.write("MEDICAL INTERPRETATION\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("Known Treatments:\n")
            known_drugs = [p for p in predictions if p['is_known']]
            if known_drugs:
                for pred in known_drugs:
                    f.write(f"  • {pred['drug']['name']} (score: {pred['score']:.4f})\n")
                    f.write(f"    This drug has documented associations with {disease_name} ")
                    f.write(f"in the training data.\n")
            else:
                f.write("  No known treatments found in top predictions.\n")
            f.write("\n")
            
            f.write("Novel Predictions:\n")
            novel_drugs = [p for p in predictions if not p['is_known']]
            if novel_drugs:
                for pred in novel_drugs[:5]:
                    f.write(f"  • {pred['drug']['name']} (score: {pred['score']:.4f})\n")
                    f.write(f"    This is a novel prediction not directly seen in training.\n")
                    if pred['intermediate_genes']:
                        genes = ', '.join([g[1] for g in pred['intermediate_genes'][:3]])
                        f.write(f"    Connected through: {genes}\n")
            else:
                f.write("  All top predictions are known treatments.\n")
            f.write("\n")
            
            # Biological insights
            f.write("=" * 80 + "\n")
            f.write("BIOLOGICAL INSIGHTS\n")
            f.write("=" * 80 + "\n\n")
            
            # Count most common intermediate genes
            all_genes = []
            for pred in predictions:
                all_genes.extend([g[1] for g in pred['intermediate_genes']])
            
            if all_genes:
                gene_counter = Counter(all_genes)
                f.write(f"Most frequent intermediate genes/proteins:\n")
                for gene, count in gene_counter.most_common(10):
                    f.write(f"  • {gene}: appears in {count} prediction paths\n")
                f.write("\n")
                f.write("These genes may play key roles in the disease mechanism and could be\n")
                f.write("potential therapeutic targets.\n")
            else:
                f.write("No intermediate genes found in prediction paths.\n")
            f.write("\n")
            
            # Recommendations
            f.write("=" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")
            f.write("1. Validate novel predictions with literature review and domain experts\n")
            f.write("2. Investigate high-confidence predictions for potential drug repurposing\n")
            f.write("3. Examine intermediate genes for mechanism of action insights\n")
            f.write("4. Consider experimental validation for top novel candidates\n")
            f.write("5. Review known treatments for model validation\n")
            f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Saved report to {output_path}")
    
    def run_case_study(
        self,
        disease_name: str,
        top_k: int = 10,
        threshold: float = 0.0,
        output_dir: str = 'results/case_studies'
    ):
        """
        Run complete case study for a disease.
        
        Args:
            disease_name: Disease name to analyze
            top_k: Number of top predictions to analyze
            threshold: Minimum prediction score threshold
            output_dir: Output directory for results
        """
        logger.info("=" * 80)
        logger.info(f"Starting case study for: {disease_name}")
        logger.info("=" * 80)
        
        # Find disease
        disease_idx = self.find_disease(disease_name)
        if disease_idx is None:
            logger.error(f"Disease '{disease_name}' not found in knowledge graph")
            logger.info("\nSuggestions: Try searching for:")
            logger.info("  - 'diabetes' → Type 1 Diabetes, Type 2 Diabetes")
            logger.info("  - 'alzheimer' → Alzheimer Disease")
            logger.info("  - 'cancer' → various cancer types")
            return
        
        disease_id, disease_name_full, _ = self.mappings['idx2node'][disease_idx]
        logger.info(f"Found disease: {disease_name_full} (ID: {disease_id})")
        
        # Get predictions
        predictions = self.predict_top_drugs(disease_idx, top_k=top_k, threshold=threshold)
        logger.info(f"Generated {len(predictions)} predictions")
        
        if not predictions:
            logger.warning("No predictions above threshold")
            return
        
        # Check known associations
        drug_indices = [drug_idx for drug_idx, _ in predictions]
        known_status = self.check_known_associations(disease_idx, drug_indices)
        
        # Analyze each prediction
        logger.info("Analyzing predictions...")
        detailed_predictions = []
        for drug_idx, score in tqdm(predictions, desc="Analyzing"):
            analysis = self.analyze_prediction(
                drug_idx, disease_idx, score, known_status[drug_idx]
            )
            detailed_predictions.append(analysis)
        
        # Create output directory
        output_path = Path(output_dir) / disease_name_full.replace('/', '_').replace(' ', '_')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate visualizations
        logger.info("Creating visualizations...")
        self.visualize_predictions(disease_name_full, detailed_predictions, output_path)
        
        # Generate report
        logger.info("Generating report...")
        report_path = output_path / 'case_study_report.txt'
        self.generate_report(disease_name_full, disease_idx, detailed_predictions, report_path)
        
        # Save detailed results as JSON
        import json
        results_path = output_path / 'predictions.json'
        with open(results_path, 'w') as f:
            json.dump(detailed_predictions, f, indent=2)
        logger.info(f"Saved detailed results to {results_path}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("CASE STUDY COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Disease: {disease_name_full}")
        logger.info(f"Top {len(detailed_predictions)} predictions analyzed")
        known_count = sum(1 for p in detailed_predictions if p['is_known'])
        logger.info(f"Known treatments: {known_count}")
        logger.info(f"Novel predictions: {len(detailed_predictions) - known_count}")
        logger.info(f"Results saved to: {output_path}")
        logger.info("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze drug-disease predictions for specific diseases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze Type 2 Diabetes
    python src/case_studies.py --disease "Type 2 Diabetes" --top_k 10
    
    # Analyze Alzheimer with threshold
    python src/case_studies.py --disease "Alzheimer" --top_k 20 --threshold 0.5
    
    # Use specific model
    python src/case_studies.py --disease "cancer" --model_path output/models/final_model.pt
        """
    )
    
    parser.add_argument(
        '--disease',
        type=str,
        required=True,
        help='Disease name to analyze (case-insensitive, supports partial matching)'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='Number of top predictions to analyze (default: 10)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Minimum prediction score threshold (default: 0.0)'
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
        default='results/case_studies',
        help='Output directory for results (default: results/case_studies)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    try:
        # Initialize case study
        case_study = DrugDiseaseCaseStudy(
            model_path=args.model_path,
            data_dir=args.data_dir
        )
        
        # Run analysis
        case_study.run_case_study(
            disease_name=args.disease,
            top_k=args.top_k,
            threshold=args.threshold,
            output_dir=args.output_dir
        )
        
    except Exception as e:
        logger.error(f"Error during case study: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    # Import pandas here to avoid issues if not installed
    import pandas as pd
    main()
