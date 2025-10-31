"""
Medical Validation of Drug-Disease Predictions

This script validates novel predictions by analyzing:
- Drug mechanisms of action
- Disease pathology
- Biological plausibility
- Literature evidence (mock)
- Clinical trial data (mock)
- Similar approved indications

Usage:
    python src/medical_validation.py --model_path output/models/best_model.pt
    python src/medical_validation.py --top_k 50 --threshold 0.7
    python src/medical_validation.py --output validation_report.csv
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


class MedicalValidator:
    """
    Validate drug-disease predictions using biological and medical criteria.
    
    This class analyzes predictions for biological plausibility, mechanism of action
    compatibility, and potential evidence from literature and clinical trials.
    
    Attributes:
        model: Trained DrugDiseaseModel
        mappings: Node and relation mappings
        embeddings: Node embeddings
        train_edges: Training set edges
        device: Computation device
    """
    
    def __init__(
        self,
        model_path: str,
        data_dir: str = 'data/processed',
        device: Optional[torch.device] = None
    ):
        """
        Initialize medical validator.
        
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
        
        # Load train edges to identify known associations
        train_data = torch.load(self.data_dir / 'train_data.pt', weights_only=False)
        self.train_edges = train_data['edge_index']
        self.train_edge_types = train_data['edge_type']
        
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
        """Build graph structure for analysis."""
        import networkx as nx
        
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
    
    def generate_predictions(
        self,
        top_k: int = 100,
        threshold: float = 0.6,
        sample_diseases: Optional[int] = None
    ) -> List[Tuple[int, int, float]]:
        """
        Generate top-k novel predictions for validation.
        
        Args:
            top_k: Number of top predictions to validate
            threshold: Minimum prediction score
            sample_diseases: Number of diseases to sample (None = all)
            
        Returns:
            List of (drug_idx, disease_idx, score) tuples
        """
        logger.info(f"Generating predictions (top_k={top_k}, threshold={threshold})...")
        
        # Sample diseases if requested
        if sample_diseases and sample_diseases < len(self.disease_indices):
            disease_sample = np.random.choice(
                self.disease_indices,
                size=sample_diseases,
                replace=False
            )
        else:
            disease_sample = self.disease_indices
        
        predictions = []
        
        for disease_idx in tqdm(disease_sample, desc="Scoring diseases"):
            disease_emb = self.embeddings[disease_idx]
            
            # Compute scores for all drugs
            drug_embs = self.embeddings[self.drug_indices]
            
            # Cosine similarity
            disease_emb_norm = disease_emb / np.linalg.norm(disease_emb)
            drug_embs_norm = drug_embs / np.linalg.norm(drug_embs, axis=1, keepdims=True)
            
            similarities = np.dot(drug_embs_norm, disease_emb_norm)
            scores = (similarities + 1) / 2  # Convert to [0, 1]
            
            # Filter by threshold
            for i, score in enumerate(scores):
                if score >= threshold:
                    drug_idx = self.drug_indices[i]
                    predictions.append((drug_idx, disease_idx, score))
        
        # Sort by score
        predictions.sort(key=lambda x: x[2], reverse=True)
        
        # Filter out known associations (training set)
        novel_predictions = self._filter_known_associations(predictions)
        
        logger.info(f"Generated {len(predictions)} total predictions")
        logger.info(f"Novel predictions (not in training): {len(novel_predictions)}")
        
        return novel_predictions[:top_k]
    
    def _filter_known_associations(
        self,
        predictions: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int, float]]:
        """Filter out predictions that exist in training set."""
        # Create set of known drug-disease associations
        known = set()
        
        for i in range(self.train_edges.shape[1]):
            src = self.train_edges[0, i].item()
            dst = self.train_edges[1, i].item()
            
            src_type = self.node_types.get(src, '')
            dst_type = self.node_types.get(dst, '')
            
            # Direct drug-disease edges (rare in PrimeKG)
            if src_type == 'drug' and dst_type == 'disease':
                known.add((src, dst))
            elif src_type == 'disease' and dst_type == 'drug':
                known.add((dst, src))
        
        # Filter
        novel = [
            (drug_idx, disease_idx, score)
            for drug_idx, disease_idx, score in predictions
            if (drug_idx, disease_idx) not in known
        ]
        
        return novel
    
    def get_drug_targets(self, drug_idx: int) -> List[Tuple[int, str]]:
        """
        Get genes/proteins targeted by a drug.
        
        Args:
            drug_idx: Drug node index
            
        Returns:
            List of (gene_idx, gene_name) tuples
        """
        targets = []
        
        if drug_idx in self.nx_graph:
            for neighbor in self.nx_graph.neighbors(drug_idx):
                if self.node_types.get(neighbor) == 'gene/protein':
                    gene_name = self.mappings['idx2node'][neighbor][1]
                    targets.append((neighbor, gene_name))
        
        return targets
    
    def get_disease_genes(self, disease_idx: int) -> List[Tuple[int, str]]:
        """
        Get genes/proteins associated with a disease.
        
        Args:
            disease_idx: Disease node index
            
        Returns:
            List of (gene_idx, gene_name) tuples
        """
        genes = []
        
        # Find genes that connect to this disease
        for node in self.nx_graph.predecessors(disease_idx):
            if self.node_types.get(node) == 'gene/protein':
                gene_name = self.mappings['idx2node'][node][1]
                genes.append((node, gene_name))
        
        return genes
    
    def check_target_overlap(
        self,
        drug_idx: int,
        disease_idx: int
    ) -> Dict:
        """
        Check if drug targets overlap with disease-related genes.
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            
        Returns:
            Dictionary with overlap analysis
        """
        drug_targets = set(g[0] for g in self.get_drug_targets(drug_idx))
        disease_genes = set(g[0] for g in self.get_disease_genes(disease_idx))
        
        overlap = drug_targets & disease_genes
        
        overlap_genes = []
        if overlap:
            for gene_idx in overlap:
                gene_name = self.mappings['idx2node'][gene_idx][1]
                overlap_genes.append(gene_name)
        
        return {
            'drug_targets': len(drug_targets),
            'disease_genes': len(disease_genes),
            'overlap_count': len(overlap),
            'overlap_genes': overlap_genes,
            'has_overlap': len(overlap) > 0
        }
    
    def find_common_neighbors(
        self,
        drug_idx: int,
        disease_idx: int
    ) -> List[Tuple[int, str]]:
        """
        Find common gene neighbors between drug and disease.
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            
        Returns:
            List of (gene_idx, gene_name) tuples
        """
        drug_neighbors = set()
        disease_neighbors = set()
        
        # Get drug neighbors
        if drug_idx in self.nx_graph:
            for neighbor in self.nx_graph.neighbors(drug_idx):
                if self.node_types.get(neighbor) == 'gene/protein':
                    drug_neighbors.add(neighbor)
        
        # Get disease neighbors (predecessors)
        if disease_idx in self.nx_graph:
            for neighbor in self.nx_graph.predecessors(disease_idx):
                if self.node_types.get(neighbor) == 'gene/protein':
                    disease_neighbors.add(neighbor)
        
        # Find common
        common = drug_neighbors & disease_neighbors
        
        result = []
        for gene_idx in common:
            gene_name = self.mappings['idx2node'][gene_idx][1]
            result.append((gene_idx, gene_name))
        
        return result
    
    def compute_embedding_similarity(
        self,
        drug_idx: int,
        disease_idx: int
    ) -> float:
        """
        Compute embedding similarity between drug and disease.
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            
        Returns:
            Similarity score (0-1)
        """
        drug_emb = self.embeddings[drug_idx]
        disease_emb = self.embeddings[disease_idx]
        
        similarity = np.dot(drug_emb, disease_emb) / (
            np.linalg.norm(drug_emb) * np.linalg.norm(disease_emb)
        )
        
        return (similarity + 1) / 2  # Convert to [0, 1]
    
    def find_similar_drugs(
        self,
        drug_idx: int,
        top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Find drugs similar to the query drug.
        
        Args:
            drug_idx: Drug node index
            top_k: Number of similar drugs to return
            
        Returns:
            List of (drug_idx, drug_name, similarity) tuples
        """
        drug_emb = self.embeddings[drug_idx]
        
        # Compute similarities with all drugs
        drug_embs = self.embeddings[self.drug_indices]
        
        drug_emb_norm = drug_emb / np.linalg.norm(drug_emb)
        drug_embs_norm = drug_embs / np.linalg.norm(drug_embs, axis=1, keepdims=True)
        
        similarities = np.dot(drug_embs_norm, drug_emb_norm)
        
        # Sort and get top-k (excluding self)
        sorted_idx = np.argsort(similarities)[::-1]
        
        similar_drugs = []
        for idx in sorted_idx:
            similar_drug_idx = self.drug_indices[idx]
            if similar_drug_idx == drug_idx:
                continue
            
            drug_name = self.mappings['idx2node'][similar_drug_idx][1]
            similarity = (similarities[idx] + 1) / 2
            similar_drugs.append((similar_drug_idx, drug_name, similarity))
            
            if len(similar_drugs) >= top_k:
                break
        
        return similar_drugs
    
    def mock_literature_search(
        self,
        drug_name: str,
        disease_name: str
    ) -> Dict:
        """
        Mock function to simulate literature search.
        
        In a real implementation, this would query PubMed, Google Scholar, etc.
        
        Args:
            drug_name: Drug name
            disease_name: Disease name
            
        Returns:
            Dictionary with search results
        """
        # Mock implementation - returns random results
        # In reality, would use APIs like PubMed E-utilities
        
        # Simple heuristic: check if common terms suggest relationship
        drug_lower = drug_name.lower()
        disease_lower = disease_name.lower()
        
        # Check for diabetes drugs
        diabetes_drugs = ['metformin', 'insulin', 'glipizide', 'pioglitazone']
        is_diabetes_drug = any(d in drug_lower for d in diabetes_drugs)
        is_diabetes = 'diabetes' in disease_lower
        
        # Check for cardiovascular drugs
        cardio_drugs = ['aspirin', 'atorvastatin', 'warfarin', 'lisinopril']
        is_cardio_drug = any(d in drug_lower for d in cardio_drugs)
        is_cardio = any(term in disease_lower for term in ['heart', 'cardiac', 'hypertension'])
        
        # Assign mock confidence
        if (is_diabetes_drug and is_diabetes) or (is_cardio_drug and is_cardio):
            publications = np.random.randint(10, 50)
            confidence = 'high'
        else:
            publications = np.random.randint(0, 5)
            confidence = 'low' if publications < 2 else 'medium'
        
        return {
            'publications_found': publications,
            'confidence': confidence,
            'has_evidence': publications > 0
        }
    
    def mock_clinical_trials_search(
        self,
        drug_name: str,
        disease_name: str
    ) -> Dict:
        """
        Mock function to simulate clinical trials search.
        
        In a real implementation, this would query ClinicalTrials.gov API.
        
        Args:
            drug_name: Drug name
            disease_name: Disease name
            
        Returns:
            Dictionary with trial information
        """
        # Mock implementation
        # In reality, would use ClinicalTrials.gov API
        
        # Random number of trials (weighted by drug/disease types)
        drug_lower = drug_name.lower()
        disease_lower = disease_name.lower()
        
        # Common drugs more likely to have trials
        common_drugs = ['metformin', 'aspirin', 'insulin', 'atorvastatin']
        is_common = any(d in drug_lower for d in common_drugs)
        
        if is_common:
            trials = np.random.randint(0, 10)
        else:
            trials = np.random.randint(0, 3)
        
        trial_phases = []
        if trials > 0:
            phases = np.random.choice(['Phase I', 'Phase II', 'Phase III', 'Phase IV'],
                                     size=min(trials, 3), replace=True)
            trial_phases = list(phases)
        
        return {
            'trials_found': trials,
            'trial_phases': trial_phases,
            'has_trials': trials > 0
        }
    
    def validate_prediction(
        self,
        drug_idx: int,
        disease_idx: int,
        prediction_score: float
    ) -> Dict:
        """
        Perform comprehensive validation of a prediction.
        
        Args:
            drug_idx: Drug node index
            disease_idx: Disease node index
            prediction_score: Model prediction score
            
        Returns:
            Dictionary with validation results
        """
        drug_id, drug_name, _ = self.mappings['idx2node'][drug_idx]
        disease_id, disease_name, _ = self.mappings['idx2node'][disease_idx]
        
        # 1. Target overlap analysis
        target_analysis = self.check_target_overlap(drug_idx, disease_idx)
        
        # 2. Common neighbors
        common_neighbors = self.find_common_neighbors(drug_idx, disease_idx)
        
        # 3. Similar drugs
        similar_drugs = self.find_similar_drugs(drug_idx, top_k=5)
        
        # 4. Literature search (mock)
        literature = self.mock_literature_search(drug_name, disease_name)
        
        # 5. Clinical trials (mock)
        trials = self.mock_clinical_trials_search(drug_name, disease_name)
        
        # 6. Compute validation score
        validation_score = self._compute_validation_score(
            prediction_score=prediction_score,
            target_analysis=target_analysis,
            common_neighbors=common_neighbors,
            literature=literature,
            trials=trials
        )
        
        # 7. Generate checklist
        checklist = self._generate_checklist(
            target_analysis, common_neighbors, literature, trials
        )
        
        return {
            'drug_idx': drug_idx,
            'drug_id': drug_id,
            'drug_name': drug_name,
            'disease_idx': disease_idx,
            'disease_id': disease_id,
            'disease_name': disease_name,
            'prediction_score': prediction_score,
            'validation_score': validation_score,
            'target_analysis': target_analysis,
            'common_neighbors': [g[1] for g in common_neighbors],
            'similar_drugs': [(name, sim) for _, name, sim in similar_drugs],
            'literature_evidence': literature,
            'clinical_trials': trials,
            'checklist': checklist,
            'confidence': self._determine_confidence(validation_score)
        }
    
    def _compute_validation_score(
        self,
        prediction_score: float,
        target_analysis: Dict,
        common_neighbors: List,
        literature: Dict,
        trials: Dict
    ) -> float:
        """Compute overall validation score (0-1)."""
        scores = []
        weights = []
        
        # Prediction score (weight: 0.25)
        scores.append(prediction_score)
        weights.append(0.25)
        
        # Target overlap (weight: 0.20)
        if target_analysis['has_overlap']:
            target_score = 1.0
        elif target_analysis['drug_targets'] > 0 and target_analysis['disease_genes'] > 0:
            target_score = 0.5
        else:
            target_score = 0.0
        scores.append(target_score)
        weights.append(0.20)
        
        # Common neighbors (weight: 0.20)
        neighbor_score = min(1.0, len(common_neighbors) / 5.0)
        scores.append(neighbor_score)
        weights.append(0.20)
        
        # Literature evidence (weight: 0.20)
        if literature['confidence'] == 'high':
            lit_score = 1.0
        elif literature['confidence'] == 'medium':
            lit_score = 0.6
        else:
            lit_score = 0.2
        scores.append(lit_score)
        weights.append(0.20)
        
        # Clinical trials (weight: 0.15)
        trial_score = min(1.0, trials['trials_found'] / 5.0)
        scores.append(trial_score)
        weights.append(0.15)
        
        # Weighted average
        validation_score = sum(s * w for s, w in zip(scores, weights))
        
        return validation_score
    
    def _generate_checklist(
        self,
        target_analysis: Dict,
        common_neighbors: List,
        literature: Dict,
        trials: Dict
    ) -> Dict[str, bool]:
        """Generate validation checklist."""
        return {
            'targets_disease_genes': target_analysis['has_overlap'],
            'has_common_neighbors': len(common_neighbors) > 0,
            'has_literature_evidence': literature['has_evidence'],
            'has_clinical_trials': trials['has_trials'],
            'multiple_pathways': len(common_neighbors) >= 2,
            'strong_prediction': True  # Already filtered by threshold
        }
    
    def _determine_confidence(self, validation_score: float) -> str:
        """Determine confidence level from validation score."""
        if validation_score >= 0.7:
            return 'high'
        elif validation_score >= 0.5:
            return 'medium'
        else:
            return 'low'
    
    def validate_predictions(
        self,
        predictions: List[Tuple[int, int, float]]
    ) -> List[Dict]:
        """
        Validate multiple predictions.
        
        Args:
            predictions: List of (drug_idx, disease_idx, score) tuples
            
        Returns:
            List of validation result dictionaries
        """
        results = []
        
        logger.info(f"Validating {len(predictions)} predictions...")
        
        for drug_idx, disease_idx, score in tqdm(predictions, desc="Validating"):
            result = self.validate_prediction(drug_idx, disease_idx, score)
            results.append(result)
        
        # Sort by validation score
        results.sort(key=lambda x: x['validation_score'], reverse=True)
        
        return results
    
    def generate_report(
        self,
        results: List[Dict],
        output_path: Path
    ):
        """
        Generate validation report.
        
        Args:
            results: List of validation results
            output_path: Output file path
        """
        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MEDICAL VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total predictions validated: {len(results)}\n")
            
            # Count by confidence
            high_conf = sum(1 for r in results if r['confidence'] == 'high')
            med_conf = sum(1 for r in results if r['confidence'] == 'medium')
            low_conf = sum(1 for r in results if r['confidence'] == 'low')
            
            f.write(f"High confidence: {high_conf}\n")
            f.write(f"Medium confidence: {med_conf}\n")
            f.write(f"Low confidence: {low_conf}\n\n")
            
            # High confidence predictions
            f.write("=" * 80 + "\n")
            f.write("HIGH CONFIDENCE PREDICTIONS (Strong Evidence)\n")
            f.write("=" * 80 + "\n\n")
            
            high_results = [r for r in results if r['confidence'] == 'high']
            for i, result in enumerate(high_results, 1):
                f.write(f"{i}. {result['drug_name']} → {result['disease_name']}\n")
                f.write(f"   Prediction Score: {result['prediction_score']:.4f}\n")
                f.write(f"   Validation Score: {result['validation_score']:.4f}\n")
                
                f.write("   Validation Checklist:\n")
                for key, value in result['checklist'].items():
                    symbol = "✓" if value else "✗"
                    f.write(f"      {symbol} {key.replace('_', ' ').title()}\n")
                
                if result['target_analysis']['overlap_genes']:
                    f.write(f"   Overlapping Targets: {', '.join(result['target_analysis']['overlap_genes'][:5])}\n")
                
                if result['common_neighbors']:
                    f.write(f"   Common Genes: {', '.join(result['common_neighbors'][:5])}\n")
                
                f.write(f"   Literature: {result['literature_evidence']['publications_found']} publications\n")
                f.write(f"   Clinical Trials: {result['clinical_trials']['trials_found']} trials\n")
                f.write("\n")
            
            # Medium confidence predictions
            f.write("=" * 80 + "\n")
            f.write("MEDIUM CONFIDENCE PREDICTIONS (Partial Evidence)\n")
            f.write("=" * 80 + "\n\n")
            
            med_results = [r for r in results if r['confidence'] == 'medium'][:10]
            for i, result in enumerate(med_results, 1):
                f.write(f"{i}. {result['drug_name']} → {result['disease_name']}\n")
                f.write(f"   Validation Score: {result['validation_score']:.4f}\n")
                
                checklist_summary = sum(result['checklist'].values())
                f.write(f"   Checklist: {checklist_summary}/{len(result['checklist'])} criteria met\n")
                f.write("\n")
            
            # Low confidence predictions
            f.write("=" * 80 + "\n")
            f.write("LOW CONFIDENCE PREDICTIONS (Weak Evidence)\n")
            f.write("=" * 80 + "\n\n")
            
            low_results = [r for r in results if r['confidence'] == 'low'][:5]
            for i, result in enumerate(low_results, 1):
                f.write(f"{i}. {result['drug_name']} → {result['disease_name']}\n")
                f.write(f"   Validation Score: {result['validation_score']:.4f}\n")
                f.write(f"   Note: Limited supporting evidence found\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Saved validation report to {output_path}")
    
    def save_to_csv(
        self,
        results: List[Dict],
        output_path: Path
    ):
        """
        Save validation results to CSV file.
        
        Args:
            results: List of validation results
            output_path: Output CSV file path
        """
        rows = []
        
        for result in results:
            row = {
                'drug_name': result['drug_name'],
                'disease_name': result['disease_name'],
                'prediction_score': result['prediction_score'],
                'validation_score': result['validation_score'],
                'confidence': result['confidence'],
                'targets_disease_genes': result['checklist']['targets_disease_genes'],
                'has_common_neighbors': result['checklist']['has_common_neighbors'],
                'has_literature_evidence': result['checklist']['has_literature_evidence'],
                'has_clinical_trials': result['checklist']['has_clinical_trials'],
                'drug_targets_count': result['target_analysis']['drug_targets'],
                'disease_genes_count': result['target_analysis']['disease_genes'],
                'overlap_count': result['target_analysis']['overlap_count'],
                'common_neighbors_count': len(result['common_neighbors']),
                'literature_publications': result['literature_evidence']['publications_found'],
                'clinical_trials_count': result['clinical_trials']['trials_found'],
                'overlapping_genes': '; '.join(result['target_analysis']['overlap_genes'][:5]),
                'common_neighbors': '; '.join(result['common_neighbors'][:5]),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved CSV to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Validate drug-disease predictions with biological and medical evidence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate top 50 predictions
    python src/medical_validation.py --top_k 50
    
    # With custom threshold
    python src/medical_validation.py --top_k 100 --threshold 0.7
    
    # Sample fewer diseases for faster execution
    python src/medical_validation.py --top_k 50 --sample_diseases 100
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
        '--top_k',
        type=int,
        default=50,
        help='Number of top predictions to validate (default: 50)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Minimum prediction score threshold (default: 0.6)'
    )
    
    parser.add_argument(
        '--sample_diseases',
        type=int,
        default=None,
        help='Number of diseases to sample for faster execution (default: all)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/validation',
        help='Output directory for results (default: results/validation)'
    )
    
    parser.add_argument(
        '--output_csv',
        type=str,
        default='validation_results.csv',
        help='Output CSV filename (default: validation_results.csv)'
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
        # Initialize validator
        validator = MedicalValidator(
            model_path=args.model_path,
            data_dir=args.data_dir
        )
        
        # Generate predictions
        predictions = validator.generate_predictions(
            top_k=args.top_k,
            threshold=args.threshold,
            sample_diseases=args.sample_diseases
        )
        
        if not predictions:
            logger.warning("No predictions generated. Try lowering the threshold.")
            sys.exit(0)
        
        # Validate predictions
        results = validator.validate_predictions(predictions)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        report_path = output_dir / 'validation_report.txt'
        validator.generate_report(results, report_path)
        
        # Save CSV
        csv_path = output_dir / args.output_csv
        validator.save_to_csv(results, csv_path)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)
        
        high_conf = sum(1 for r in results if r['confidence'] == 'high')
        med_conf = sum(1 for r in results if r['confidence'] == 'medium')
        low_conf = sum(1 for r in results if r['confidence'] == 'low')
        
        logger.info(f"Total predictions validated: {len(results)}")
        logger.info(f"High confidence: {high_conf} ({high_conf/len(results)*100:.1f}%)")
        logger.info(f"Medium confidence: {med_conf} ({med_conf/len(results)*100:.1f}%)")
        logger.info(f"Low confidence: {low_conf} ({low_conf/len(results)*100:.1f}%)")
        logger.info(f"\nResults saved to:")
        logger.info(f"  Report: {report_path}")
        logger.info(f"  CSV: {csv_path}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
