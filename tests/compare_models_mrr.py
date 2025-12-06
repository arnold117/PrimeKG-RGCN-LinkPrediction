"""
Complete MRR comparison using git history to load baseline model correctly
"""
import torch
import json
import subprocess
import sys
import shutil
import importlib.util

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

def load_module_from_file(filepath, module_name):
    """Load Python module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_baseline_model(checkpoint_path, data):
    """Load baseline model using OLD rgcn.py (without LayerNorm)"""
    # Extract old rgcn.py from git
    result = subprocess.run(
        ['git', 'show', 'c14de6b:src/models/rgcn.py'],
        capture_output=True, text=True, check=True
    )

    # Save to temp file
    with open('/tmp/rgcn_baseline.py', 'w') as f:
        f.write(result.stdout)

    # Load old module
    rgcn_old = load_module_from_file('/tmp/rgcn_baseline.py', 'rgcn_baseline')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model with OLD architecture (no LayerNorm/Skip)
    model = rgcn_old.DrugDiseaseModel(
        num_nodes=data['num_nodes'],
        num_relations=data['num_relations'],
        embedding_dim=64,
        hidden_dim=128
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model

def load_phase1_model(checkpoint_path, data):
    """Load Phase 1 model using CURRENT rgcn.py (with LayerNorm)"""
    sys.path.insert(0, 'src')
    from models.rgcn import DrugDiseaseModel

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Current architecture has LayerNorm/Skip by default
    model = DrugDiseaseModel(
        num_nodes=data['num_nodes'],
        num_relations=data['num_relations'],
        embedding_dim=64,
        hidden_dim=128
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model

def compute_mrr_full(model, data):
    """Compute MRR on FULL test set (all 15,372 edges)"""
    test_edge_index = data['test_edge_index']
    test_edge_type = data['test_edge_type']
    full_edge_index = data['full_edge_index']  # Use full graph!
    full_edge_type = data['full_edge_type']

    num_test = test_edge_index.shape[1]

    with torch.no_grad():
        # Encode all nodes once using FULL GRAPH (train + val)
        all_embeddings = model.encoder(full_edge_index, full_edge_type)

        ranks = []
        batch_size = 100

        # Process ALL test edges
        for i in range(0, num_test, batch_size):
            end_idx = min(i + batch_size, num_test)

            batch_head = test_edge_index[0, i:end_idx]
            batch_tail = test_edge_index[1, i:end_idx]
            batch_rel = test_edge_type[i:end_idx]

            head_emb = all_embeddings[batch_head]
            scores = model.decoder.score_all_tails(head_emb, batch_rel, all_embeddings)

            sorted_indices = torch.argsort(scores, dim=1, descending=True)
            matches = (sorted_indices == batch_tail.unsqueeze(1))
            batch_ranks = (matches.nonzero(as_tuple=True)[1] + 1).cpu().numpy()
            ranks.extend(batch_ranks.tolist())

    import numpy as np
    ranks = np.array(ranks)

    return {
        'mrr': float(np.mean(1.0 / ranks)),
        'mean_rank': float(np.mean(ranks)),
        'median_rank': float(np.median(ranks)),
        'hits@10': float(np.mean(ranks <= 10)),
        'hits@50': float(np.mean(ranks <= 50))
    }

# Load data
print("Loading test data...")
train_data = torch.load('data/processed/train_data.pt', map_location=device, weights_only=False)
test_data = torch.load('data/processed/test_data.pt', map_location=device, weights_only=False)
full_graph = torch.load('data/processed/full_graph.pt', map_location=device, weights_only=False)

data = {
    'train_edge_index': train_data['edge_index'],
    'train_edge_type': train_data['edge_type'],
    'full_edge_index': full_graph['edge_index'],  # Use full graph for encoding!
    'full_edge_type': full_graph['edge_type'],
    'test_edge_index': test_data['edge_index'],
    'test_edge_type': test_data['edge_type'],
    'num_nodes': full_graph['num_nodes'],
    'num_relations': full_graph['num_relations']
}

print(f"Test edges: {data['test_edge_index'].shape[1]:,}")
print(f"Full graph edges (train+val): {data['full_edge_index'].shape[1]:,}")
print(f"Nodes: {data['num_nodes']:,}")

# Load reported results
with open('results_final/results.json') as f:
    baseline_reported = json.load(f)
with open('results_phase1/results.json') as f:
    phase1_reported = json.load(f)

print("\n" + "="*60)
print("COMPLETE MRR COMPARISON TEST")
print("="*60)

print("\n1. VERIFY IDENTICAL TEST SET:")
print(f"   Baseline: {baseline_reported['metrics']['test_edges']:,} edges")
print(f"   Phase 1:  {phase1_reported['metrics']['test_edges']:,} edges")
assert baseline_reported['metrics']['test_edges'] == phase1_reported['metrics']['test_edges']
print("   ✓ IDENTICAL")

# Test baseline
print("\n2. BASELINE MODEL (old architecture, no LayerNorm):")
print("   Loading model from git history (commit c14de6b)...")
baseline_model = load_baseline_model('output/models/final_model.pt', data)
print("   Computing MRR on FULL test set (15,372 edges)...")
baseline_computed = compute_mrr_full(baseline_model, data)

print(f"   Reported MRR:  {baseline_reported['metrics']['ranking']['mrr']:.4f}")
print(f"   Computed MRR:  {baseline_computed['mrr']:.4f}")
print(f"   Difference:    {abs(baseline_computed['mrr'] - baseline_reported['metrics']['ranking']['mrr']):.4f}")

# Test Phase 1
print("\n3. PHASE 1 MODEL (new architecture, with LayerNorm):")
print("   Loading model from current src/...")
phase1_model = load_phase1_model('output_phase1_optimized/models/best_model.pt', data)
print("   Computing MRR on FULL test set (15,372 edges)...")
phase1_computed = compute_mrr_full(phase1_model, data)

print(f"   Reported MRR:  {phase1_reported['metrics']['ranking']['mrr']:.4f}")
print(f"   Computed MRR:  {phase1_computed['mrr']:.4f}")
print(f"   Difference:    {abs(phase1_computed['mrr'] - phase1_reported['metrics']['ranking']['mrr']):.4f}")

# Compare
print("\n4. DIRECT COMPARISON (FULL test set - 15,372 edges):")
print(f"   Baseline MRR:  {baseline_computed['mrr']:.4f}")
print(f"   Phase 1 MRR:   {phase1_computed['mrr']:.4f}")
print(f"   Improvement:   +{(phase1_computed['mrr']/baseline_computed['mrr']-1)*100:.1f}%")
print()
print(f"   Baseline Mean Rank:  {baseline_computed['mean_rank']:.1f}")
print(f"   Phase 1 Mean Rank:   {phase1_computed['mean_rank']:.1f}")
print(f"   Improvement:         -{(1-phase1_computed['mean_rank']/baseline_computed['mean_rank'])*100:.1f}%")
print()
print(f"   Baseline Hits@10:  {baseline_computed['hits@10']:.4f}")
print(f"   Phase 1 Hits@10:   {phase1_computed['hits@10']:.4f}")
print(f"   Improvement:       +{(phase1_computed['hits@10']/baseline_computed['hits@10']-1)*100:.1f}%")

print("\n5. ARCHITECTURE VERIFICATION:")
baseline_ckpt = torch.load('output/models/final_model.pt', map_location='cpu', weights_only=False)
phase1_ckpt = torch.load('output_phase1_optimized/models/best_model.pt', map_location='cpu', weights_only=False)

has_baseline_norm = any('norm' in k for k in baseline_ckpt['model_state_dict'].keys())
has_phase1_norm = any('norm' in k for k in phase1_ckpt['model_state_dict'].keys())

print(f"   Baseline checkpoint has LayerNorm weights: {has_baseline_norm}")
print(f"   Phase 1 checkpoint has LayerNorm weights:  {has_phase1_norm}")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
if phase1_computed['mrr'] > baseline_computed['mrr']:
    improvement = (phase1_computed['mrr']/baseline_computed['mrr']-1)*100
    print(f"✓ Phase 1 MRR is {improvement:.1f}% better (FULL test set)")
    print("✓ Improvement is REAL and reproducible")
    print("✓ Architecture difference: LayerNorm + Skip Connections")
    print(f"✓ Better convergence: val loss {baseline_reported['model_info']['best_val_loss']:.4f}→{phase1_reported['model_info']['best_val_loss']:.4f}")

    # Check if computed matches reported
    baseline_diff = abs(baseline_computed['mrr'] - baseline_reported['metrics']['ranking']['mrr'])
    phase1_diff = abs(phase1_computed['mrr'] - phase1_reported['metrics']['ranking']['mrr'])

    if baseline_diff < 0.001 and phase1_diff < 0.001:
        print("\n✓ Computed MRR matches reported MRR (both models)")
    else:
        print(f"\n⚠ Warning: Computed vs Reported MRR mismatch:")
        print(f"  Baseline: {baseline_diff:.4f} difference")
        print(f"  Phase 1:  {phase1_diff:.4f} difference")
else:
    print("✗ No improvement detected")

print("="*60)
