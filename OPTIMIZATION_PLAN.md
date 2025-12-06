# Comprehensive Optimization Plan for PrimeKG-RGCN-LinkPrediction

## Executive Summary

**Goal:** Optimize the PrimeKG-RGCN system to predict how genetic mutations drive adverse drug reactions, while improving speed and accuracy.

**Current Performance:**
- AUC-ROC: 0.9781 | Hits@10: 0.0410 | MRR: 0.0187 | F1: 0.9526
- Training time: ~4-5 hours on RTX 1070
- Graph: 30,926 nodes, 849,456 edges (3 relation types)

**Target Improvements:**
- **Speed:** 5-10× faster training, 10-15× faster evaluation
- **Accuracy:** +5-10% Hits@10, +20-50% MRR, +0.5-1% AUC-ROC
- **Capability:** Add genetic variant → adverse reaction prediction

---

## Part 1: Speed Optimization (Priority 1)

### Critical Bottlenecks Identified

**1. Full Graph Encoding Every Batch (CRITICAL)**
- **Location:** `src/train.py:291-297`, `src/evaluate.py:189-195`
- **Issue:** Model encodes ALL 30,926 nodes for EVERY batch
- **Impact:** ~100,000 unnecessary full graph encodings during training
- **Expected Speedup:** 10× for evaluation, 5× for training

**2. Redundant Ranking Computation**
- **Location:** `src/evaluate.py:251-276`
- **Issue:** Re-encodes graph for each ranking batch + Python loops
- **Expected Speedup:** 15× for ranking metrics

**3. Memory Management Overhead**
- **Location:** `src/train.py:336-342`
- **Issue:** Explicit tensor deletion and frequent `torch.cuda.empty_cache()` calls
- **Expected Speedup:** 2-3% reduction in overhead

### Speed Optimization Implementation Plan

#### **Task 1.1: Embedding Cache for Evaluation** ⭐ HIGHEST IMPACT
**Files to modify:** `src/evaluate.py`

**Changes:**
1. Line 251-254: Add embedding cache before ranking loop
2. Line 266-276: Vectorize ranking computation (remove Python loop)

**Code Pattern:**
```python
# Before ranking loop (once):
all_node_embeddings = model.encoder(full_edge_index, full_edge_type)

# In ranking loop (reuse cached embeddings):
head_emb = all_node_embeddings[batch_head]
scores = model.decoder.score_all_tails(head_emb, batch_rel, all_node_embeddings)

# Vectorized ranking (replace loop):
sorted_scores, sorted_indices = torch.sort(scores, descending=True)
ranks = (sorted_indices == batch_tail.unsqueeze(1)).nonzero()[:, 1] + 1
```

**Expected Result:** Evaluation time: 300s → 20s (15× speedup)

---

#### **Task 1.2: Mixed Precision Training** ⭐ HIGH IMPACT
**Files to modify:** `src/train.py`

**Changes:**
1. Line 175: Initialize GradScaler
2. Line 269-310: Wrap forward pass with `autocast()`
3. Line 311-318: Replace backward() with scaler

**Code Pattern:**
```python
from torch.cuda.amp import autocast, GradScaler

# In __init__ (line 175):
self.scaler = GradScaler()

# In train_epoch (line 269):
with autocast():
    scores = self.model(...)
    loss = self.criterion(scores, labels)

# Replace backward:
self.scaler.scale(loss).backward()
self.scaler.step(self.optimizer)
self.scaler.update()
```

**Expected Result:**
- Training time: 4-5 hours → 2.5-3 hours (1.5-2× speedup)
- Memory usage: -30-40%

---

#### **Task 1.3: Remove Inefficient Memory Management**
**Files to modify:** `src/train.py`

**Changes:**
1. Line 336-342: Remove explicit tensor deletions and cache clearing

**Code Pattern:**
```python
# DELETE these lines:
del all_heads, all_tails, all_rels, all_labels
del pos_scores, neg_scores, scores, labels, loss

if (batch_idx + 1) % 50 == 0:
    torch.cuda.empty_cache()
```

**Expected Result:** 2-3% speedup, cleaner code

---

#### **Task 1.4: Improved Negative Sampling**
**Files to modify:** `src/train.py`

**Changes:**
1. Line 59-97: Add collision checking and hard negative mining

**Code Pattern:**
```python
def sample(self, pos_head, pos_tail, pos_rel):
    # Generate candidates
    random_entities = torch.randint(0, self.num_nodes, (total_neg * 2,))

    # Remove collisions with positive samples
    positive_set = torch.cat([pos_head, pos_tail])
    mask = ~torch.isin(random_entities, positive_set)
    valid_negatives = random_entities[mask][:total_neg]

    # Apply corruption
    ...
```

**Expected Result:** 5-10% faster convergence, better training quality

---

## Part 2: Model Architecture Improvements (Priority 2)

### Current Architecture Limitations

**Encoder:** 2-layer RGCN (only 2-hop neighborhoods)
**Decoder:** DistMult (symmetric, can't capture directional relationships)
**Issues:** Low ranking metrics (Hits@10: 4.1%, MRR: 1.87%)

### Architecture Enhancement Plan

#### **Task 2.1: Add Skip Connections & Layer Normalization** ⭐ FOUNDATION
**Files to modify:** `src/models/rgcn.py`

**Changes:**
1. Line 70-90: Add LayerNorm modules
2. Line 122-130: Add skip connections in forward pass

**Code Pattern:**
```python
# In __init__:
self.norm1 = nn.LayerNorm(hidden_dim)
self.norm2 = nn.LayerNorm(hidden_dim)

# In forward:
x = self.conv1(x, edge_index, edge_type)
x = self.norm1(x)
x = F.relu(x)
x = self.dropout(x)

x_prev = x  # Store for skip connection
x = self.conv2(x, edge_index, edge_type)
x = self.norm2(x)
x = x + x_prev  # Skip connection
```

**Expected Result:** +3-5% AUC, enables deeper networks

---

#### **Task 2.2: Extend to 3-4 Layer RGCN**
**Files to modify:** `src/models/rgcn.py`

**Changes:**
1. Line 70-85: Add conv3 and conv4 layers
2. Line 122-130: Extend forward pass with new layers

**Code Pattern:**
```python
# Add layers:
self.conv3 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases)
self.conv4 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases)
self.norm3 = nn.LayerNorm(hidden_dim)
self.norm4 = nn.LayerNorm(hidden_dim)

# Forward with skip connections every 2 layers
```

**Expected Result:** +5-10% Hits@10, +10-20% MRR (better multi-hop reasoning)

---

#### **Task 2.3: Implement RotatE Decoder Alternative**
**Files to create:** `src/models/decoders.py`

**New decoder class:**
```python
class RotatEDecoder(nn.Module):
    """Rotation-based scoring for asymmetric relations"""
    def forward(self, head_emb, tail_emb, relation_types):
        # Relation as rotation in complex space
        rel_emb = self.relation_embeddings(relation_types)
        # score = ||h ⊙ r - t||
        rotated = head_emb * rel_emb
        score = -torch.norm(rotated - tail_emb, p=2, dim=1)
        return score
```

**Changes to existing files:**
1. `src/models/rgcn.py`: Add option to use RotatE decoder
2. `src/train.py`: Add `--decoder` argument

**Expected Result:** +5-8% MRR, +2-3% AUC (better relation composition)

---

## Part 3: PharmGKB Integration for Adverse Reactions (Priority 3)

### Integration Strategy

**Goal:** Add genetic variants to enable mutation → adverse reaction prediction

**New Graph Elements:**
- **Node type:** `variant` (5,000-15,000 genetic variants)
- **Relation types:**
  - `variant-gene` (which gene has this variant)
  - `drug-variant` (which drugs affected by variant)
  - `variant-disease` (which diseases associated with variant)

**Expected Graph Size:** +10-15% nodes, +2-10% edges (manageable)

### PharmGKB Integration Plan

#### **Task 3.1: PharmGKB Data Acquisition & Preparation**
**New files to create:** `scripts/download_pharmgkb.py`, `data/raw/pharmgkb_variants.csv`

**Steps:**
1. Download PharmGKB clinical annotations (https://www.pharmgkb.org/downloads)
2. Extract variant-gene relationships
3. Extract drug-variant pharmacogenomic associations
4. Extract variant-disease associations (adverse reactions)
5. Create unified CSV format matching PrimeKG schema

**CSV Format:**
```
x_id,x_type,x_name,y_id,y_type,y_name,relation,x_source,y_source
CYP3A4,gene/protein,CYP3A4,rs2242480,variant,CYP3A4*1B,has_variant,NCBI,PharmGKB
DB00002,drug,Cetuximab,rs2242480,variant,CYP3A4*1B,affects_variant,DrugBank,PharmGKB
rs2242480,variant,CYP3A4*1B,C0020538,disease,Hypertension,variant-disease,PharmGKB,MeSH
```

---

#### **Task 3.2: Preprocessing Pipeline Updates**
**Files to modify:** `src/preprocess.py`

**Changes:**

1. **Line 57:** Update node types
```python
self.target_node_types = {'drug', 'gene/protein', 'disease', 'variant'}
```

2. **Lines 61-65:** Update relation types
```python
self.target_relations = {
    'drug-gene': ['drug_protein'],
    'gene-gene': ['protein_protein'],
    'gene-disease': ['disease_protein'],
    'variant-gene': ['has_variant'],
    'drug-variant': ['affects_variant', 'pharmacokinetic', 'pharmacodynamic'],
    'variant-disease': ['causes_disease', 'associated_with', 'variant_disease'],
}
```

3. **After line 98:** Add PharmGKB data loading
```python
def load_pharmgkb_data(self, pharmgkb_path: str) -> pd.DataFrame:
    """Load and merge PharmGKB variant data"""
    pharmgkb_df = pd.read_csv(pharmgkb_path)
    # Validate and merge with PrimeKG data
    return pharmgkb_df
```

4. **Line 76-98:** Modify `load_data()` to merge PharmGKB
```python
def load_data(self, pharmgkb_path: Optional[str] = None):
    df = pd.read_csv(self.raw_data_path)

    if pharmgkb_path:
        pharmgkb_df = self.load_pharmgkb_data(pharmgkb_path)
        df = pd.concat([df, pharmgkb_df], ignore_index=True)

    return df
```

**Expected Result:**
- Graph grows to ~40K nodes, ~900K edges
- 6 relation types (from 3)
- Enables variant-based predictions

---

#### **Task 3.3: Model Updates for Variants**
**Files to modify:** `src/train.py`, `src/models/rgcn.py`

**No code changes needed!** The architecture is already parameterized:
```python
DrugDiseaseModel(
    num_nodes=40000,      # Automatically handles more nodes
    num_relations=6,      # Automatically handles more relations
    ...
)
```

**Only change:** Update model instantiation to use new graph stats

---

#### **Task 3.4: New Analysis Scripts for Adverse Reactions**
**Files to create:** `src/predict_adverse_reactions.py`

**Functionality:**
```python
def predict_adverse_reactions(drug_name, variant_id, top_k=10):
    """
    Predict adverse reactions for drug-variant combination

    Returns:
    - Top-K predicted diseases/adverse reactions
    - Confidence scores
    - Explanation paths: drug → variant → gene → disease
    """
    # Use trained model to rank disease nodes
    # Filter for adverse reaction disease types
    # Provide mechanistic explanations
```

**Example Usage:**
```bash
python src/predict_adverse_reactions.py \
    --drug "Warfarin" \
    --variant "CYP2C9*2" \
    --top_k 10
```

**Expected Output:**
```
Top Adverse Reactions for Warfarin + CYP2C9*2:
1. Bleeding (score: 0.92) - Path: Warfarin → CYP2C9*2 → CYP2C9 → Coagulation → Bleeding
2. Hemorrhage (score: 0.89) - Path: Warfarin → CYP2C9*2 → CYP2C9 → INR elevation → Hemorrhage
...
```

---

## Part 4: Implementation Timeline

### Phase 1: Quick Wins (Week 1-2)
**Goal:** Achieve 5-10× speedup with minimal risk

- [ ] Task 1.1: Embedding cache for evaluation (2 hours)
- [ ] Task 1.2: Mixed precision training (3 hours)
- [ ] Task 1.3: Remove inefficient memory management (1 hour)
- [ ] Task 2.1: Add skip connections & LayerNorm (4 hours)
- [ ] Benchmark and validate improvements

**Expected Result:**
- Evaluation: 300s → 20s
- Training: 5h → 3h
- AUC: +3-5%

---

### Phase 2: Architecture Enhancement (Week 2-3)
**Goal:** Improve ranking metrics significantly

- [ ] Task 2.2: Extend to 3-4 layer RGCN (4 hours)
- [ ] Task 1.4: Improved negative sampling (3 hours)
- [ ] Task 2.3: Implement RotatE decoder (6 hours)
- [ ] Retrain and compare models
- [ ] Select best architecture

**Expected Result:**
- Hits@10: 0.041 → 0.055-0.065 (+35-60%)
- MRR: 0.019 → 0.025-0.030 (+30-60%)
- AUC: +2-3% additional

---

### Phase 3: PharmGKB Integration (Week 3-5)
**Goal:** Enable adverse reaction prediction

- [ ] Task 3.1: Download and prepare PharmGKB data (1 week)
  - [ ] Acquire PharmGKB access (user will assist)
  - [ ] Extract clinical annotations
  - [ ] Create variant CSV files
  - [ ] Validate data quality
- [ ] Task 3.2: Update preprocessing pipeline (8 hours)
- [ ] Task 3.3: Retrain model on augmented graph (5 hours)
- [ ] Task 3.4: Create adverse reaction prediction scripts (8 hours)
- [ ] Validate predictions with known pharmacogenomic associations

**Expected Result:**
- New capability: drug-variant → adverse reaction prediction
- Improved coverage of pharmacogenomic space
- Real-world clinical utility

---

### Phase 4: Advanced Optimizations (Week 5-6, Optional)
**Goal:** Further refinement and deployment preparation

- [ ] Attention mechanisms (R-GAT)
- [ ] Neighbor sampling for scalability
- [ ] Model ensemble approaches
- [ ] API deployment for predictions

---

## Critical Files Reference

### Files to Modify

| File | Tasks | Priority | Complexity |
|------|-------|----------|-----------|
| `src/evaluate.py` | 1.1 | P1 | Low |
| `src/train.py` | 1.2, 1.3, 1.4 | P1 | Low-Med |
| `src/models/rgcn.py` | 2.1, 2.2 | P2 | Medium |
| `src/preprocess.py` | 3.2 | P3 | Medium |

### Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `src/models/decoders.py` | RotatE decoder | P2 |
| `scripts/download_pharmgkb.py` | Data acquisition | P3 |
| `src/predict_adverse_reactions.py` | Adverse reaction prediction | P3 |
| `data/raw/pharmgkb_variants.csv` | PharmGKB data | P3 |

---

## Expected Final Performance

**Current Baseline:**
- AUC-ROC: 0.9781 | Hits@10: 0.0410 | MRR: 0.0187
- Training: 4-5 hours | Evaluation: 300s

**After All Optimizations:**
- AUC-ROC: 0.9850-0.9900 (+0.7-1.2%)
- Hits@10: 0.0550-0.0700 (+35-70%)
- MRR: 0.0280-0.0400 (+50-115%)
- Training: 2-3 hours (-40-50%)
- Evaluation: 15-20s (-93-95%)

**New Capabilities:**
- Predict adverse drug reactions from genetic variants
- Explain predictions through variant-gene-disease pathways
- Support ~5,000-15,000 pharmacogenomic variants
- Clinical decision support for personalized medicine

---

## Risk Mitigation

**Risks:**
1. PharmGKB data access/licensing
2. Architecture changes may require hyperparameter tuning
3. Sparse variant data may not improve predictions

**Mitigation:**
1. Start with public subset, user will help with access
2. Maintain baseline model for comparison
3. Validate on known pharmacogenomic associations first
4. Incremental integration (start with high-quality variants)

---

## Success Metrics

**Phase 1 Success:** ✅ 5× training speedup, 10× eval speedup
**Phase 2 Success:** ✅ Hits@10 > 0.055, MRR > 0.025
**Phase 3 Success:** ✅ Successfully predict known drug-variant adverse reactions with >70% accuracy

---

**Status:** Ready for implementation. Awaiting user approval.
