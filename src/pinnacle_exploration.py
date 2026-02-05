"""
PINNACLE Data Exploration and Integration Assessment

Analyzes PINNACLE embeddings and evaluates integration with PrimeKG:
1. Data format and structure
2. Protein ID overlap with PrimeKG
3. Cell type coverage
4. Integration feasibility

Usage:
    python src/pinnacle_exploration.py
"""

import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import pickle

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


PINNACLE_DIR = Path("data/raw/pinnacle")


def explore_directory():
    """List all files in PINNACLE directory."""
    print("="*60)
    print("PINNACLE Directory Contents")
    print("="*60)

    if not PINNACLE_DIR.exists():
        print(f"Directory not found: {PINNACLE_DIR}")
        return []

    files = list(PINNACLE_DIR.glob("**/*"))
    files = [f for f in files if f.is_file()]

    print(f"\nFound {len(files)} files:")
    for f in sorted(files)[:50]:  # Limit output
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"  {f.relative_to(PINNACLE_DIR)} ({size_mb:.1f} MB)")

    return files


def load_pinnacle_embeddings(emb_path: Path = None):
    """Load PINNACLE embeddings - try various formats."""
    print("\n" + "="*60)
    print("Loading PINNACLE Embeddings")
    print("="*60)

    # Try to find embedding files
    possible_paths = [
        PINNACLE_DIR / "embeddings.pt",
        PINNACLE_DIR / "embeddings.pkl",
        PINNACLE_DIR / "protein_embeddings.pt",
        PINNACLE_DIR / "cell_type_embeddings.pt",
    ]

    # Also check for .npy files
    npy_files = list(PINNACLE_DIR.glob("*.npy"))
    pt_files = list(PINNACLE_DIR.glob("*.pt"))
    pkl_files = list(PINNACLE_DIR.glob("*.pkl"))
    h5_files = list(PINNACLE_DIR.glob("*.h5"))
    csv_files = list(PINNACLE_DIR.glob("*.csv"))
    tsv_files = list(PINNACLE_DIR.glob("*.tsv"))
    txt_files = list(PINNACLE_DIR.glob("*.txt"))

    print(f"\nFile types found:")
    print(f"  .pt files: {len(pt_files)}")
    print(f"  .pkl files: {len(pkl_files)}")
    print(f"  .npy files: {len(npy_files)}")
    print(f"  .h5 files: {len(h5_files)}")
    print(f"  .csv files: {len(csv_files)}")
    print(f"  .tsv files: {len(tsv_files)}")
    print(f"  .txt files: {len(txt_files)}")

    # Try loading first .pt file if exists
    if pt_files:
        print(f"\nTrying to load: {pt_files[0]}")
        try:
            data = torch.load(pt_files[0], weights_only=False)
            print(f"  Type: {type(data)}")
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())[:10]}")
            elif isinstance(data, torch.Tensor):
                print(f"  Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"  Error: {e}")

    # Try loading first .pkl file
    if pkl_files:
        print(f"\nTrying to load: {pkl_files[0]}")
        try:
            with open(pkl_files[0], 'rb') as f:
                data = pickle.load(f)
            print(f"  Type: {type(data)}")
            if isinstance(data, dict):
                print(f"  Keys: {list(data.keys())[:10]}")
            return data
        except Exception as e:
            print(f"  Error: {e}")

    # Try loading CSV
    if csv_files:
        print(f"\nTrying to load: {csv_files[0]}")
        try:
            df = pd.read_csv(csv_files[0], nrows=5)
            print(f"  Columns: {df.columns.tolist()}")
            print(f"  Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"  Error: {e}")

    print("\nNo loadable files found. Please check the data format.")
    return None


def get_primekg_protein_ids():
    """Get protein/gene IDs from PrimeKG."""
    print("\n" + "="*60)
    print("Loading PrimeKG Protein IDs")
    print("="*60)

    mappings = torch.load('data/processed/mappings.pt', weights_only=False)
    idx2node = mappings['idx2node']

    # Collect gene/protein IDs
    protein_ids = set()
    protein_names = set()

    for idx, (node_id, node_name, node_type) in idx2node.items():
        if node_type == 'gene/protein':
            protein_ids.add(str(node_id))
            protein_names.add(node_name)

    print(f"PrimeKG proteins: {len(protein_ids)}")
    print(f"Sample IDs: {list(protein_ids)[:5]}")
    print(f"Sample names: {list(protein_names)[:5]}")

    return protein_ids, protein_names


def check_id_overlap(pinnacle_ids, primekg_ids, primekg_names):
    """Check overlap between PINNACLE and PrimeKG protein IDs."""
    print("\n" + "="*60)
    print("ID Overlap Analysis")
    print("="*60)

    if pinnacle_ids is None:
        print("No PINNACLE IDs available for comparison.")
        return

    pinnacle_set = set(str(i) for i in pinnacle_ids)

    # Check direct ID match
    id_overlap = pinnacle_set & primekg_ids
    print(f"\nDirect ID overlap: {len(id_overlap)}/{len(pinnacle_set)} "
          f"({len(id_overlap)/len(pinnacle_set)*100:.1f}%)")

    # Check name match
    name_overlap = pinnacle_set & primekg_names
    print(f"Name overlap: {len(name_overlap)}/{len(pinnacle_set)} "
          f"({len(name_overlap)/len(pinnacle_set)*100:.1f}%)")

    # Show samples
    if id_overlap:
        print(f"\nSample matching IDs: {list(id_overlap)[:5]}")
    if name_overlap:
        print(f"Sample matching names: {list(name_overlap)[:5]}")


def main():
    """Run PINNACLE exploration."""

    print("="*60)
    print("PINNACLE Data Exploration")
    print("="*60)

    # Explore directory
    files = explore_directory()

    if not files:
        print("\nNo files found. Please download PINNACLE data to data/raw/pinnacle/")
        print("\nDownload links:")
        print("  GitHub: https://github.com/mims-harvard/PINNACLE")
        print("  Figshare: https://figshare.com/projects/PINNACLE/...")
        return

    # Try loading embeddings
    pinnacle_data = load_pinnacle_embeddings()

    # Get PrimeKG proteins
    primekg_ids, primekg_names = get_primekg_protein_ids()

    # If we have PINNACLE data, extract IDs and check overlap
    pinnacle_ids = None
    if pinnacle_data is not None:
        if isinstance(pinnacle_data, dict):
            # Try to extract protein IDs from keys
            pinnacle_ids = list(pinnacle_data.keys())[:100]  # Sample
            print(f"\nSample PINNACLE keys: {pinnacle_ids[:5]}")

    check_id_overlap(pinnacle_ids, primekg_ids, primekg_names)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Files found: {len(files)}")
    print(f"PrimeKG proteins: {len(primekg_ids)}")

    if pinnacle_data is not None:
        print("PINNACLE data loaded successfully")
    else:
        print("PINNACLE data needs to be downloaded")


if __name__ == "__main__":
    main()
