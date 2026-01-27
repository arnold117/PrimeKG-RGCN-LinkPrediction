"""
Drug ID Mapping: PrimeKG (DrugBank) <-> LINCS L1000

This script creates mappings between PrimeKG drug IDs (DrugBank format)
and LINCS L1000 perturbagen names for downstream expression prediction.

Mapping strategies:
1. Direct name matching (case-insensitive)
2. PubChem API lookup for additional matches
3. Manual curation for common drugs

Usage:
    python src/drug_id_mapping.py --lincs-meta path/to/siginfo.txt
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time


def load_primekg_drugs(drug_info_path: str = "data/processed/drug_info.csv") -> pd.DataFrame:
    """Load PrimeKG drug information."""
    df = pd.read_csv(drug_info_path)
    # Normalize drug names for matching
    df['drug_name_lower'] = df['drug_name'].str.lower().str.strip()
    print(f"Loaded {len(df)} drugs from PrimeKG")
    return df


def load_lincs_metadata(lincs_meta_path: str) -> pd.DataFrame:
    """
    Load LINCS L1000 metadata.

    Expected file: siginfo_beta.txt or similar from CMap/LINCS
    Key columns: pert_iname, pert_type, cell_id, pert_dose, pert_time
    """
    print(f"Loading LINCS metadata from {lincs_meta_path}...")

    # Try different separators
    try:
        df = pd.read_csv(lincs_meta_path, sep='\t', low_memory=False)
    except:
        df = pd.read_csv(lincs_meta_path, low_memory=False)

    print(f"Loaded {len(df)} signatures from LINCS")
    print(f"Columns: {df.columns.tolist()[:10]}...")

    # Filter to compound perturbations only (not genetic perturbations)
    if 'pert_type' in df.columns:
        df_compounds = df[df['pert_type'] == 'trt_cp'].copy()
        print(f"Filtered to {len(df_compounds)} compound perturbations")
    else:
        df_compounds = df.copy()

    return df_compounds


def get_unique_lincs_drugs(lincs_df: pd.DataFrame) -> pd.DataFrame:
    """Get unique drugs from LINCS metadata."""

    # Get unique drug names
    if 'pert_iname' in lincs_df.columns:
        drug_col = 'pert_iname'
    elif 'cmap_name' in lincs_df.columns:
        drug_col = 'cmap_name'
    else:
        raise ValueError("Cannot find drug name column in LINCS data")

    unique_drugs = lincs_df[[drug_col]].drop_duplicates()
    unique_drugs = unique_drugs.rename(columns={drug_col: 'lincs_name'})
    unique_drugs['lincs_name_lower'] = unique_drugs['lincs_name'].str.lower().str.strip()

    # Add signature counts
    sig_counts = lincs_df[drug_col].value_counts().reset_index()
    sig_counts.columns = ['lincs_name', 'num_signatures']
    unique_drugs = unique_drugs.merge(sig_counts, on='lincs_name', how='left')

    print(f"Found {len(unique_drugs)} unique drugs in LINCS")
    return unique_drugs


def direct_name_matching(
    primekg_drugs: pd.DataFrame,
    lincs_drugs: pd.DataFrame
) -> pd.DataFrame:
    """Match drugs by exact name (case-insensitive)."""

    # Merge on lowercase names
    matched = primekg_drugs.merge(
        lincs_drugs,
        left_on='drug_name_lower',
        right_on='lincs_name_lower',
        how='inner'
    )

    print(f"Direct name matching: {len(matched)} matches")
    return matched


def fuzzy_name_matching(
    primekg_drugs: pd.DataFrame,
    lincs_drugs: pd.DataFrame,
    threshold: float = 0.85
) -> pd.DataFrame:
    """Match drugs using fuzzy string matching."""

    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        print("Warning: rapidfuzz not installed. Skipping fuzzy matching.")
        print("Install with: pip install rapidfuzz")
        return pd.DataFrame()

    # Get unmatched PrimeKG drugs
    lincs_names_lower = set(lincs_drugs['lincs_name_lower'])
    unmatched_primekg = primekg_drugs[
        ~primekg_drugs['drug_name_lower'].isin(lincs_names_lower)
    ]

    matches = []
    lincs_names_list = lincs_drugs['lincs_name_lower'].tolist()

    for _, row in unmatched_primekg.iterrows():
        result = process.extractOne(
            row['drug_name_lower'],
            lincs_names_list,
            scorer=fuzz.ratio
        )
        if result and result[1] >= threshold * 100:
            lincs_match = lincs_drugs[
                lincs_drugs['lincs_name_lower'] == result[0]
            ].iloc[0]
            matches.append({
                'drug_id': row['drug_id'],
                'drug_name': row['drug_name'],
                'lincs_name': lincs_match['lincs_name'],
                'match_score': result[1] / 100,
                'match_type': 'fuzzy'
            })

    if matches:
        fuzzy_df = pd.DataFrame(matches)
        print(f"Fuzzy matching: {len(fuzzy_df)} additional matches")
        return fuzzy_df

    return pd.DataFrame()


def pubchem_lookup(drug_name: str) -> Optional[Dict]:
    """
    Look up drug in PubChem to get additional identifiers.

    Returns dict with CID, synonyms, etc.
    """
    try:
        import pubchempy as pcp

        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            c = compounds[0]
            return {
                'pubchem_cid': c.cid,
                'iupac_name': c.iupac_name,
                'synonyms': c.synonyms[:10] if c.synonyms else []
            }
    except Exception as e:
        pass

    return None


def batch_pubchem_matching(
    unmatched_primekg: pd.DataFrame,
    lincs_drugs: pd.DataFrame,
    max_lookups: int = 100
) -> pd.DataFrame:
    """
    Use PubChem to find additional matches via synonyms.

    Rate limited to avoid API throttling.
    """
    try:
        import pubchempy as pcp
    except ImportError:
        print("Warning: pubchempy not installed. Skipping PubChem matching.")
        print("Install with: pip install pubchempy")
        return pd.DataFrame()

    matches = []
    lincs_names_lower = set(lincs_drugs['lincs_name_lower'])

    print(f"Running PubChem lookups for up to {max_lookups} drugs...")

    for i, (_, row) in enumerate(unmatched_primekg.iterrows()):
        if i >= max_lookups:
            break

        if i > 0 and i % 10 == 0:
            print(f"  Processed {i}/{min(len(unmatched_primekg), max_lookups)}...")
            time.sleep(1)  # Rate limiting

        result = pubchem_lookup(row['drug_name'])
        if result and result['synonyms']:
            # Check if any synonym matches LINCS
            for syn in result['synonyms']:
                syn_lower = syn.lower().strip()
                if syn_lower in lincs_names_lower:
                    lincs_match = lincs_drugs[
                        lincs_drugs['lincs_name_lower'] == syn_lower
                    ].iloc[0]
                    matches.append({
                        'drug_id': row['drug_id'],
                        'drug_name': row['drug_name'],
                        'lincs_name': lincs_match['lincs_name'],
                        'matched_via': syn,
                        'pubchem_cid': result['pubchem_cid'],
                        'match_type': 'pubchem'
                    })
                    break

    if matches:
        pubchem_df = pd.DataFrame(matches)
        print(f"PubChem matching: {len(pubchem_df)} additional matches")
        return pubchem_df

    return pd.DataFrame()


def create_mapping(
    primekg_path: str = "data/processed/drug_info.csv",
    lincs_meta_path: str = None,
    output_path: str = "data/processed/drug_id_mapping.csv",
    use_pubchem: bool = False,
    use_fuzzy: bool = True
) -> pd.DataFrame:
    """
    Create comprehensive drug ID mapping.

    Args:
        primekg_path: Path to PrimeKG drug info
        lincs_meta_path: Path to LINCS metadata (siginfo)
        output_path: Where to save the mapping
        use_pubchem: Whether to use PubChem API for additional matches
        use_fuzzy: Whether to use fuzzy string matching

    Returns:
        DataFrame with mapping
    """

    # Load PrimeKG drugs
    primekg_drugs = load_primekg_drugs(primekg_path)

    if lincs_meta_path is None:
        print("\nNo LINCS metadata provided.")
        print("To create mapping, download LINCS data and provide path.")
        print("\nFor now, saving PrimeKG drug list for reference...")
        primekg_drugs.to_csv(output_path.replace('.csv', '_primekg_only.csv'), index=False)
        return primekg_drugs

    # Load LINCS drugs
    lincs_df = load_lincs_metadata(lincs_meta_path)
    lincs_drugs = get_unique_lincs_drugs(lincs_df)

    # Strategy 1: Direct name matching
    direct_matches = direct_name_matching(primekg_drugs, lincs_drugs)
    direct_matches['match_type'] = 'exact'

    all_matches = [direct_matches]
    matched_primekg_ids = set(direct_matches['drug_id'])

    # Strategy 2: Fuzzy matching
    if use_fuzzy:
        unmatched = primekg_drugs[~primekg_drugs['drug_id'].isin(matched_primekg_ids)]
        fuzzy_matches = fuzzy_name_matching(unmatched, lincs_drugs)
        if len(fuzzy_matches) > 0:
            all_matches.append(fuzzy_matches)
            matched_primekg_ids.update(fuzzy_matches['drug_id'])

    # Strategy 3: PubChem lookup
    if use_pubchem:
        unmatched = primekg_drugs[~primekg_drugs['drug_id'].isin(matched_primekg_ids)]
        pubchem_matches = batch_pubchem_matching(unmatched, lincs_drugs)
        if len(pubchem_matches) > 0:
            all_matches.append(pubchem_matches)

    # Combine all matches
    final_mapping = pd.concat(all_matches, ignore_index=True)

    # Select relevant columns
    cols_to_keep = ['drug_id', 'drug_name', 'lincs_name', 'match_type']
    cols_to_keep = [c for c in cols_to_keep if c in final_mapping.columns]
    if 'num_signatures' in final_mapping.columns:
        cols_to_keep.append('num_signatures')

    final_mapping = final_mapping[cols_to_keep].drop_duplicates(subset=['drug_id'])

    # Summary
    print("\n" + "="*50)
    print("MAPPING SUMMARY")
    print("="*50)
    print(f"Total PrimeKG drugs: {len(primekg_drugs)}")
    print(f"Total LINCS drugs: {len(lincs_drugs)}")
    print(f"Matched drugs: {len(final_mapping)}")
    print(f"Coverage: {len(final_mapping)/len(primekg_drugs)*100:.1f}%")
    print("\nMatches by type:")
    print(final_mapping['match_type'].value_counts())
    print("="*50)

    # Save
    final_mapping.to_csv(output_path, index=False)
    print(f"\nSaved mapping to {output_path}")

    return final_mapping


def analyze_coverage(mapping_path: str = "data/processed/drug_id_mapping.csv"):
    """Analyze the drug mapping coverage."""

    mapping = pd.read_csv(mapping_path)
    primekg = pd.read_csv("data/processed/drug_info.csv")

    print("\n" + "="*50)
    print("COVERAGE ANALYSIS")
    print("="*50)

    print(f"\nMapped drugs: {len(mapping)}/{len(primekg)} ({len(mapping)/len(primekg)*100:.1f}%)")

    if 'num_signatures' in mapping.columns:
        print(f"\nTotal LINCS signatures available: {mapping['num_signatures'].sum():,}")
        print(f"Mean signatures per drug: {mapping['num_signatures'].mean():.1f}")

    # Show sample mappings
    print("\nSample mappings:")
    for _, row in mapping.head(10).iterrows():
        print(f"  {row['drug_id']}: {row['drug_name']} -> {row['lincs_name']}")


def main():
    parser = argparse.ArgumentParser(description='Create drug ID mapping')
    parser.add_argument('--lincs-meta', type=str, default=None,
                        help='Path to LINCS metadata file (siginfo)')
    parser.add_argument('--primekg', type=str, default='data/processed/drug_info.csv',
                        help='Path to PrimeKG drug info')
    parser.add_argument('--output', type=str, default='data/processed/drug_id_mapping.csv',
                        help='Output path for mapping')
    parser.add_argument('--use-pubchem', action='store_true',
                        help='Use PubChem API for additional matches')
    parser.add_argument('--use-fuzzy', action='store_true', default=True,
                        help='Use fuzzy string matching')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze existing mapping')

    args = parser.parse_args()

    if args.analyze:
        analyze_coverage(args.output)
    else:
        create_mapping(
            primekg_path=args.primekg,
            lincs_meta_path=args.lincs_meta,
            output_path=args.output,
            use_pubchem=args.use_pubchem,
            use_fuzzy=args.use_fuzzy
        )


if __name__ == "__main__":
    main()
