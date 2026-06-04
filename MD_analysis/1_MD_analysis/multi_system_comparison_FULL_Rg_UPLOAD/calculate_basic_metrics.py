#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to calculate basic metrics (Mean, Std, CV) for all replica data files.
"""

import os
import pandas as pd
import numpy as np

SYSTEMS = ["CrtM-FSPP", "CrtM-GGPP", "PSY-FSPP", "PSY-GGPP"]
REPLICA_IDS = ["MD", "MD2", "MD3"]
EQUIL_SKIP_FRAC = 0.10

def load_numeric_data(path, skip_frac=0.10):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        # Drop 'Frame' if present
        if 'Frame' in df.columns:
            df = df.drop(columns=['Frame'])
        # Drop non-numeric columns
        df = df.select_dtypes(include=[np.number])
        # Skip equilibration
        skip = int(len(df) * skip_frac)
        return df.iloc[skip:]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def main():
    rows = []
    
    # We will search for CSV files in each system folder starting with replica prefixes
    for system in SYSTEMS:
        if not os.path.isdir(system):
            continue
        
        # List all files in the system directory
        files = os.listdir(system)
        for replica in REPLICA_IDS:
            # Find files matching replica prefix, e.g. "MD_main_metrics.csv", "MD_pca_rg.csv", etc.
            replica_files = [f for f in files if f.startswith(f"{replica}_") and f.endswith(".csv")]
            
            for fname in sorted(replica_files):
                # Extract file type (e.g. "main_metrics", "pca_rg", "hb_counts", "mg_distances")
                # Remove replica prefix and extension
                source_type = fname[len(replica)+1:-4]
                
                path = os.path.join(system, fname)
                df = load_numeric_data(path, EQUIL_SKIP_FRAC)
                if df is None or df.empty:
                    continue
                
                for col in df.columns:
                    vals = df[col].dropna().values
                    if len(vals) == 0:
                        continue
                    
                    mean_val = np.mean(vals)
                    std_val = np.std(vals)
                    sem_val = std_val / np.sqrt(len(vals))
                    min_val = np.min(vals)
                    max_val = np.max(vals)
                    median_val = np.median(vals)
                    # If mean is very close to 0, CV is not meaningful
                    if abs(mean_val) > 1e-9:
                        cv_val = (std_val / mean_val) * 100.0
                    else:
                        cv_val = np.nan
                        
                    rows.append({
                        'System': system,
                        'Replica': replica,
                        'Source': source_type,
                        'Metric': col,
                        'Mean': mean_val,
                        'Std': std_val,
                        'SEM': sem_val,
                        'Min': min_val,
                        'Max': max_val,
                        'Median': median_val,
                        'CV (%)': cv_val
                    })
                    
    if not rows:
        print("No data found to process.")
        return
        
    df_out = pd.DataFrame(rows)
    
    # Save to CSV
    output_csv = "basic_metrics_summary.csv"
    df_out.to_csv(output_csv, index=False)
    print(f"Successfully calculated basic metrics and saved to: {output_csv}\n")
    
    # Format and print to console
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    
    print("=" * 120)
    print("BASIC DESCRIPTIVE METRICS SUMMARY")
    print("=" * 120)
    print(df_out.to_string(index=False))
    print("=" * 120)

if __name__ == "__main__":
    main()
