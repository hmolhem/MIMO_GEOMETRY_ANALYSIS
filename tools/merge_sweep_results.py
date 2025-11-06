"""
Merge all ALSS sweep CSV files into a single all_runs.csv file.
"""
import pandas as pd
from pathlib import Path

def main():
    results_dir = Path('results/alss')
    
    # Find all CSV files except all_runs.csv
    csv_files = [f for f in results_dir.glob('*.csv') if f.name != 'all_runs.csv']
    
    if not csv_files:
        print("❌ No CSV files found to merge!")
        return
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    for f in sorted(csv_files):
        print(f"  - {f.name}")
    
    # Load and concatenate all CSVs
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    # Merge all
    merged = pd.concat(dfs, ignore_index=True)
    
    # Save
    output_path = results_dir / 'all_runs.csv'
    merged.to_csv(output_path, index=False)
    
    print(f"\n✅ Merged {len(dfs)} files → {output_path}")
    print(f"   Total trials: {len(merged)}")
    print(f"   SNR values: {sorted(merged['SNR_dB'].unique())}")
    print(f"   Delta values: {sorted(merged['delta_deg'].unique())}")
    print(f"   ALSS modes: {merged['alss'].unique()}")

if __name__ == '__main__':
    main()
