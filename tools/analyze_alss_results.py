#!/usr/bin/env python3
"""
Analyze ALSS benchmark results and generate summary table.

Usage:
    python tools/analyze_alss_results.py results/bench/alss_off_sweep.csv results/bench/alss_on_sweep.csv

Outputs:
    - Summary table (mean RMSE, resolve %, etc.) by SNR
    - Comparison showing ALSS improvement
"""

import sys
import pandas as pd
import numpy as np

def analyze_sweep(csv_path):
    """Load and analyze a single sweep CSV."""
    df = pd.read_csv(csv_path)
    
    # Group by SNR
    results = []
    for snr in sorted(df['SNR_dB'].unique()):
        subset = df[df['SNR_dB'] == snr]
        n_trials = len(subset)
        rmse_mean = subset['rmse_deg'].mean()
        rmse_std = subset['rmse_deg'].std()
        resolve_rate = (subset['resolved'].sum() / n_trials) * 100
        runtime_mean = subset['runtime_ms'].mean()
        
        results.append({
            'SNR_dB': snr,
            'trials': n_trials,
            'rmse_mean': rmse_mean,
            'rmse_std': rmse_std,
            'resolve_%': resolve_rate,
            'runtime_ms': runtime_mean
        })
    
    return pd.DataFrame(results)

def main():
    if len(sys.argv) < 3:
        print("Usage: python analyze_alss_results.py <baseline_csv> <alss_csv>")
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    alss_path = sys.argv[2]
    
    print("\n" + "="*70)
    print("ALSS BENCHMARK ANALYSIS")
    print("="*70 + "\n")
    
    # Analyze both sweeps
    baseline_df = analyze_sweep(baseline_path)
    alss_df = analyze_sweep(alss_path)
    
    # Display baseline
    print("BASELINE (ALSS OFF):")
    print(baseline_df.to_string(index=False))
    print()
    
    # Display ALSS results
    print("ALSS ON (mode=zero, tau=1.0, coreL=3):")
    print(alss_df.to_string(index=False))
    print()
    
    # Compute improvement
    print("IMPROVEMENT (ALSS vs Baseline):")
    print("-" * 70)
    comparison = pd.DataFrame({
        'SNR_dB': baseline_df['SNR_dB'],
        'RMSE_baseline': baseline_df['rmse_mean'],
        'RMSE_alss': alss_df['rmse_mean'],
        'RMSE_reduction': baseline_df['rmse_mean'] - alss_df['rmse_mean'],
        'RMSE_improve_%': ((baseline_df['rmse_mean'] - alss_df['rmse_mean']) / baseline_df['rmse_mean'] * 100),
        'Resolve_baseline_%': baseline_df['resolve_%'],
        'Resolve_alss_%': alss_df['resolve_%'],
        'Resolve_gain': alss_df['resolve_%'] - baseline_df['resolve_%']
    })
    print(comparison.to_string(index=False))
    print()
    
    # Key findings
    print("KEY FINDINGS:")
    print("-" * 70)
    max_improve_idx = comparison['RMSE_improve_%'].idxmax()
    max_improve = comparison.loc[max_improve_idx]
    print(f"Maximum RMSE improvement: {max_improve['RMSE_improve_%']:.1f}% at SNR={max_improve['SNR_dB']}dB")
    print(f"  Baseline RMSE: {max_improve['RMSE_baseline']:.2f}°")
    print(f"  ALSS RMSE:     {max_improve['RMSE_alss']:.2f}°")
    print(f"  Reduction:     {max_improve['RMSE_reduction']:.2f}°")
    print()
    
    # Resolve improvement
    max_resolve_idx = comparison['Resolve_gain'].idxmax()
    max_resolve = comparison.loc[max_resolve_idx]
    if max_resolve['Resolve_gain'] > 0:
        print(f"Maximum resolve rate gain: {max_resolve['Resolve_gain']:.1f}% at SNR={max_resolve['SNR_dB']}dB")
        print(f"  Baseline: {max_resolve['Resolve_baseline_%']:.1f}%")
        print(f"  ALSS:     {max_resolve['Resolve_alss_%']:.1f}%")
        print()
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
