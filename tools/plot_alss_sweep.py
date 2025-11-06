#!/usr/bin/env python3
"""
Plot ALSS results from comprehensive SNR sweep.

Usage:
    python tools/plot_alss_sweep.py results/alss/all_runs.csv

Generates:
    - RMSE vs SNR for each delta value
    - Resolve rate vs SNR for each delta value
    - Combined comparison plots
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Publication-quality styling
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.linewidth': 1.2,
    'legend.fontsize': 10,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '#cccccc',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'figure.figsize': (4.5, 3.2),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
})

color_baseline = '#E74C3C'  # Red
color_alss = '#3498DB'      # Blue

def load_sweep_data(csv_path):
    """Load and parse sweep CSV data."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    
    # Parse ALSS flag
    if 'alss' in df.columns:
        df['alss_flag'] = df['alss'].astype(str).str.upper()
        df.loc[df['alss_flag'].isin(['TRUE', '1', 'YES', 'ON']), 'alss_flag'] = 'ON'
        df.loc[df['alss_flag'].isin(['FALSE', '0', 'NO', 'OFF']), 'alss_flag'] = 'OFF'
    else:
        df['alss_flag'] = 'OFF'
    
    return df

def plot_rmse_by_delta(df, output_dir):
    """Plot RMSE vs SNR for each delta value."""
    deltas = sorted(df['delta_deg'].unique())
    
    for delta in deltas:
        df_delta = df[df['delta_deg'] == delta]
        
        baseline = df_delta[df_delta['alss_flag'] == 'OFF'].groupby('SNR_dB').agg(
            rmse_mean=('rmse_deg', 'mean'),
            rmse_std=('rmse_deg', 'std'),
            resolve_rate=('resolved', 'mean')
        ).reset_index()
        
        alss = df_delta[df_delta['alss_flag'] == 'ON'].groupby('SNR_dB').agg(
            rmse_mean=('rmse_deg', 'mean'),
            rmse_std=('rmse_deg', 'std'),
            resolve_rate=('resolved', 'mean')
        ).reset_index()
        
        if baseline.empty or alss.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        
        ax.errorbar(baseline['SNR_dB'], baseline['rmse_mean'],
                   yerr=baseline['rmse_std'],
                   marker='s', markersize=9, color=color_baseline,
                   label='Baseline (ALSS OFF)',
                   capsize=4, capthick=2, linewidth=2.5,
                   markeredgewidth=1.5, markeredgecolor='white')
        
        ax.errorbar(alss['SNR_dB'], alss['rmse_mean'],
                   yerr=alss['rmse_std'],
                   marker='o', markersize=9, color=color_alss,
                   label='ALSS (mode=zero)',
                   capsize=4, capthick=2, linewidth=2.5,
                   markeredgewidth=1.5, markeredgecolor='white')
        
        ax.set_xlabel('SNR (dB)', fontweight='bold')
        ax.set_ylabel('RMSE (degrees)', fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.legend(loc='best', framealpha=0.95, edgecolor='#cccccc')
        ax.set_title(f'RMSE vs SNR (Δθ={int(delta)}°)', fontsize=12, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        output_file = output_dir / f'alss_rmse_delta{int(delta)}.pdf'
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        plt.savefig(output_file.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"✓ Saved: {output_file}")
        plt.close()

def plot_resolve_by_delta(df, output_dir):
    """Plot resolve rate vs SNR for each delta value."""
    deltas = sorted(df['delta_deg'].unique())
    
    for delta in deltas:
        df_delta = df[df['delta_deg'] == delta]
        
        baseline = df_delta[df_delta['alss_flag'] == 'OFF'].groupby('SNR_dB').agg(
            resolve_rate=('resolved', 'mean')
        ).reset_index()
        baseline['resolve_pct'] = baseline['resolve_rate'] * 100
        
        alss = df_delta[df_delta['alss_flag'] == 'ON'].groupby('SNR_dB').agg(
            resolve_rate=('resolved', 'mean')
        ).reset_index()
        alss['resolve_pct'] = alss['resolve_rate'] * 100
        
        if baseline.empty or alss.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        
        ax.plot(baseline['SNR_dB'], baseline['resolve_pct'],
               marker='s', markersize=9, color=color_baseline,
               label='Baseline (ALSS OFF)', linewidth=2.5,
               markeredgewidth=1.5, markeredgecolor='white')
        
        ax.plot(alss['SNR_dB'], alss['resolve_pct'],
               marker='o', markersize=9, color=color_alss,
               label='ALSS (mode=zero)', linewidth=2.5,
               linestyle='--', alpha=0.85,
               markeredgewidth=1.5, markeredgecolor='white')
        
        ax.set_xlabel('SNR (dB)', fontweight='bold')
        ax.set_ylabel('Resolve Rate (%)', fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.legend(loc='best', framealpha=0.95, edgecolor='#cccccc')
        ax.set_title(f'Resolve Rate vs SNR (Δθ={int(delta)}°)', fontsize=12, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        output_file = output_dir / f'alss_resolve_delta{int(delta)}.pdf'
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        plt.savefig(output_file.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"✓ Saved: {output_file}")
        plt.close()

def plot_combined_comparison(df, output_dir):
    """Create combined comparison plot for all deltas."""
    deltas = sorted(df['delta_deg'].unique())
    
    if len(deltas) < 2:
        print("⚠ Need at least 2 delta values for combined comparison")
        return
    
    fig, axes = plt.subplots(1, len(deltas), figsize=(4.5 * len(deltas), 3.2))
    if len(deltas) == 1:
        axes = [axes]
    
    for idx, delta in enumerate(deltas):
        ax = axes[idx]
        df_delta = df[df['delta_deg'] == delta]
        
        baseline = df_delta[df_delta['alss_flag'] == 'OFF'].groupby('SNR_dB').agg(
            rmse_mean=('rmse_deg', 'mean'),
            rmse_std=('rmse_deg', 'std')
        ).reset_index()
        
        alss = df_delta[df_delta['alss_flag'] == 'ON'].groupby('SNR_dB').agg(
            rmse_mean=('rmse_deg', 'mean'),
            rmse_std=('rmse_deg', 'std')
        ).reset_index()
        
        if baseline.empty or alss.empty:
            continue
        
        ax.errorbar(baseline['SNR_dB'], baseline['rmse_mean'],
                   yerr=baseline['rmse_std'],
                   marker='s', markersize=9, color=color_baseline,
                   label='Baseline',
                   capsize=4, capthick=2, linewidth=2.5,
                   markeredgewidth=1.5, markeredgecolor='white')
        
        ax.errorbar(alss['SNR_dB'], alss['rmse_mean'],
                   yerr=alss['rmse_std'],
                   marker='o', markersize=9, color=color_alss,
                   label='ALSS',
                   capsize=4, capthick=2, linewidth=2.5,
                   markeredgewidth=1.5, markeredgecolor='white')
        
        ax.set_xlabel('SNR (dB)', fontweight='bold')
        if idx == 0:
            ax.set_ylabel('RMSE (degrees)', fontweight='bold')
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc', fontsize=9)
        ax.set_title(f'({"abcdefgh"[idx]}) Δθ={int(delta)}°', fontsize=12, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=1.5)
    output_file = output_dir / 'alss_combined_all_deltas.pdf'
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_file.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: {output_file}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_alss_sweep.py <path_to_all_runs.csv>")
        print("Example: python plot_alss_sweep.py results/alss/all_runs.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    if not csv_path.exists():
        print(f"✗ Error: File not found: {csv_path}")
        sys.exit(1)
    
    output_dir = csv_path.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING ALSS SWEEP PLOTS")
    print("="*70 + "\n")
    print(f"Input:  {csv_path}")
    print(f"Output: {output_dir}\n")
    
    df = load_sweep_data(csv_path)
    print(f"Loaded {len(df)} trials")
    print(f"Delta values: {sorted(df['delta_deg'].unique())}")
    print(f"SNR values: {sorted(df['SNR_dB'].unique())}\n")
    
    print("Generating RMSE plots...")
    plot_rmse_by_delta(df, output_dir)
    
    print("\nGenerating resolve rate plots...")
    plot_resolve_by_delta(df, output_dir)
    
    print("\nGenerating combined comparison...")
    plot_combined_comparison(df, output_dir)
    
    print("\n" + "="*70)
    print("✓ All plots generated successfully!")
    print(f"  Location: {output_dir}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
