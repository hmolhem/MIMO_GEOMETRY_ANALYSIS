#!/usr/bin/env python3
"""
Generate publication-quality plots for ALSS ablation study.

Usage:
    python tools/plot_alss_figures.py

Outputs:
    - papers/radarcon2025_alss/figures/alss_rmse_vs_snr.pdf
    - papers/radarcon2025_alss/figures/alss_resolve_vs_snr.pdf
    - papers/radarcon2025_alss/figures/alss_combined.pdf
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set publication-quality defaults (IEEE-style)
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

def load_and_analyze(csv_path):
    """Load CSV and compute mean/std by SNR."""
    df = pd.read_csv(csv_path)
    results = []
    for snr in sorted(df['SNR_dB'].unique()):
        subset = df[df['SNR_dB'] == snr]
        results.append({
            'SNR_dB': snr,
            'rmse_mean': subset['rmse_deg'].mean(),
            'rmse_std': subset['rmse_deg'].std(),
            'resolve_rate': (subset['resolved'].sum() / len(subset)) * 100,
        })
    return pd.DataFrame(results)

def plot_rmse_vs_snr():
    """Figure 1: RMSE vs SNR (delta=13°, moderate difficulty)."""
    baseline = load_and_analyze('results/bench/alss_off_hard.csv')
    alss = load_and_analyze('results/bench/alss_on_hard.csv')
    
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    
    # Clean color scheme
    color_baseline = '#E74C3C'  # Red
    color_alss = '#3498DB'      # Blue
    
    # Plot lines with cleaner error bars
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
    
    # Annotate improvement at SNR=5dB
    snr5_baseline = baseline[baseline['SNR_dB'] == 5]['rmse_mean'].values[0]
    snr5_alss = alss[alss['SNR_dB'] == 5]['rmse_mean'].values[0]
    improvement = (snr5_baseline - snr5_alss) / snr5_baseline * 100
    
    # Add value labels
    ax.text(5.4, snr5_baseline, f'{snr5_baseline:.2f}°', 
            fontsize=10, va='center', fontweight='bold', color=color_baseline)
    ax.text(5.4, snr5_alss, f'{snr5_alss:.2f}°', 
            fontsize=10, va='center', fontweight='bold', color=color_alss)
    
    # Arrow showing improvement
    arrow_x = 5.2
    ax.annotate('', xy=(arrow_x, snr5_alss), xytext=(arrow_x, snr5_baseline),
                arrowprops=dict(arrowstyle='<->', color='#27AE60', lw=2.5))
    ax.text(arrow_x + 0.15, (snr5_baseline + snr5_alss) / 2, 
            f'−{improvement:.1f}%', 
            fontsize=10, color='#27AE60', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='#27AE60', alpha=0.9))
    
    ax.set_xlabel('SNR (dB)', fontweight='bold')
    ax.set_ylabel('RMSE (degrees)', fontweight='bold')
    ax.set_xticks([0, 5, 10])
    ax.set_xlim(-0.8, 10.8)
    ax.set_ylim(11, 27)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc')
    ax.set_title('Z5 (N=7), M=64 snapshots, Δθ=13°', fontsize=12, pad=10)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save
    output_dir = Path('papers/radarcon2025_alss/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'alss_rmse_vs_snr.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_dir / 'alss_rmse_vs_snr.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: {output_dir / 'alss_rmse_vs_snr.pdf'}")
    plt.close()

def plot_resolve_vs_snr():
    """Figure 2: Resolve rate vs SNR (delta=2°, easy case)."""
    baseline = load_and_analyze('results/bench/alss_off_sweep.csv')
    alss = load_and_analyze('results/bench/alss_on_sweep.csv')
    
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    
    # Clean color scheme
    color_baseline = '#E74C3C'  # Red
    color_alss = '#3498DB'      # Blue
    
    ax.plot(baseline['SNR_dB'], baseline['resolve_rate'], 
            marker='s', markersize=9, color=color_baseline, 
            label='Baseline (ALSS OFF)', linewidth=2.5,
            markeredgewidth=1.5, markeredgecolor='white')
    
    ax.plot(alss['SNR_dB'], alss['resolve_rate'], 
            marker='o', markersize=9, color=color_alss, 
            label='ALSS (mode=zero)', 
            linewidth=2.5, linestyle='--', alpha=0.85,
            markeredgewidth=1.5, markeredgecolor='white')
    
    ax.set_xlabel('SNR (dB)', fontweight='bold')
    ax.set_ylabel('Resolve Rate (%)', fontweight='bold')
    ax.set_xticks([0, 5, 10])
    ax.set_xlim(-0.8, 10.8)
    ax.set_ylim(32, 68)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax.legend(loc='lower right', framealpha=0.95, edgecolor='#cccccc')
    ax.set_title('Z5 (N=7), M=64 snapshots, Δθ=2° (easy case)', fontsize=12, pad=10)
    
    # Add annotation showing "nearly identical"
    ax.text(5, 47, 'Curves overlap\n(ALSS harmless)', 
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', 
                     edgecolor='#95A5A6', alpha=0.9))
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save
    output_dir = Path('papers/radarcon2025_alss/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'alss_resolve_vs_snr.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_dir / 'alss_resolve_vs_snr.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: {output_dir / 'alss_resolve_vs_snr.pdf'}")
    plt.close()

def plot_combined():
    """Combined figure: side-by-side RMSE and Resolve plots."""
    baseline_hard = load_and_analyze('results/bench/alss_off_hard.csv')
    alss_hard = load_and_analyze('results/bench/alss_on_hard.csv')
    baseline_easy = load_and_analyze('results/bench/alss_off_sweep.csv')
    alss_easy = load_and_analyze('results/bench/alss_on_sweep.csv')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
    
    # Clean color scheme
    color_baseline = '#E74C3C'  # Red
    color_alss = '#3498DB'      # Blue
    
    # Left: RMSE vs SNR (hard)
    ax1.errorbar(baseline_hard['SNR_dB'], baseline_hard['rmse_mean'], 
                 yerr=baseline_hard['rmse_std'],
                 marker='s', markersize=9, color=color_baseline, 
                 label='Baseline',
                 capsize=4, capthick=2, linewidth=2.5,
                 markeredgewidth=1.5, markeredgecolor='white')
    ax1.errorbar(alss_hard['SNR_dB'], alss_hard['rmse_mean'],
                 yerr=alss_hard['rmse_std'],
                 marker='o', markersize=9, color=color_alss, 
                 label='ALSS',
                 capsize=4, capthick=2, linewidth=2.5,
                 markeredgewidth=1.5, markeredgecolor='white')
    
    snr5_baseline = baseline_hard[baseline_hard['SNR_dB'] == 5]['rmse_mean'].values[0]
    snr5_alss = alss_hard[alss_hard['SNR_dB'] == 5]['rmse_mean'].values[0]
    improvement = (snr5_baseline - snr5_alss) / snr5_baseline * 100
    
    arrow_x = 5.2
    ax1.annotate('', xy=(arrow_x, snr5_alss), xytext=(arrow_x, snr5_baseline),
                 arrowprops=dict(arrowstyle='<->', color='#27AE60', lw=2.5))
    ax1.text(arrow_x + 0.15, (snr5_baseline + snr5_alss) / 2, 
             f'−{improvement:.1f}%', 
             fontsize=9, color='#27AE60', va='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                      edgecolor='#27AE60', alpha=0.9))
    
    ax1.set_xlabel('SNR (dB)', fontweight='bold')
    ax1.set_ylabel('RMSE (degrees)', fontweight='bold')
    ax1.set_xticks([0, 5, 10])
    ax1.set_xlim(-0.8, 10.8)
    ax1.set_ylim(11, 27)
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax1.legend(loc='upper right', framealpha=0.95, edgecolor='#cccccc', fontsize=10)
    ax1.set_title('(a) Moderate difficulty (Δθ=13°)', fontsize=12, fontweight='bold', pad=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Right: Resolve rate vs SNR (easy)
    ax2.plot(baseline_easy['SNR_dB'], baseline_easy['resolve_rate'],
             marker='s', markersize=9, color=color_baseline, 
             label='Baseline', linewidth=2.5,
             markeredgewidth=1.5, markeredgecolor='white')
    ax2.plot(alss_easy['SNR_dB'], alss_easy['resolve_rate'],
             marker='o', markersize=9, color=color_alss, 
             label='ALSS',
             linewidth=2.5, linestyle='--', alpha=0.85,
             markeredgewidth=1.5, markeredgecolor='white')
    
    ax2.set_xlabel('SNR (dB)', fontweight='bold')
    ax2.set_ylabel('Resolve Rate (%)', fontweight='bold')
    ax2.set_xticks([0, 5, 10])
    ax2.set_xlim(-0.8, 10.8)
    ax2.set_ylim(32, 68)
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax2.legend(loc='lower right', framealpha=0.95, edgecolor='#cccccc', fontsize=10)
    ax2.set_title('(b) Easy case (Δθ=2°)', fontsize=12, fontweight='bold', pad=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=1.5)
    
    # Save
    output_dir = Path('papers/radarcon2025_alss/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'alss_combined.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(output_dir / 'alss_combined.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Saved: {output_dir / 'alss_combined.pdf'}")
    plt.close()

def main():
    print("\n" + "="*70)
    print("GENERATING ALSS PUBLICATION FIGURES")
    print("="*70 + "\n")
    
    print("Generating figures...")
    plot_rmse_vs_snr()
    plot_resolve_vs_snr()
    plot_combined()
    
    print("\n" + "="*70)
    print("✓ All figures generated!")
    print("  Location: papers/radarcon2025_alss/figures/")
    print("  Files: alss_rmse_vs_snr.pdf, alss_resolve_vs_snr.pdf, alss_combined.pdf")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
