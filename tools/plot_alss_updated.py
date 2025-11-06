"""
Plot ALSS sweep results for updated configuration.
SNR: 0, 5, 10, 15 dB
Delta: 10, 20, 30, 45 degrees
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Publication-quality styling
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 13,
    'lines.linewidth': 2.5,
    'lines.markersize': 9,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def load_csv_safe(path):
    """Load CSV if it exists, return None otherwise."""
    if Path(path).exists():
        return pd.read_csv(path)
    return None

def aggregate_results(df):
    """Compute mean RMSE and resolve rate per SNR."""
    return df.groupby('SNR_dB').agg({
        'rmse_deg': 'mean',
        'resolved': 'mean'
    }).reset_index().rename(columns={'SNR_dB': 'snr_db'})

def plot_rmse_single_delta(baseline_stats, alss_stats, delta, output_dir):
    """Plot RMSE vs SNR comparison for a single delta."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=300)
    
    snr = baseline_stats['snr_db']
    rmse_baseline = baseline_stats['rmse_deg']
    rmse_alss = alss_stats['rmse_deg']
    
    # Plot lines
    ax.plot(snr, rmse_baseline, 'o-', color='#E74C3C', 
            label='Baseline', linewidth=2.5, markersize=9,
            markeredgewidth=1.5, markeredgecolor='white')
    ax.plot(snr, rmse_alss, 's-', color='#3498DB',
            label='ALSS', linewidth=2.5, markersize=9,
            markeredgewidth=1.5, markeredgecolor='white')
    
    # Calculate average improvement
    improvement = ((rmse_alss - rmse_baseline) / rmse_baseline * 100).mean()
    
    # Add improvement annotation
    ax.text(0.97, 0.97, f'{improvement:+.1f}%',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=11, fontweight='bold', 
            color='#27AE60' if improvement < 0 else '#E74C3C',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor='#27AE60' if improvement < 0 else '#E74C3C', 
                     linewidth=1.5, alpha=0.9))
    
    ax.set_xlabel('SNR (dB)', fontweight='bold')
    ax.set_ylabel('RMSE (degrees)', fontweight='bold')
    ax.set_title(f'RMSE vs SNR (Δθ = {delta}°)', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / f'alss_rmse_delta{delta}.pdf'
    png_path = output_dir / f'alss_rmse_delta{delta}.png'
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

def plot_resolve_single_delta(baseline_stats, alss_stats, delta, output_dir):
    """Plot resolve rate vs SNR comparison for a single delta."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=300)
    
    snr = baseline_stats['snr_db']
    resolve_baseline = baseline_stats['resolved'] * 100
    resolve_alss = alss_stats['resolved'] * 100
    
    # Plot lines
    ax.plot(snr, resolve_baseline, 'o-', color='#E74C3C',
            label='Baseline', linewidth=2.5, markersize=9,
            markeredgewidth=1.5, markeredgecolor='white')
    ax.plot(snr, resolve_alss, 's-', color='#3498DB',
            label='ALSS', linewidth=2.5, markersize=9,
            markeredgewidth=1.5, markeredgecolor='white')
    
    # Check if curves are close (ALSS harmless case)
    max_diff = np.abs(resolve_alss - resolve_baseline).max()
    if max_diff < 5:  # Less than 5% difference
        ax.text(0.5, 0.15, 'Curves overlap\n(ALSS harmless)',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=10, style='italic', color='#7F8C8D',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='#BDC3C7', linewidth=1.0, alpha=0.8))
    
    ax.set_xlabel('SNR (dB)', fontweight='bold')
    ax.set_ylabel('Resolve Rate (%)', fontweight='bold')
    ax.set_title(f'Resolve Rate vs SNR (Δθ = {delta}°)', fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend(loc='lower right', frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / f'alss_resolve_delta{delta}.pdf'
    png_path = output_dir / f'alss_resolve_delta{delta}.png'
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

def plot_combined_all_deltas(all_data, output_dir):
    """Plot combined comparison across all delta values (2×2 grid)."""
    deltas = sorted(all_data.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=300)
    axes = axes.flatten()
    
    colors_baseline = ['#E74C3C', '#C0392B', '#A93226', '#922B21']
    colors_alss = ['#3498DB', '#2E86C1', '#2874A6', '#21618C']
    
    for idx, delta in enumerate(deltas):
        ax = axes[idx]
        baseline_stats, alss_stats = all_data[delta]
        
        snr = baseline_stats['snr_db']
        rmse_baseline = baseline_stats['rmse_deg']
        rmse_alss = alss_stats['rmse_deg']
        
        # Plot
        ax.plot(snr, rmse_baseline, 'o-', color=colors_baseline[idx],
                label='Baseline', linewidth=2.0, markersize=7,
                markeredgewidth=1.2, markeredgecolor='white')
        ax.plot(snr, rmse_alss, 's-', color=colors_alss[idx],
                label='ALSS', linewidth=2.0, markersize=7,
                markeredgewidth=1.2, markeredgecolor='white')
        
        # Calculate improvement
        improvement = ((rmse_alss - rmse_baseline) / rmse_baseline * 100).mean()
        
        # Add annotation
        ax.text(0.95, 0.95, f'{improvement:+.1f}%',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, fontweight='bold',
                color='#27AE60' if improvement < 0 else '#E74C3C',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#27AE60' if improvement < 0 else '#E74C3C',
                         linewidth=1.2, alpha=0.9))
        
        ax.set_title(f'Δθ = {delta}°', fontweight='bold', fontsize=11)
        ax.set_xlabel('SNR (dB)', fontweight='bold', fontsize=10)
        ax.set_ylabel('RMSE (degrees)', fontweight='bold', fontsize=10)
        ax.legend(loc='upper right', fontsize=8, frameon=True)
        ax.grid(True, alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = output_dir / 'alss_combined_all_deltas.pdf'
    png_path = output_dir / 'alss_combined_all_deltas.png'
    
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

def main():
    results_dir = Path('results/alss')
    output_dir = Path('papers/radarcon2025_alss/figures')
    
    print("Loading CSV files...")
    
    # Configuration
    SNRs = [0, 5, 10, 15]
    Deltas = [10, 20, 30, 45]
    
    # Try to load all available files
    data = {}
    for delta in Deltas:
        for snr in SNRs:
            for mode in ['baseline', 'alss']:
                key = f"{mode}_d{delta}_snr{snr}"
                path = results_dir / f"{mode}_Z5_M64_d{delta}_snr{snr}.csv"
                df = load_csv_safe(path)
                if df is not None:
                    data[key] = df
                    print(f"  ✓ Loaded: {path.name}")
    
    if not data:
        print("❌ No CSV files found!")
        return
    
    print(f"\nFound {len(data)} CSV files")
    
    # Process each delta that has complete data
    all_delta_data = {}
    
    for delta in Deltas:
        # Check if we have complete data for this delta
        baseline_dfs = []
        alss_dfs = []
        
        for snr in SNRs:
            baseline_key = f"baseline_d{delta}_snr{snr}"
            alss_key = f"alss_d{delta}_snr{snr}"
            
            if baseline_key in data and alss_key in data:
                baseline_dfs.append(data[baseline_key])
                alss_dfs.append(data[alss_key])
        
        if len(baseline_dfs) != len(SNRs) or len(alss_dfs) != len(SNRs):
            print(f"\n⚠ Skipping delta={delta}° (incomplete: {len(baseline_dfs)}/{len(SNRs)} SNR points)")
            continue
        
        print(f"\n📊 Generating plots for Δθ = {delta}° ({len(baseline_dfs)} SNR points)...")
        
        # Concatenate all SNR data
        baseline_all = pd.concat(baseline_dfs, ignore_index=True)
        alss_all = pd.concat(alss_dfs, ignore_index=True)
        
        # Aggregate statistics
        baseline_stats = aggregate_results(baseline_all)
        alss_stats = aggregate_results(alss_all)
        
        # Store for combined plot
        all_delta_data[delta] = (baseline_stats, alss_stats)
        
        # Generate individual plots
        plot_rmse_single_delta(baseline_stats, alss_stats, delta, output_dir)
        plot_resolve_single_delta(baseline_stats, alss_stats, delta, output_dir)
    
    # Generate combined plot if we have data for all deltas
    if len(all_delta_data) == len(Deltas):
        print(f"\n📊 Generating combined comparison plot...")
        plot_combined_all_deltas(all_delta_data, output_dir)
    else:
        print(f"\n⚠ Skipping combined plot (need all {len(Deltas)} deltas, have {len(all_delta_data)})")
    
    print(f"\n✅ Done! Check: {output_dir}")

if __name__ == '__main__':
    main()
