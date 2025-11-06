# tools/plot_paper_benchmarks.py
"""
Plot paper-ready benchmark results with confidence intervals.
Generates publication-quality figures with error bars and CRB overlays.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

def plot_rmse_with_ci(df, output_prefix, figsize=(12, 8), dpi=300):
    """
    Plot RMSE vs SNR for all delta values with bootstrap confidence intervals.
    """
    deltas = sorted(df['Delta_deg'].unique())
    snrs = sorted(df['SNR_dB'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, delta in enumerate(deltas):
        if idx >= len(axes):
            break
        ax = axes[idx]
        subset = df[df['Delta_deg'] == delta].sort_values('SNR_dB')
        
        # Plot RMSE with error bars
        ax.errorbar(
            subset['SNR_dB'], subset['RMSE_mean'],
            yerr=[
                subset['RMSE_mean'] - subset['RMSE_CI_low'],
                subset['RMSE_CI_high'] - subset['RMSE_mean']
            ],
            fmt='o-', linewidth=2, markersize=8,
            capsize=5, capthick=2,
            label=f'{df["Array"].iloc[0]} (N={df["N"].iloc[0]})',
            color='#2E86AB'
        )
        
        # Plot CRB if available
        if 'CRB_deg' in subset.columns and not subset['CRB_deg'].isna().all():
            ax.plot(subset['SNR_dB'], subset['CRB_deg'], 
                   'r--', linewidth=1.5, alpha=0.7, label='CRB')
        
        ax.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE (degrees)', fontsize=11, fontweight='bold')
        ax.set_title(f'Δ = {delta:.1f}°', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        
        # Add RMSE/CRB ratio annotations at high SNR
        if not subset.empty and 'RMSE_over_CRB' in subset.columns:
            high_snr = subset[subset['SNR_dB'] == subset['SNR_dB'].max()]
            if not high_snr.empty and not pd.isna(high_snr['RMSE_over_CRB'].iloc[0]):
                ratio = high_snr['RMSE_over_CRB'].iloc[0]
                ax.text(0.98, 0.98, f'{ratio:.1f}× CRB',
                       transform=ax.transAxes,
                       ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=9)
    
    # Hide unused subplots
    for idx in range(len(deltas), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save both formats
    for ext in ['png', 'pdf']:
        output_file = f"{output_prefix}_rmse_with_ci.{ext}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close()

def plot_resolve_with_ci(df, output_prefix, figsize=(12, 8), dpi=300):
    """
    Plot resolve percentage vs SNR with Wilson binomial confidence intervals.
    """
    deltas = sorted(df['Delta_deg'].unique())
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, delta in enumerate(deltas):
        if idx >= len(axes):
            break
        ax = axes[idx]
        subset = df[df['Delta_deg'] == delta].sort_values('SNR_dB')
        
        # Plot resolve percentage with error bars
        ax.errorbar(
            subset['SNR_dB'], subset['Resolve_pct'],
            yerr=[
                subset['Resolve_pct'] - subset['Resolve_CI_low'],
                subset['Resolve_CI_high'] - subset['Resolve_pct']
            ],
            fmt='s-', linewidth=2, markersize=8,
            capsize=5, capthick=2,
            label=f'{df["Array"].iloc[0]} (N={df["N"].iloc[0]})',
            color='#A23B72'
        )
        
        ax.set_xlabel('SNR (dB)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Resolve Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Δ = {delta:.1f}°', fontsize=12, fontweight='bold')
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9)
        
        # Add 50% threshold line
        ax.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    # Hide unused subplots
    for idx in range(len(deltas), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save both formats
    for ext in ['png', 'pdf']:
        output_file = f"{output_prefix}_resolve_with_ci.{ext}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close()

def plot_condition_numbers(df, output_prefix, figsize=(10, 6), dpi=300):
    """
    Plot average condition numbers vs SNR for all delta values.
    """
    deltas = sorted(df['Delta_deg'].unique())
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(deltas)))
    
    for delta, color in zip(deltas, colors):
        subset = df[df['Delta_deg'] == delta].sort_values('SNR_dB')
        if 'Avg_Condition' in subset.columns:
            ax.semilogy(subset['SNR_dB'], subset['Avg_Condition'],
                       'o-', linewidth=2, markersize=8,
                       label=f'Δ={delta:.0f}°', color=color)
    
    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Condition Number (κ)', fontsize=12, fontweight='bold')
    ax.set_title(f'Covariance Matrix Condition Numbers: {df["Array"].iloc[0]} (N={df["N"].iloc[0]})',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.legend(loc='best', fontsize=10, ncol=2)
    
    plt.tight_layout()
    
    # Save both formats
    for ext in ['png', 'pdf']:
        output_file = f"{output_prefix}_condition_numbers.{ext}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close()

def plot_combined_comparison(df, output_prefix, figsize=(14, 10), dpi=300):
    """
    Create combined 2x2 grid: RMSE and Resolve for hard cases (small deltas).
    """
    # Focus on challenging scenarios (Delta <= 20°)
    hard_cases = df[df['Delta_deg'] <= 20].copy()
    deltas = sorted(hard_cases['Delta_deg'].unique())
    
    if len(deltas) == 0:
        print("  Skipping combined plot: no hard cases (Delta <= 20°) found")
        return
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # Top row: RMSE
    ax_rmse = fig.add_subplot(gs[0, :])
    for delta, color in zip(deltas, colors[:len(deltas)]):
        subset = hard_cases[hard_cases['Delta_deg'] == delta].sort_values('SNR_dB')
        ax_rmse.errorbar(
            subset['SNR_dB'], subset['RMSE_mean'],
            yerr=[
                subset['RMSE_mean'] - subset['RMSE_CI_low'],
                subset['RMSE_CI_high'] - subset['RMSE_mean']
            ],
            fmt='o-', linewidth=2.5, markersize=9,
            capsize=5, capthick=2,
            label=f'Δ={delta:.0f}°', color=color
        )
    
    ax_rmse.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax_rmse.set_ylabel('RMSE (degrees)', fontsize=13, fontweight='bold')
    ax_rmse.set_title(f'Hard Cases RMSE: {df["Array"].iloc[0]} (N={df["N"].iloc[0]}, {df["Trials"].iloc[0]} trials/point)',
                     fontsize=14, fontweight='bold')
    ax_rmse.grid(True, alpha=0.3, linestyle='--')
    ax_rmse.legend(loc='best', fontsize=11, ncol=len(deltas))
    
    # Bottom left: Resolve Rate
    ax_resolve = fig.add_subplot(gs[1, 0])
    for delta, color in zip(deltas, colors[:len(deltas)]):
        subset = hard_cases[hard_cases['Delta_deg'] == delta].sort_values('SNR_dB')
        ax_resolve.errorbar(
            subset['SNR_dB'], subset['Resolve_pct'],
            yerr=[
                subset['Resolve_pct'] - subset['Resolve_CI_low'],
                subset['Resolve_CI_high'] - subset['Resolve_pct']
            ],
            fmt='s-', linewidth=2.5, markersize=9,
            capsize=5, capthick=2,
            label=f'Δ={delta:.0f}°', color=color
        )
    
    ax_resolve.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax_resolve.set_ylabel('Resolve Rate (%)', fontsize=13, fontweight='bold')
    ax_resolve.set_title('Resolution Performance', fontsize=14, fontweight='bold')
    ax_resolve.set_ylim(-5, 105)
    ax_resolve.axhline(y=50, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    ax_resolve.grid(True, alpha=0.3, linestyle='--')
    ax_resolve.legend(loc='best', fontsize=10)
    
    # Bottom right: RMSE/CRB Ratio
    ax_ratio = fig.add_subplot(gs[1, 1])
    for delta, color in zip(deltas, colors[:len(deltas)]):
        subset = hard_cases[hard_cases['Delta_deg'] == delta].sort_values('SNR_dB')
        if 'RMSE_over_CRB' in subset.columns:
            ax_ratio.plot(subset['SNR_dB'], subset['RMSE_over_CRB'],
                         'o-', linewidth=2.5, markersize=9,
                         label=f'Δ={delta:.0f}°', color=color)
    
    ax_ratio.set_xlabel('SNR (dB)', fontsize=13, fontweight='bold')
    ax_ratio.set_ylabel('RMSE / CRB', fontsize=13, fontweight='bold')
    ax_ratio.set_title('Efficiency Relative to CRB', fontsize=14, fontweight='bold')
    ax_ratio.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='CRB')
    ax_ratio.grid(True, alpha=0.3, linestyle='--')
    ax_ratio.legend(loc='best', fontsize=10)
    ax_ratio.set_yscale('log')
    
    # Save both formats
    for ext in ['png', 'pdf']:
        output_file = f"{output_prefix}_combined_hard_cases.{ext}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {output_file}")
    
    plt.close()

def generate_summary_table(df, output_file):
    """
    Generate LaTeX-ready summary table.
    """
    summary = df.pivot_table(
        values=['RMSE_mean', 'Resolve_pct', 'RMSE_over_CRB'],
        index='Delta_deg',
        columns='SNR_dB',
        aggfunc='first'
    )
    
    # Save as CSV
    summary.to_csv(output_file)
    print(f"  Saved: {output_file}")
    
    # Print markdown table (if tabulate available)
    try:
        print("\n" + "="*80)
        print("SUMMARY TABLE (Markdown):")
        print("="*80)
        print(summary.to_markdown(floatfmt='.3f'))
        print("="*80 + "\n")
    except ImportError:
        print("  (Markdown table skipped - install 'tabulate' package for formatted output)")
        print(f"  View summary in: {output_file}\n")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot paper benchmark results with confidence intervals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot Z5 results (50 trials)
  python tools/plot_paper_benchmarks.py results/bench/z5_paper_N7_T50_alss_zero.csv
  
  # Plot Z5 results (400 trials) with custom output
  python tools/plot_paper_benchmarks.py results/bench/z5_paper_N7_T400_alss_zero.csv \\
      --output results/bench/figures/z5_paper_400trials
  
  # Generate all plots including combined view
  python tools/plot_paper_benchmarks.py results/bench/z5_paper_N7_T50_alss_zero.csv --all
        """)
    
    parser.add_argument('input_csv', type=str,
                       help='Input CSV file from run_paper_benchmarks.py')
    parser.add_argument('--output', type=str, default=None,
                       help='Output prefix (default: same as input basename)')
    parser.add_argument('--all', action='store_true',
                       help='Generate all plot types (RMSE, resolve, condition, combined)')
    parser.add_argument('--rmse-only', action='store_true',
                       help='Generate RMSE plot only')
    parser.add_argument('--resolve-only', action='store_true',
                       help='Generate resolve plot only')
    parser.add_argument('--combined-only', action='store_true',
                       help='Generate combined hard cases plot only')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Figure resolution (default: 300)')
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input_csv)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1
    
    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Array: {df['Array'].iloc[0]}, N={df['N'].iloc[0]}")
    print(f"  SNR range: {df['SNR_dB'].min()} to {df['SNR_dB'].max()} dB")
    print(f"  Delta range: {df['Delta_deg'].min()}° to {df['Delta_deg'].max()}°")
    print(f"  Trials/point: {df['Trials'].iloc[0]}")
    
    # Determine output prefix
    if args.output:
        output_prefix = args.output
    else:
        output_prefix = str(input_path.parent / 'figures' / input_path.stem)
    
    # Create output directory
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput prefix: {output_prefix}")
    print("="*80)
    
    # Generate plots based on flags
    if args.all or (not args.rmse_only and not args.resolve_only and not args.combined_only):
        # Default: generate all plots
        print("Generating RMSE plot with confidence intervals...")
        plot_rmse_with_ci(df, output_prefix, dpi=args.dpi)
        
        print("Generating resolve rate plot with confidence intervals...")
        plot_resolve_with_ci(df, output_prefix, dpi=args.dpi)
        
        print("Generating condition number plot...")
        plot_condition_numbers(df, output_prefix, dpi=args.dpi)
        
        print("Generating combined hard cases plot...")
        plot_combined_comparison(df, output_prefix, dpi=args.dpi)
        
        print("Generating summary table...")
        summary_file = f"{output_prefix}_summary.csv"
        generate_summary_table(df, summary_file)
    else:
        # Generate specific plots
        if args.rmse_only:
            print("Generating RMSE plot with confidence intervals...")
            plot_rmse_with_ci(df, output_prefix, dpi=args.dpi)
        
        if args.resolve_only:
            print("Generating resolve rate plot with confidence intervals...")
            plot_resolve_with_ci(df, output_prefix, dpi=args.dpi)
        
        if args.combined_only:
            print("Generating combined hard cases plot...")
            plot_combined_comparison(df, output_prefix, dpi=args.dpi)
    
    print("\n" + "="*80)
    print("✓ Plotting complete!")
    print("="*80 + "\n")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
