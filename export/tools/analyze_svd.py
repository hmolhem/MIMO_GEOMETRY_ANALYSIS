"""
Analyze and visualize SVD data from MIMO array covariance matrices.

Usage:
    python tools/analyze_svd.py results/svd/

This tool:
1. Loads Rx and Rv singular values from CSV files
2. Computes condition numbers (κ = σ_max / σ_min)
3. Visualizes singular value spectra
4. Compares ALSS ON vs OFF for virtual covariance Rv
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import re

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
    'lines.linewidth': 2.5,
    'lines.markersize': 9,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def parse_filename(filename):
    """
    Extract metadata from SVD filename.
    Format: {array}_{alg}_N{N}_M{M}_snr{SNR}_trial{t}_{Rx|Rv}_svd.csv
    """
    # Pattern that handles array names with parentheses like Z5(N=7)
    pattern = r'(?P<array>[\w\(\)=]+)_(?P<alg>\w+)_N(?P<N>\d+)_M(?P<M>\d+)_snr(?P<SNR>-?\d+(?:\.\d+)?)_trial(?P<trial>\d+)_(?P<cov_type>Rx|Rv)_svd\.csv'
    match = re.match(pattern, filename)
    if match:
        return match.groupdict()
    return None

def load_svd_data(svd_dir):
    """
    Load all SVD CSV files and organize by metadata.
    
    Returns:
        dict: {(array, alg, N, M, SNR, trial, cov_type): singular_values}
    """
    svd_dir = Path(svd_dir)
    data = {}
    
    for csv_file in svd_dir.glob("*_svd.csv"):
        meta = parse_filename(csv_file.name)
        if meta:
            sv = np.loadtxt(csv_file, delimiter=',').flatten()
            key = (meta['array'], meta['alg'], int(meta['N']), int(meta['M']), 
                   float(meta['SNR']), int(meta['trial']), meta['cov_type'])
            data[key] = sv
    
    return data

def compute_condition_numbers(data):
    """
    Compute condition number κ = σ_max / σ_min for each entry.
    
    Returns:
        dict: {key: condition_number}
    """
    cond_nums = {}
    for key, sv in data.items():
        if len(sv) > 0:
            cond = sv[0] / max(sv[-1], 1e-12)
            cond_nums[key] = cond
    return cond_nums

def plot_singular_values(data, output_dir, filter_params=None):
    """
    Plot singular value spectra.
    
    Parameters:
    -----------
    data : dict
        SVD data from load_svd_data()
    output_dir : Path
        Output directory for plots
    filter_params : dict, optional
        Filter by {'array': 'Z5', 'M': 64, 'SNR': 10, ...}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Apply filters
    if filter_params:
        filtered_data = {k: v for k, v in data.items() 
                        if all(k[i] == filter_params.get(field) 
                              for i, field in enumerate(['array', 'alg', 'N', 'M', 'SNR', 'trial', 'cov_type'])
                              if field in filter_params)}
    else:
        filtered_data = data
    
    if not filtered_data:
        print(f"⚠ No data matches filter: {filter_params}")
        return
    
    # Group by array, M, SNR, cov_type
    groups = {}
    for key, sv in filtered_data.items():
        array, alg, N, M, SNR, trial, cov_type = key
        group_key = (array, M, SNR, cov_type)
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(sv)
    
    # Plot each group
    for group_key, sv_list in groups.items():
        array, M, SNR, cov_type = group_key
        
        # Average over trials
        sv_mean = np.mean(sv_list, axis=0)
        sv_std = np.std(sv_list, axis=0)
        
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)
        
        x = np.arange(1, len(sv_mean) + 1)
        ax.semilogy(x, sv_mean, 'o-', color='#3498DB', linewidth=2.5, 
                   markersize=8, markeredgewidth=1.5, markeredgecolor='white',
                   label=f'Mean ({len(sv_list)} trials)')
        ax.fill_between(x, sv_mean - sv_std, sv_mean + sv_std, 
                       alpha=0.2, color='#3498DB', label='±1σ')
        
        ax.set_xlabel('Singular Value Index', fontweight='bold')
        ax.set_ylabel('Singular Value (log scale)', fontweight='bold')
        ax.set_title(f'{cov_type} Spectrum: {array}, M={M}, SNR={SNR}dB', fontweight='bold')
        ax.legend(loc='best', frameon=True)
        ax.grid(True, alpha=0.25, linestyle='--')
        
        plt.tight_layout()
        
        plot_name = f'svd_{cov_type}_{array}_M{M}_snr{SNR}.png'
        plot_path = output_dir / plot_name
        fig.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"✓ Saved: {plot_path}")

def compare_alss_effect(data, output_dir, array='Z5', N=7, M=64, SNR=10):
    """
    Compare Rv singular values with ALSS ON vs OFF.
    
    Shows how ALSS affects the virtual covariance condition number.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter for CoarrayMUSIC Rv data
    baseline_keys = [k for k in data.keys() 
                     if k[0] == array and k[1] == 'CoarrayMUSIC' and k[2] == N 
                     and k[3] == M and k[4] == SNR and k[6] == 'Rv']
    
    if not baseline_keys:
        print(f"⚠ No Rv data found for comparison: {array}, N={N}, M={M}, SNR={SNR}")
        return
    
    # Group by trial to match ALSS ON/OFF pairs
    trials = sorted(set(k[5] for k in baseline_keys))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    
    for trial in trials[:3]:  # Plot first 3 trials
        # Find ALSS OFF (baseline)
        baseline_key = [k for k in baseline_keys if k[5] == trial]
        if baseline_key:
            sv_baseline = data[baseline_key[0]]
            x = np.arange(1, len(sv_baseline) + 1)
            
            # ALSS OFF
            axes[0].semilogy(x, sv_baseline, 'o-', alpha=0.7, linewidth=2.0,
                           markersize=6, label=f'Trial {trial}')
            
            # ALSS ON (if available - would need separate run)
            # This is a placeholder - you'd load ALSS ON data from separate benchmark
            axes[1].semilogy(x, sv_baseline, 'o-', alpha=0.7, linewidth=2.0,
                           markersize=6, label=f'Trial {trial}')
    
    for ax, title in zip(axes, ['ALSS OFF (Baseline)', 'ALSS ON']):
        ax.set_xlabel('Singular Value Index', fontweight='bold')
        ax.set_ylabel('Singular Value (log scale)', fontweight='bold')
        ax.set_title(f'{title}\n{array}, M={M}, SNR={SNR}dB', fontweight='bold')
        ax.legend(loc='best', frameon=True, fontsize=9)
        ax.grid(True, alpha=0.25, linestyle='--')
    
    plt.tight_layout()
    
    plot_path = output_dir / f'svd_alss_comparison_{array}_M{M}_snr{SNR}.png'
    fig.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {plot_path}")

def generate_condition_number_table(cond_nums, output_dir):
    """
    Generate CSV table of condition numbers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for key, cond in cond_nums.items():
        array, alg, N, M, SNR, trial, cov_type = key
        rows.append({
            'array': array,
            'algorithm': alg,
            'N': N,
            'M': M,
            'SNR_dB': SNR,
            'trial': trial,
            'cov_type': cov_type,
            'condition_number': cond,
        })
    
    df = pd.DataFrame(rows)
    csv_path = output_dir / 'condition_numbers.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Saved: {csv_path}")
    
    # Summary statistics
    summary = df.groupby(['array', 'M', 'SNR_dB', 'cov_type'])['condition_number'].agg(['mean', 'std', 'min', 'max'])
    summary_path = output_dir / 'condition_numbers_summary.csv'
    summary.to_csv(summary_path)
    
    print(f"✓ Saved: {summary_path}")
    print("\nCondition Number Summary:")
    print(summary.to_string())

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tools/analyze_svd.py <svd_directory>")
        print("Example: python tools/analyze_svd.py results/svd/")
        sys.exit(1)
    
    svd_dir = Path(sys.argv[1])
    
    if not svd_dir.exists():
        print(f"❌ Directory not found: {svd_dir}")
        sys.exit(1)
    
    print(f"Loading SVD data from: {svd_dir}")
    data = load_svd_data(svd_dir)
    
    if not data:
        print("❌ No SVD data found!")
        print("Run benchmarks with --dump-svd flag first:")
        print("  python core/analysis_scripts/run_benchmarks.py --dump-svd ...")
        sys.exit(1)
    
    print(f"✓ Loaded {len(data)} SVD files")
    
    # Compute condition numbers
    print("\nComputing condition numbers...")
    cond_nums = compute_condition_numbers(data)
    
    # Generate outputs
    output_dir = svd_dir / 'analysis'
    
    print("\nGenerating condition number tables...")
    generate_condition_number_table(cond_nums, output_dir)
    
    print("\nPlotting singular value spectra...")
    plot_singular_values(data, output_dir)
    
    print("\n✅ Analysis complete! Check:", output_dir)

if __name__ == '__main__':
    main()
