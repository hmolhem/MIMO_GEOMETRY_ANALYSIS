"""
Create placeholder figures for ALSS-II paper compilation
This script generates simple placeholder PDFs so the LaTeX compiles without errors
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

def create_snr_sweep_placeholder():
    """Create placeholder for SNR sweep figure"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    snr = np.array([0, 5, 10, 15, 20])
    
    # Placeholder data showing expected trends
    rmse_baseline = np.array([15, 10, 7, 5, 4])
    rmse_mcm = np.array([18, 12, 8.5, 6.2, 5])
    rmse_alss_mcm = np.array([12, 7.8, 5.5, 4.3, 3.8])
    rmse_alss_ii_mcm = np.array([11, 7.2, 5.1, 4.1, 3.7])
    
    ax.plot(snr, rmse_baseline, 'k--', marker='s', label='Baseline (No MCM)', linewidth=2)
    ax.plot(snr, rmse_mcm, 'r-', marker='o', label='With MCM', linewidth=2)
    ax.plot(snr, rmse_alss_mcm, 'b-', marker='^', label='ALSS + MCM', linewidth=2)
    ax.plot(snr, rmse_alss_ii_mcm, 'g-', marker='d', label='ALSS-II + MCM', linewidth=2)
    
    ax.set_xlabel('SNR (dB)', fontsize=11)
    ax.set_ylabel('RMSE (degrees)', fontsize=11)
    ax.set_title('DOA Estimation Performance vs SNR (Z5, M=200)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 20])
    
    plt.tight_layout()
    plt.savefig('alss_ii_snr_sweep.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('alss_ii_snr_sweep.png', dpi=300, bbox_inches='tight')
    print("Created: alss_ii_snr_sweep.pdf/png")
    plt.close()

def create_array_comparison_placeholder():
    """Create placeholder for array comparison figure"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    arrays = ['Z1', 'Z3.2', 'Z5', 'Nested', 'ULA']
    x = np.arange(len(arrays))
    width = 0.25
    
    # Placeholder data showing ALSS-II improvements
    rmse_baseline = np.array([8.5, 7.8, 7.2, 9.1, 12.5])
    rmse_alss = np.array([6.2, 5.9, 5.5, 7.3, 10.8])
    rmse_alss_ii = np.array([5.8, 5.5, 5.1, 7.0, 10.5])
    
    ax.bar(x - width, rmse_baseline, width, label='Baseline + MCM', color='lightcoral')
    ax.bar(x, rmse_alss, width, label='ALSS + MCM', color='skyblue')
    ax.bar(x + width, rmse_alss_ii, width, label='ALSS-II + MCM', color='lightgreen')
    
    ax.set_xlabel('Array Type', fontsize=11)
    ax.set_ylabel('RMSE (degrees)', fontsize=11)
    ax.set_title('Performance Across Array Geometries (SNR=10dB, M=200)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(arrays)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 14])
    
    plt.tight_layout()
    plt.savefig('alss_ii_array_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('alss_ii_array_comparison.png', dpi=300, bbox_inches='tight')
    print("Created: alss_ii_array_comparison.pdf/png")
    plt.close()

if __name__ == "__main__":
    # Change to figures directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Generating placeholder figures for ALSS-II paper...")
    create_snr_sweep_placeholder()
    create_array_comparison_placeholder()
    print("\nPlaceholder figures created successfully!")
    print("These are illustrative plots - replace with actual experimental data before submission.")
