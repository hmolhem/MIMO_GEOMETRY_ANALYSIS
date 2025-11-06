"""
Plot ALSS comparison for hard case (delta=13 degrees)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    'lines.linewidth': 2.5,
    'lines.markersize': 9,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Load data
print("Loading data...")
df_off = pd.read_csv('results/bench/alss_off_hard.csv')
df_on = pd.read_csv('results/bench/alss_on_hard.csv')

# Aggregate by SNR
print("Computing statistics...")
stats_off = df_off.groupby('SNR_dB').agg({
    'rmse_deg': 'mean',
    'resolved': 'mean'
}).reset_index()

stats_on = df_on.groupby('SNR_dB').agg({
    'rmse_deg': 'mean',
    'resolved': 'mean'
}).reset_index()

# Create output directory
output_dir = Path('results/bench/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Plot RMSE comparison
print("\nGenerating RMSE plot...")
fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

snr = stats_off['SNR_dB']
rmse_off = stats_off['rmse_deg']
rmse_on = stats_on['rmse_deg']

ax.plot(snr, rmse_off, 'o-', color='#E74C3C', 
        label='Baseline', linewidth=2.5, markersize=9,
        markeredgewidth=1.5, markeredgecolor='white')
ax.plot(snr, rmse_on, 's-', color='#3498DB',
        label='ALSS', linewidth=2.5, markersize=9,
        markeredgewidth=1.5, markeredgecolor='white')

# Calculate improvement at SNR=10dB
improvement = (rmse_on.iloc[-1] - rmse_off.iloc[-1]) / rmse_off.iloc[-1] * 100

# Add improvement annotation
ax.text(0.97, 0.97, f'{improvement:+.1f}%',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=11, fontweight='bold', color='#27AE60',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                 edgecolor='#27AE60', linewidth=1.5, alpha=0.9))

ax.set_xlabel('SNR (dB)', fontweight='bold')
ax.set_ylabel('RMSE (degrees)', fontweight='bold')
ax.set_title('RMSE vs SNR (Δθ = 13°, Hard Case)', fontweight='bold')
ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)
ax.grid(True, alpha=0.25, linestyle='--')

plt.tight_layout()

pdf_path = output_dir / 'alss_rmse_hard_delta13.pdf'
png_path = output_dir / 'alss_rmse_hard_delta13.png'

fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
fig.savefig(png_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {pdf_path}")
print(f"✓ Saved: {png_path}")

# Plot Resolve Rate comparison
print("\nGenerating Resolve Rate plot...")
fig, ax = plt.subplots(figsize=(5, 3.5), dpi=300)

resolve_off = stats_off['resolved'] * 100
resolve_on = stats_on['resolved'] * 100

ax.plot(snr, resolve_off, 'o-', color='#E74C3C',
        label='Baseline', linewidth=2.5, markersize=9,
        markeredgewidth=1.5, markeredgecolor='white')
ax.plot(snr, resolve_on, 's-', color='#3498DB',
        label='ALSS', linewidth=2.5, markersize=9,
        markeredgewidth=1.5, markeredgecolor='white')

# Add note about zero resolve rate
if resolve_off.max() == 0 and resolve_on.max() == 0:
    ax.text(0.5, 0.5, 'Sources not resolved\n(Δθ=13° too close)',
            transform=ax.transAxes, ha='center', va='center',
            fontsize=11, style='italic', color='#7F8C8D',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='#BDC3C7', linewidth=1.0, alpha=0.8))

ax.set_xlabel('SNR (dB)', fontweight='bold')
ax.set_ylabel('Resolve Rate (%)', fontweight='bold')
ax.set_title('Resolve Rate vs SNR (Δθ = 13°, Hard Case)', fontweight='bold')
ax.set_ylim([0, 105])
ax.legend(loc='lower right', frameon=True, fancybox=False, shadow=False)
ax.grid(True, alpha=0.25, linestyle='--')

plt.tight_layout()

pdf_path = output_dir / 'alss_resolve_hard_delta13.pdf'
png_path = output_dir / 'alss_resolve_hard_delta13.png'

fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
fig.savefig(png_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"✓ Saved: {pdf_path}")
print(f"✓ Saved: {png_path}")

print("\n✅ Done! Opening figures folder...")
