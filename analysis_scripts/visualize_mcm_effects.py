"""
MCM Effect Visualization
========================

Creates visual comparison of MCM effects across all array geometries.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Read the results
csv_path = os.path.join('results', 'summaries', 'mcm_comparison_summary.csv')
df = pd.read_csv(csv_path)

# Extract numeric RMSE values
df['RMSE_NoMCM_val'] = df['RMSE_NoMCM'].str.replace('°', '').astype(float)
df['RMSE_WithMCM_val'] = df['RMSE_WithMCM'].str.replace('°', '').astype(float)
df['Degradation_val'] = df['Degradation'].str.replace('×', '')

# Handle inf values
df['Degradation_numeric'] = df['Degradation_val'].apply(
    lambda x: np.nan if x == 'inf' else float(x)
)

# Create figure with 3 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MCM Effect Analysis Across Array Geometries', fontsize=16, fontweight='bold')

# Plot 1: RMSE Comparison (Bar chart)
ax1 = axes[0, 0]
x = np.arange(len(df))
width = 0.35
bars1 = ax1.bar(x - width/2, df['RMSE_NoMCM_val'], width, label='No MCM', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x + width/2, df['RMSE_WithMCM_val'], width, label='With MCM', color='#e74c3c', alpha=0.8)
ax1.set_xlabel('Array Type', fontweight='bold')
ax1.set_ylabel('RMSE (degrees)', fontweight='bold')
ax1.set_title('RMSE: MCM OFF vs MCM ON')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Array'], rotation=45, ha='right', fontsize=9)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, max(df['RMSE_WithMCM_val'].max(), df['RMSE_NoMCM_val'].max()) * 1.1])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.01:  # Only label non-zero bars
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}°', ha='center', va='bottom', fontsize=7)

# Plot 2: Degradation Factor
ax2 = axes[0, 1]
# Only plot arrays with numeric degradation
df_plot = df[df['Degradation_numeric'].notna()].copy()
colors = ['#e74c3c' if d > 1.2 else '#f39c12' if d > 1.0 else '#2ecc71' 
          for d in df_plot['Degradation_numeric']]
bars = ax2.barh(df_plot['Array'], df_plot['Degradation_numeric'], color=colors, alpha=0.8)
ax2.set_xlabel('Degradation Factor (MCM/No-MCM)', fontweight='bold')
ax2.set_ylabel('Array Type', fontweight='bold')
ax2.set_title('MCM Degradation Factor\n(< 1.0 = improvement, > 1.0 = degradation)')
ax2.axvline(x=1.0, color='black', linestyle='--', linewidth=1, label='No Change')
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, df_plot['Degradation_numeric'])):
    ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
            f'{val:.2f}×', va='center', fontsize=9)

# Plot 3: Performance by K_max
ax3 = axes[1, 0]
scatter1 = ax3.scatter(df['K_max'], df['RMSE_NoMCM_val'], s=150, alpha=0.7, 
                      c='#2ecc71', edgecolors='black', linewidth=1.5, label='No MCM')
scatter2 = ax3.scatter(df['K_max'], df['RMSE_WithMCM_val'], s=150, alpha=0.7,
                      c='#e74c3c', marker='s', edgecolors='black', linewidth=1.5, label='With MCM')
ax3.set_xlabel('K_max (Maximum Detectable Sources)', fontweight='bold')
ax3.set_ylabel('RMSE (degrees)', fontweight='bold')
ax3.set_title('Performance vs. Array Capacity')
ax3.legend()
ax3.grid(alpha=0.3)

# Annotate interesting points
for i, row in df.iterrows():
    if row['RMSE_WithMCM_val'] > 10:  # Label high-error points
        ax3.annotate(row['Array'], 
                    xy=(row['K_max'], row['RMSE_WithMCM_val']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)

# Plot 4: Summary table
ax4 = axes[1, 1]
ax4.axis('off')

# Create summary statistics
summary_text = "MCM EFFECT SUMMARY\n" + "="*50 + "\n\n"

# Most affected
most_affected = df.loc[df['Degradation_numeric'].idxmax()] if df['Degradation_numeric'].notna().any() else None
if most_affected is not None:
    summary_text += f"MOST AFFECTED:\n"
    summary_text += f"  {most_affected['Array']}\n"
    summary_text += f"  Degradation: {most_affected['Degradation']}\n"
    summary_text += f"  {most_affected['RMSE_NoMCM']} → {most_affected['RMSE_WithMCM']}\n\n"

# Least affected (excluding perfect cases)
df_nonzero = df[(df['Degradation_numeric'].notna()) & (df['Degradation_numeric'] > 0)]
if len(df_nonzero) > 0:
    least_affected = df_nonzero.loc[df_nonzero['Degradation_numeric'].idxmin()]
    summary_text += f"LEAST AFFECTED:\n"
    summary_text += f"  {least_affected['Array']}\n"
    summary_text += f"  Degradation: {least_affected['Degradation']}\n"
    summary_text += f"  {least_affected['RMSE_NoMCM']} → {least_affected['RMSE_WithMCM']}\n\n"

# Perfect arrays
perfect = df[df['RMSE_NoMCM_val'] == 0]
summary_text += f"PERFECT ESTIMATION (0.000° RMSE):\n"
for _, row in perfect.iterrows():
    summary_text += f"  • {row['Array']} (K_max={row['K_max']})\n"

summary_text += f"\nTEST CONDITIONS:\n"
summary_text += f"  • SNR: 10 dB\n"
summary_text += f"  • Sources: 3 at [-30°, 0°, 30°]\n"
summary_text += f"  • Snapshots: 200\n"
summary_text += f"  • Trials: 50 (Monte Carlo)\n"
summary_text += f"  • MCM: Exponential (c1=0.3, α=0.5)\n"

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save figure
output_path = os.path.join('results', 'plots', 'mcm_comparison_analysis.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_path}")
print("  Open the file to view the charts.")

# Don't show interactively (no display available)
# plt.show()
