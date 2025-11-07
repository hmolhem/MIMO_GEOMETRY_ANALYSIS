"""
MCM Effect Summary - Text Report Generator
===========================================

Quick text-based summary of MCM effects across all array geometries.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

def print_header(text, char='='):
    """Print formatted header."""
    print(f"\n{char * 80}")
    print(f"{text:^80}")
    print(f"{char * 80}\n")

def print_section(text):
    """Print section header."""
    print(f"\n{text}")
    print("-" * len(text))

# Read results
csv_path = os.path.join('results', 'summaries', 'mcm_comparison_summary.csv')
if not os.path.exists(csv_path):
    print(f"Error: Results file not found: {csv_path}")
    print("Run 'python analysis_scripts/compare_mcm_effects.py' first")
    sys.exit(1)

df = pd.read_csv(csv_path)

# Extract numeric values
df['RMSE_NoMCM_val'] = df['RMSE_NoMCM'].str.replace('°', '').astype(float)
df['RMSE_WithMCM_val'] = df['RMSE_WithMCM'].str.replace('°', '').astype(float)
df['MAE_NoMCM_val'] = df['MAE_NoMCM'].str.replace('°', '').astype(float)
df['MAE_WithMCM_val'] = df['MAE_WithMCM'].str.replace('°', '').astype(float)
df['MaxErr_NoMCM_val'] = df['MaxErr_NoMCM'].str.replace('°', '').astype(float)
df['MaxErr_WithMCM_val'] = df['MaxErr_WithMCM'].str.replace('°', '').astype(float)

# Parse degradation
df['Degradation_numeric'] = df['Degradation'].str.replace('×', '').apply(
    lambda x: np.nan if x == 'inf' else float(x)
)

# Main report
print_header("MCM EFFECT ANALYSIS - SUMMARY REPORT")

print("Test Configuration:")
print("  SNR: 10 dB  |  Sources: 3 at [-30°, 0°, 30°]  |  Snapshots: 200  |  Trials: 50")
print("  MCM Model: Exponential decay (c1=0.3, alpha=0.5)")

# Full comparison table
print_section("COMPLETE COMPARISON TABLE")
print("\n" + df.to_string(index=False, max_colwidth=20))

# Separate arrays by behavior
print_section("ARRAYS BY MCM SENSITIVITY")

perfect = df[df['RMSE_NoMCM_val'] == 0].copy()
affected = df[df['RMSE_NoMCM_val'] > 0].copy()

print("\n✓ PERFECT ESTIMATION (0.000° RMSE, both MCM ON/OFF):")
if len(perfect) > 0:
    for _, row in perfect.iterrows():
        print(f"  • {row['Array']:<20} (N={row['N']}, K_max={row['K_max']})")
else:
    print("  None")

if len(affected) > 0:
    # Sort by degradation
    affected = affected.sort_values('Degradation_numeric', ascending=False)
    
    print("\n⚠ AFFECTED BY MCM (sorted by degradation):")
    for _, row in affected.iterrows():
        deg = row['Degradation_numeric']
        symbol = "⚠️" if deg > 1.5 else "⚠" if deg > 1.2 else "✓"
        status = "SEVERE" if deg > 1.5 else "MODERATE" if deg > 1.2 else "MINOR"
        
        print(f"\n  {symbol} {row['Array']:<20} [{status}]")
        print(f"     No MCM:   {row['RMSE_NoMCM_val']:6.3f}° RMSE  |  {row['MAE_NoMCM_val']:6.3f}° MAE  |  {row['MaxErr_NoMCM_val']:6.3f}° Max")
        print(f"     With MCM: {row['RMSE_WithMCM_val']:6.3f}° RMSE  |  {row['MAE_WithMCM_val']:6.3f}° MAE  |  {row['MaxErr_WithMCM_val']:6.3f}° Max")
        print(f"     Degradation: {deg:.2f}× worse")

# Key findings
print_section("KEY FINDINGS")

if len(affected) > 0:
    most = affected.iloc[0]
    least = affected.iloc[-1]
    
    print(f"\n1. MOST AFFECTED: {most['Array']}")
    print(f"   - Degradation: {most['Degradation_numeric']:.2f}× worse RMSE")
    print(f"   - {most['RMSE_NoMCM_val']:.3f}° → {most['RMSE_WithMCM_val']:.3f}°")
    
    print(f"\n2. LEAST AFFECTED: {least['Array']}")
    print(f"   - Degradation: {least['Degradation_numeric']:.2f}× worse RMSE")
    print(f"   - {least['RMSE_NoMCM_val']:.3f}° → {least['RMSE_WithMCM_val']:.3f}°")
    
    # Check for improvements
    improved = affected[affected['Degradation_numeric'] < 1.0]
    if len(improved) > 0:
        print(f"\n3. UNEXPECTED: MCM IMPROVES PERFORMANCE!")
        for _, row in improved.iterrows():
            improvement = (1.0 - row['Degradation_numeric']) * 100
            print(f"   - {row['Array']}: {improvement:.1f}% better with MCM")
            print(f"     {row['RMSE_NoMCM_val']:.3f}° → {row['RMSE_WithMCM_val']:.3f}°")
    
    avg_deg = affected['Degradation_numeric'].mean()
    print(f"\n4. AVERAGE DEGRADATION: {avg_deg:.2f}× worse RMSE across affected arrays")

print(f"\n5. ROBUST ARRAYS: {len(perfect)} arrays show perfect estimation regardless of MCM")
print(f"   (Dense geometries: ULA, TCA, Z3_1, Z4)")

# Recommendations
print_section("PRACTICAL RECOMMENDATIONS")

print("\n✓ CHOOSE THESE ARRAYS FOR ROBUSTNESS:")
robust = df[(df['RMSE_NoMCM_val'] == 0) | (df['Degradation_numeric'] < 1.2)]
for _, row in robust.iterrows():
    reason = "Perfect estimation" if row['RMSE_NoMCM_val'] == 0 else f"Low degradation ({row['Degradation_numeric']:.2f}×)"
    print(f"  • {row['Array']:<20} - {reason}")

print("\n⚠ AVOID IN HIGH-COUPLING ENVIRONMENTS:")
sensitive = affected[affected['Degradation_numeric'] > 1.3]
if len(sensitive) > 0:
    for _, row in sensitive.iterrows():
        print(f"  • {row['Array']:<20} - {row['Degradation_numeric']:.2f}× degradation")
else:
    print("  None (all arrays reasonably robust)")

print("\n🔧 WHEN TO ENABLE MCM:")
print("  • Realistic hardware simulation of sparse arrays")
print("  • Arrays: Z1, Z3_2 (if coupling coefficients known)")
print("  • Planning MCM compensation algorithms")
print("  • Comparing theoretical vs. practical performance")

print("\n🎯 WHEN TO DISABLE MCM:")
print("  • Best-case performance benchmarking")
print("  • Dense/robust arrays: ULA, TCA, Z4")
print("  • High SNR environments (>15 dB)")
print("  • Algorithm development phase (ideal assumptions)")

# Output files
print_section("OUTPUT FILES")
print("\n📁 Generated Files:")
print(f"  • {csv_path}")
print(f"  • results/MCM_EFFECT_ANALYSIS.md (detailed report)")
print(f"  • results/MCM_COMPARISON_README.md (quick reference)")
print(f"  • results/plots/mcm_comparison_analysis.png (visualization)")

print_header("END OF REPORT")

print("\n💡 TIP: For detailed analysis, see results/MCM_EFFECT_ANALYSIS.md")
print("💡 TIP: For visualization, open results/plots/mcm_comparison_analysis.png\n")
