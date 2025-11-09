"""
MCM Discrepancy Interpretation
===============================

Analyzes ONLY arrays where MCM ON/OFF shows measurable differences.
Filters out arrays with perfect estimation (0.000° both modes).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np

def print_header(text, char='='):
    print(f"\n{char * 80}")
    print(f"{text:^80}")
    print(f"{char * 80}\n")

def print_section(text):
    print(f"\n{text}")
    print("-" * len(text))

# Read results
csv_path = os.path.join('results', 'summaries', 'mcm_comparison_summary.csv')
df = pd.read_csv(csv_path)

# Extract numeric values
df['RMSE_NoMCM_val'] = df['RMSE_NoMCM'].str.replace('°', '').astype(float)
df['RMSE_WithMCM_val'] = df['RMSE_WithMCM'].str.replace('°', '').astype(float)
df['MAE_NoMCM_val'] = df['MAE_NoMCM'].str.replace('°', '').astype(float)
df['MAE_WithMCM_val'] = df['MAE_WithMCM'].str.replace('°', '').astype(float)
df['MaxErr_NoMCM_val'] = df['MaxErr_NoMCM'].str.replace('°', '').astype(float)
df['MaxErr_WithMCM_val'] = df['MaxErr_WithMCM'].str.replace('°', '').astype(float)

df['Degradation_numeric'] = df['Degradation'].str.replace('×', '').apply(
    lambda x: np.nan if x == 'inf' else float(x)
)

# Calculate absolute differences
df['RMSE_diff'] = df['RMSE_WithMCM_val'] - df['RMSE_NoMCM_val']
df['MAE_diff'] = df['MAE_WithMCM_val'] - df['MAE_NoMCM_val']
df['MaxErr_diff'] = df['MaxErr_WithMCM_val'] - df['MaxErr_NoMCM_val']

# Filter: ONLY arrays with measurable discrepancy (RMSE > 0.01° in at least one mode)
threshold = 0.01
discrepancy_arrays = df[
    (df['RMSE_NoMCM_val'] > threshold) | (df['RMSE_WithMCM_val'] > threshold)
].copy()

perfect_arrays = df[
    (df['RMSE_NoMCM_val'] <= threshold) & (df['RMSE_WithMCM_val'] <= threshold)
].copy()

print_header("MCM DISCREPANCY-FOCUSED INTERPRETATION")

print("Interpretation Philosophy:")
print("  • Focus ONLY on arrays showing measurable MCM effects")
print("  • Exclude arrays with perfect estimation (0.000° both modes)")
print("  • Analyze actual performance differences, not theoretical ones")
print(f"  • Threshold: {threshold}° (arrays below this considered 'perfect')")

print_section("ARRAYS WITH MEASURABLE MCM EFFECTS")

if len(discrepancy_arrays) == 0:
    print("\n⚠ NO ARRAYS show MCM discrepancy at current SNR/configuration!")
    print("  All arrays achieve near-perfect estimation regardless of MCM.")
    print("  To see MCM effects, try:")
    print("    - Lower SNR (e.g., 5 dB)")
    print("    - More sources (closer to K_max)")
    print("    - Closer angular spacing")
else:
    print(f"\n✓ Found {len(discrepancy_arrays)} arrays with MCM sensitivity:\n")
    
    # Sort by absolute RMSE difference
    discrepancy_arrays = discrepancy_arrays.sort_values('RMSE_diff', ascending=False)
    
    for idx, row in discrepancy_arrays.iterrows():
        print(f"{'='*80}")
        print(f"Array: {row['Array']}")
        print(f"{'='*80}")
        
        print(f"\nConfiguration:")
        print(f"  Physical Sensors (N): {row['N']}")
        print(f"  Max Detectable (K_max): {row['K_max']}")
        
        print(f"\nPerformance Comparison:")
        print(f"  {'Metric':<15} {'No MCM':>12} {'With MCM':>12} {'Difference':>12} {'Impact':>12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        rmse_impact = "WORSE" if row['RMSE_diff'] > 0 else "BETTER"
        mae_impact = "WORSE" if row['MAE_diff'] > 0 else "BETTER"
        max_impact = "WORSE" if row['MaxErr_diff'] > 0 else "BETTER"
        
        print(f"  {'RMSE':<15} {row['RMSE_NoMCM_val']:>11.3f}° {row['RMSE_WithMCM_val']:>11.3f}° "
              f"{row['RMSE_diff']:>+11.3f}° {rmse_impact:>12}")
        print(f"  {'MAE':<15} {row['MAE_NoMCM_val']:>11.3f}° {row['MAE_WithMCM_val']:>11.3f}° "
              f"{row['MAE_diff']:>+11.3f}° {mae_impact:>12}")
        print(f"  {'Max Error':<15} {row['MaxErr_NoMCM_val']:>11.3f}° {row['MaxErr_WithMCM_val']:>11.3f}° "
              f"{row['MaxErr_diff']:>+11.3f}° {max_impact:>12}")
        
        if not np.isnan(row['Degradation_numeric']):
            print(f"\nDegradation Factor: {row['Degradation_numeric']:.2f}×")
            
            if row['Degradation_numeric'] > 1.5:
                severity = "SEVERE"
                symbol = "⚠️"
            elif row['Degradation_numeric'] > 1.2:
                severity = "MODERATE"
                symbol = "⚠"
            elif row['Degradation_numeric'] < 1.0:
                severity = "IMPROVEMENT"
                symbol = "✓"
            else:
                severity = "MINOR"
                symbol = "ℹ"
            
            print(f"  {symbol} MCM Impact: {severity}")
        
        # Interpretation
        print(f"\n📊 Interpretation:")
        
        if row['RMSE_diff'] > 5.0:
            print(f"  • CRITICAL: MCM causes {row['RMSE_diff']:.1f}° worse RMSE")
            print(f"    → This array is HIGHLY SENSITIVE to mutual coupling")
            print(f"    → Recommendation: Use MCM compensation or choose different geometry")
        elif row['RMSE_diff'] > 2.0:
            print(f"  • SIGNIFICANT: MCM degrades performance by {row['RMSE_diff']:.1f}°")
            print(f"    → Consider MCM effects in system design")
            print(f"    → May need calibration for real hardware")
        elif row['RMSE_diff'] > 0.5:
            print(f"  • MODERATE: MCM causes {row['RMSE_diff']:.1f}° degradation")
            print(f"    → Acceptable for most applications")
            print(f"    → Monitor in production systems")
        elif abs(row['RMSE_diff']) < 0.5:
            print(f"  • MINOR: MCM has minimal effect ({abs(row['RMSE_diff']):.3f}°)")
            print(f"    → This geometry is naturally robust to coupling")
            if row['RMSE_diff'] < 0:
                print(f"    → UNEXPECTED: MCM slightly improves performance!")
        
        if row['RMSE_diff'] < 0:
            print(f"\n  🔬 Special Case: MCM IMPROVES Performance")
            print(f"     • This is UNUSUAL and geometry-specific")
            print(f"     • Possible explanations:")
            print(f"       - MCM matrix acts as regularization")
            print(f"       - Coupling effects cancel with this specific spacing")
            print(f"       - MCM introduces beneficial correlation")
            print(f"     • Enable MCM for this array in practice!")
        
        print()

print_section("ARRAYS WITH NO MEASURABLE MCM EFFECT")

if len(perfect_arrays) > 0:
    print(f"\n✓ {len(perfect_arrays)} arrays show PERFECT estimation (both modes):\n")
    
    for _, row in perfect_arrays.iterrows():
        print(f"  • {row['Array']:<25} (N={row['N']}, K_max={row['K_max']})")
    
    print(f"\n📊 Interpretation:")
    print(f"  • These arrays are ROBUST to mutual coupling")
    print(f"  • Reasons for immunity:")
    print(f"    - Dense/uniform spacing (ULA)")
    print(f"    - Redundant virtual array (Coprime, Nested)")
    print(f"    - Large minimum spacing (Z4: starts at 5λ)")
    print(f"    - Optimal geometry design")
    print(f"  • MCM modeling NOT needed for these arrays at this SNR")
    print(f"  • Safe to use in high-coupling environments")

print_section("COMPARATIVE SUMMARY: DISCREPANCY ANALYSIS")

if len(discrepancy_arrays) > 0:
    print("\nRanked by RMSE Difference (Absolute Impact):\n")
    print(f"{'Rank':<6} {'Array':<20} {'RMSE Δ':>10} {'MAE Δ':>10} {'Max Error Δ':>12} {'Verdict':<15}")
    print(f"{'-'*6} {'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*15}")
    
    for rank, (_, row) in enumerate(discrepancy_arrays.iterrows(), 1):
        verdict = "CRITICAL" if abs(row['RMSE_diff']) > 5 else \
                  "SIGNIFICANT" if abs(row['RMSE_diff']) > 2 else \
                  "MODERATE" if abs(row['RMSE_diff']) > 0.5 else \
                  "MINOR"
        
        if row['RMSE_diff'] < 0:
            verdict = "IMPROVES"
        
        print(f"{rank:<6} {row['Array']:<20} {row['RMSE_diff']:>+9.3f}° {row['MAE_diff']:>+9.3f}° "
              f"{row['MaxErr_diff']:>+11.3f}° {verdict:<15}")

print_section("PRACTICAL RECOMMENDATIONS BASED ON DISCREPANCY")

if len(discrepancy_arrays) > 0:
    # Find most and least affected
    most_affected = discrepancy_arrays.iloc[0]
    least_affected = discrepancy_arrays.iloc[-1]
    
    print(f"\n1. MOST AFFECTED BY MCM:")
    print(f"   Array: {most_affected['Array']}")
    print(f"   Impact: {most_affected['RMSE_diff']:+.3f}° RMSE difference")
    print(f"   Action: {'Enable MCM compensation' if most_affected['RMSE_diff'] > 0 else 'Enable MCM (helps!)'}")
    
    print(f"\n2. LEAST AFFECTED:")
    print(f"   Array: {least_affected['Array']}")
    print(f"   Impact: {least_affected['RMSE_diff']:+.3f}° RMSE difference")
    print(f"   Action: MCM modeling optional")
    
    # Check for improvements
    improved = discrepancy_arrays[discrepancy_arrays['RMSE_diff'] < 0]
    if len(improved) > 0:
        print(f"\n3. ARRAYS WHERE MCM HELPS:")
        for _, row in improved.iterrows():
            improvement_pct = abs(row['RMSE_diff'] / row['RMSE_NoMCM_val']) * 100
            print(f"   • {row['Array']}: {improvement_pct:.1f}% better with MCM")
            print(f"     Recommendation: ENABLE MCM for production use!")
    
    # Arrays needing compensation
    severe = discrepancy_arrays[discrepancy_arrays['RMSE_diff'] > 5.0]
    if len(severe) > 0:
        print(f"\n4. REQUIRE MCM COMPENSATION:")
        for _, row in severe.iterrows():
            print(f"   • {row['Array']}: {row['RMSE_diff']:+.1f}° degradation")
            print(f"     Status: CRITICAL - Unusable without compensation")

print_section("KEY INSIGHTS FROM DISCREPANCY ANALYSIS")

print(f"\n1. TOTAL ARRAYS TESTED: {len(df)}")
print(f"   - Showing MCM effects: {len(discrepancy_arrays)} ({len(discrepancy_arrays)/len(df)*100:.0f}%)")
print(f"   - Perfect (immune): {len(perfect_arrays)} ({len(perfect_arrays)/len(df)*100:.0f}%)")

if len(discrepancy_arrays) > 0:
    avg_rmse_diff = discrepancy_arrays['RMSE_diff'].mean()
    max_rmse_diff = discrepancy_arrays['RMSE_diff'].max()
    min_rmse_diff = discrepancy_arrays['RMSE_diff'].min()
    
    print(f"\n2. RMSE IMPACT STATISTICS (Affected Arrays Only):")
    print(f"   - Average impact: {avg_rmse_diff:+.3f}°")
    print(f"   - Maximum degradation: {max_rmse_diff:+.3f}°")
    print(f"   - Minimum impact: {min_rmse_diff:+.3f}° {'(improvement!)' if min_rmse_diff < 0 else ''}")

print(f"\n3. INTERPRETATION PRINCIPLE:")
print(f"   ✓ Focus on arrays with ACTUAL discrepancy")
print(f"   ✓ Perfect arrays (0.000° both) → MCM irrelevant at this SNR")
print(f"   ✓ Only analyze where MCM makes measurable difference")
print(f"   ✓ Real-world guidance from observable effects")

print_header("END OF DISCREPANCY ANALYSIS")

print("\n💡 CONCLUSION:")
if len(discrepancy_arrays) == 0:
    print("   At SNR=10dB with 3 well-separated sources, MCM effects are minimal.")
    print("   To reveal MCM sensitivity, test at lower SNR or with closer sources.")
elif len(discrepancy_arrays) < len(df) / 2:
    print(f"   Only {len(discrepancy_arrays)}/{len(df)} arrays show MCM sensitivity.")
    print(f"   Focus system design on these {len(discrepancy_arrays)} arrays for MCM considerations.")
else:
    print(f"   Most arrays ({len(discrepancy_arrays)}/{len(df)}) affected by MCM.")
    print("   MCM modeling critical for realistic performance prediction.")

print()
