"""
ALSS Role in MCM Scenarios
===========================

Analyzes the role of ALSS (Adaptive Lag-Selective Shrinkage) in two critical scenarios:

Scenario 1: ALSS effectiveness when MCM is enabled
Scenario 2: ALSS impact on arrays showing MCM ON/OFF discrepancy

Purpose: Understand if ALSS can mitigate MCM-induced degradation
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd

# Import DOA estimation with ALSS support
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))
from radarpy.doa.coarray_music import CoarrayMUSIC
from radarpy.algorithms.alss import apply_alss

# Import array processors
from geometry_processors.z1_processor import Z1ArrayProcessor
from geometry_processors.z3_2_processor import Z3_2ArrayProcessor
from geometry_processors.z5_processor import Z5ArrayProcessor


def print_header(text, char='='):
    print(f"\n{char * 80}")
    print(f"{text:^80}")
    print(f"{char * 80}\n")


def print_section(text):
    print(f"\n{text}")
    print("-" * len(text))


def run_scenario1_alss_with_mcm():
    """
    Scenario 1: ALSS Performance When MCM is Enabled
    
    Question: Does ALSS help when mutual coupling is present?
    Hypothesis: ALSS should reduce noise variance even under coupling distortion
    """
    print_header("SCENARIO 1: ALSS EFFECTIVENESS WITH MCM ENABLED")
    
    print("Research Question:")
    print("  Can ALSS (Adaptive Lag-Selective Shrinkage) mitigate performance")
    print("  degradation caused by Mutual Coupling Matrix (MCM)?")
    
    print("\nHypothesis:")
    print("  ALSS reduces noise variance in lag estimates, which should help")
    print("  even when MCM introduces systematic coupling distortion.")
    
    # Test arrays that show MCM sensitivity
    test_arrays = [
        ("Z3_2 (N=6)", Z3_2ArrayProcessor(N=6, d=1.0)),
        ("Z1 (N=7)", Z1ArrayProcessor(N=7, d=1.0)),
        ("Z5 (N=7)", Z5ArrayProcessor(N=7, d=1.0))
    ]
    
    # Test configuration
    true_angles = [-30, 0, 30]
    snr_db = 10
    snapshots = 200
    num_trials = 30
    
    # MCM parameters
    mcm_params = {'c1': 0.3, 'alpha': 0.5}
    
    results = []
    
    for array_name, processor in test_arrays:
        print_section(f"Testing: {array_name}")
        
        # Get array geometry
        data = processor.run_full_analysis()
        positions = data.sensors_positions
        N = data.num_sensors
        
        print(f"  Sensors: {N}, Positions: {positions}")
        print(f"  Testing K=3 sources at {true_angles}°")
        print(f"  SNR: {snr_db} dB, Snapshots: {snapshots}, Trials: {num_trials}")
        
        # Storage for 4 conditions
        rmse_mcm_noalss = []
        rmse_mcm_alss = []
        
        print(f"\n  Running {num_trials} trials...")
        
        for trial in range(num_trials):
            if trial % 10 == 0:
                print(f"    Trial {trial+1}/{num_trials}...", end='\r')
            
            try:
                # Condition 1: MCM ON, ALSS OFF
                estimator_mcm_noalss = CoarrayMUSIC(
                    sensor_positions=positions,
                    wavelength=2.0,
                    enable_mcm=True,
                    mcm_model='exponential',
                    mcm_params=mcm_params,
                    enable_alss=False
                )
                X = estimator_mcm_noalss.simulate_signals(
                    true_angles=true_angles, SNR_dB=snr_db, snapshots=snapshots
                )
                est = estimator_mcm_noalss.estimate(X, K_sources=3)
                
                # Calculate RMSE
                est_sorted = np.sort(est)
                true_sorted = np.sort(true_angles)
                rmse = np.sqrt(np.mean((est_sorted - true_sorted)**2))
                rmse_mcm_noalss.append(rmse)
                
                # Condition 2: MCM ON, ALSS ON
                estimator_mcm_alss = CoarrayMUSIC(
                    sensor_positions=positions,
                    wavelength=2.0,
                    enable_mcm=True,
                    mcm_model='exponential',
                    mcm_params=mcm_params,
                    enable_alss=True,
                    alss_mode='zero',
                    alss_tau=1.0
                )
                X = estimator_mcm_alss.simulate_signals(
                    true_angles=true_angles, SNR_dB=snr_db, snapshots=snapshots
                )
                est = estimator_mcm_alss.estimate(X, K_sources=3)
                
                est_sorted = np.sort(est)
                rmse = np.sqrt(np.mean((est_sorted - true_sorted)**2))
                rmse_mcm_alss.append(rmse)
                
            except Exception as e:
                print(f"\n    ✗ Trial {trial+1} failed: {e}")
                continue
        
        print(f"    Completed {num_trials} trials      ")
        
        # Calculate statistics
        mean_mcm_noalss = np.mean(rmse_mcm_noalss)
        std_mcm_noalss = np.std(rmse_mcm_noalss)
        
        mean_mcm_alss = np.mean(rmse_mcm_alss)
        std_mcm_alss = np.std(rmse_mcm_alss)
        
        improvement = (mean_mcm_noalss - mean_mcm_alss) / mean_mcm_noalss * 100
        
        # Display results
        print(f"\n  Results:")
        print(f"    MCM ON, ALSS OFF:  {mean_mcm_noalss:.3f}° ± {std_mcm_noalss:.3f}°")
        print(f"    MCM ON, ALSS ON:   {mean_mcm_alss:.3f}° ± {std_mcm_alss:.3f}°")
        print(f"    ALSS Improvement:  {improvement:+.1f}%")
        
        if improvement > 5:
            print(f"    ✓ SIGNIFICANT: ALSS helps under MCM!")
        elif improvement > 0:
            print(f"    ✓ MINOR: ALSS slightly helps")
        else:
            print(f"    ✗ NO BENEFIT: ALSS doesn't help under MCM")
        
        results.append({
            'Array': array_name,
            'MCM_NoALSS_RMSE': mean_mcm_noalss,
            'MCM_ALSS_RMSE': mean_mcm_alss,
            'ALSS_Improvement_%': improvement,
            'MCM_NoALSS_Std': std_mcm_noalss,
            'MCM_ALSS_Std': std_mcm_alss
        })
    
    # Summary
    print_section("SCENARIO 1 SUMMARY")
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    # Interpretation
    print("\n📊 Interpretation:")
    avg_improvement = df['ALSS_Improvement_%'].mean()
    
    if avg_improvement > 5:
        print(f"  ✓ ALSS provides {avg_improvement:.1f}% average improvement under MCM")
        print("  → ALSS can partially mitigate MCM-induced degradation")
        print("  → Recommendation: Enable ALSS when MCM is present")
    elif avg_improvement > 0:
        print(f"  ≈ ALSS provides minor {avg_improvement:.1f}% improvement")
        print("  → ALSS helps slightly but not dramatically")
        print("  → Optional feature, not critical for MCM scenarios")
    else:
        print(f"  ✗ ALSS shows {avg_improvement:.1f}% impact (not helpful)")
        print("  → ALSS cannot overcome MCM systematic errors")
        print("  → Focus on MCM compensation instead")
    
    return df


def run_scenario2_alss_discrepancy_arrays():
    """
    Scenario 2: ALSS Impact on Arrays with MCM ON/OFF Discrepancy
    
    Question: For arrays showing MCM sensitivity, can ALSS bridge the gap?
    Focus: Only arrays with measurable MCM discrepancy (Z1, Z3_2, Z5)
    """
    print_header("SCENARIO 2: ALSS ON ARRAYS WITH MCM DISCREPANCY")
    
    print("Research Question:")
    print("  For arrays showing MCM ON/OFF discrepancy, can ALSS reduce")
    print("  the performance gap between ideal (No MCM) and realistic (MCM ON)?")
    
    print("\nFocus Arrays:")
    print("  • Z3_2: +6.0° degradation with MCM (CRITICAL)")
    print("  • Z1: +2.3° degradation with MCM (SIGNIFICANT)")
    print("  • Z5: -0.1° improvement with MCM (UNEXPECTED)")
    
    # Test only arrays with discrepancy
    test_arrays = [
        ("Z3_2 (N=6)", Z3_2ArrayProcessor(N=6, d=1.0)),
        ("Z1 (N=7)", Z1ArrayProcessor(N=7, d=1.0)),
        ("Z5 (N=7)", Z5ArrayProcessor(N=7, d=1.0))
    ]
    
    # Test configuration
    true_angles = [-30, 0, 30]
    snr_db = 10
    snapshots = 200
    num_trials = 30
    
    # MCM parameters
    mcm_params = {'c1': 0.3, 'alpha': 0.5}
    
    results = []
    
    for array_name, processor in test_arrays:
        print_section(f"Testing: {array_name}")
        
        # Get array geometry
        data = processor.run_full_analysis()
        positions = data.sensors_positions
        N = data.num_sensors
        
        print(f"  Sensors: {N}, Positions: {positions}")
        
        # Storage for 4 conditions
        rmse_nomcm_noalss = []
        rmse_nomcm_alss = []
        rmse_mcm_noalss = []
        rmse_mcm_alss = []
        
        print(f"\n  Running {num_trials} trials (4 conditions)...")
        
        for trial in range(num_trials):
            if trial % 10 == 0:
                print(f"    Trial {trial+1}/{num_trials}...", end='\r')
            
            try:
                # Condition 1: No MCM, No ALSS (baseline)
                estimator = CoarrayMUSIC(
                    sensor_positions=positions,
                    wavelength=2.0,
                    enable_mcm=False,
                    enable_alss=False
                )
                X = estimator.simulate_signals(
                    true_angles=true_angles, SNR_dB=snr_db, snapshots=snapshots
                )
                est = estimator.estimate(X, K_sources=3)
                est_sorted = np.sort(est)
                true_sorted = np.sort(true_angles)
                rmse = np.sqrt(np.mean((est_sorted - true_sorted)**2))
                rmse_nomcm_noalss.append(rmse)
                
                # Condition 2: No MCM, ALSS ON
                estimator = CoarrayMUSIC(
                    sensor_positions=positions,
                    wavelength=2.0,
                    enable_mcm=False,
                    enable_alss=True,
                    alss_mode='zero',
                    alss_tau=1.0
                )
                X = estimator.simulate_signals(
                    true_angles=true_angles, SNR_dB=snr_db, snapshots=snapshots
                )
                est = estimator.estimate(X, K_sources=3)
                est_sorted = np.sort(est)
                rmse = np.sqrt(np.mean((est_sorted - true_sorted)**2))
                rmse_nomcm_alss.append(rmse)
                
                # Condition 3: MCM ON, No ALSS
                estimator = CoarrayMUSIC(
                    sensor_positions=positions,
                    wavelength=2.0,
                    enable_mcm=True,
                    mcm_model='exponential',
                    mcm_params=mcm_params,
                    enable_alss=False
                )
                X = estimator.simulate_signals(
                    true_angles=true_angles, SNR_dB=snr_db, snapshots=snapshots
                )
                est = estimator.estimate(X, K_sources=3)
                est_sorted = np.sort(est)
                rmse = np.sqrt(np.mean((est_sorted - true_sorted)**2))
                rmse_mcm_noalss.append(rmse)
                
                # Condition 4: MCM ON, ALSS ON (BEST EFFORT)
                estimator = CoarrayMUSIC(
                    sensor_positions=positions,
                    wavelength=2.0,
                    enable_mcm=True,
                    mcm_model='exponential',
                    mcm_params=mcm_params,
                    enable_alss=True,
                    alss_mode='zero',
                    alss_tau=1.0
                )
                X = estimator.simulate_signals(
                    true_angles=true_angles, SNR_dB=snr_db, snapshots=snapshots
                )
                est = estimator.estimate(X, K_sources=3)
                est_sorted = np.sort(est)
                rmse = np.sqrt(np.mean((est_sorted - true_sorted)**2))
                rmse_mcm_alss.append(rmse)
                
            except Exception as e:
                print(f"\n    ✗ Trial {trial+1} failed: {e}")
                continue
        
        print(f"    Completed {num_trials} trials      ")
        
        # Calculate statistics
        mean_nomcm_noalss = np.mean(rmse_nomcm_noalss)
        mean_nomcm_alss = np.mean(rmse_nomcm_alss)
        mean_mcm_noalss = np.mean(rmse_mcm_noalss)
        mean_mcm_alss = np.mean(rmse_mcm_alss)
        
        std_nomcm_noalss = np.std(rmse_nomcm_noalss)
        std_nomcm_alss = np.std(rmse_nomcm_alss)
        std_mcm_noalss = np.std(rmse_mcm_noalss)
        std_mcm_alss = np.std(rmse_mcm_alss)
        
        # Calculate key metrics
        mcm_degradation = mean_mcm_noalss - mean_nomcm_noalss
        alss_helps_ideal = mean_nomcm_noalss - mean_nomcm_alss
        alss_helps_mcm = mean_mcm_noalss - mean_mcm_alss
        gap_reduction = (mean_nomcm_noalss - mean_mcm_alss) / mcm_degradation * 100 if mcm_degradation != 0 else 0
        
        # Display results
        print(f"\n  Results (4 Conditions):")
        print(f"    {'Condition':<25} {'RMSE':<12} {'Std Dev':<12}")
        print(f"    {'-'*25} {'-'*12} {'-'*12}")
        print(f"    {'1. No MCM, No ALSS':<25} {mean_nomcm_noalss:>10.3f}° {std_nomcm_noalss:>10.3f}°")
        print(f"    {'2. No MCM, ALSS ON':<25} {mean_nomcm_alss:>10.3f}° {std_nomcm_alss:>10.3f}°")
        print(f"    {'3. MCM ON, No ALSS':<25} {mean_mcm_noalss:>10.3f}° {std_mcm_noalss:>10.3f}°")
        print(f"    {'4. MCM ON, ALSS ON':<25} {mean_mcm_alss:>10.3f}° {std_mcm_alss:>10.3f}°")
        
        print(f"\n  Analysis:")
        print(f"    MCM Degradation (3-1):     {mcm_degradation:+.3f}°")
        print(f"    ALSS Help (Ideal, 1-2):    {alss_helps_ideal:+.3f}°")
        print(f"    ALSS Help (MCM, 3-4):      {alss_helps_mcm:+.3f}°")
        print(f"    Gap Reduction (4 vs 1):    {gap_reduction:.1f}%")
        
        # Interpretation
        if gap_reduction > 50:
            print(f"    ✓ EXCELLENT: ALSS recovers >{gap_reduction:.0f}% of MCM loss!")
        elif gap_reduction > 25:
            print(f"    ✓ GOOD: ALSS recovers {gap_reduction:.0f}% of MCM loss")
        elif gap_reduction > 0:
            print(f"    ≈ MINOR: ALSS recovers {gap_reduction:.0f}% of MCM loss")
        else:
            print(f"    ✗ NO HELP: ALSS doesn't bridge MCM gap")
        
        results.append({
            'Array': array_name,
            'NoMCM_NoALSS': mean_nomcm_noalss,
            'NoMCM_ALSS': mean_nomcm_alss,
            'MCM_NoALSS': mean_mcm_noalss,
            'MCM_ALSS': mean_mcm_alss,
            'MCM_Degradation': mcm_degradation,
            'ALSS_Help_Ideal': alss_helps_ideal,
            'ALSS_Help_MCM': alss_helps_mcm,
            'Gap_Reduction_%': gap_reduction
        })
    
    # Summary
    print_section("SCENARIO 2 SUMMARY")
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    # Overall interpretation
    print("\n📊 Overall Interpretation:")
    avg_gap_reduction = df['Gap_Reduction_%'].mean()
    
    print(f"\n  Average Gap Reduction: {avg_gap_reduction:.1f}%")
    
    if avg_gap_reduction > 50:
        print("  ✓ ALSS HIGHLY EFFECTIVE: Recovers majority of MCM-induced loss")
        print("  → ALSS is a viable compensation technique for MCM effects")
        print("  → Recommendation: Always enable ALSS with MCM-sensitive arrays")
    elif avg_gap_reduction > 25:
        print("  ✓ ALSS MODERATELY EFFECTIVE: Partially mitigates MCM degradation")
        print("  → ALSS provides meaningful but not complete compensation")
        print("  → Consider ALSS as part of multi-strategy approach")
    elif avg_gap_reduction > 0:
        print("  ≈ ALSS MINIMALLY EFFECTIVE: Marginal impact on MCM gap")
        print("  → ALSS helps but not sufficient alone")
        print("  → Pair with other compensation techniques")
    else:
        print("  ✗ ALSS INEFFECTIVE: Cannot compensate for MCM")
        print("  → MCM systematic errors dominate over ALSS noise reduction")
        print("  → Need dedicated MCM compensation algorithms")
    
    # Array-specific insights
    print("\n  Array-Specific Insights:")
    for _, row in df.iterrows():
        print(f"\n  {row['Array']}:")
        print(f"    - MCM causes {row['MCM_Degradation']:+.3f}° degradation")
        print(f"    - ALSS recovers {row['Gap_Reduction_%']:.1f}% of this loss")
        if row['Gap_Reduction_%'] > 50:
            print(f"    → Excellent ALSS candidate!")
        elif row['Gap_Reduction_%'] < 0:
            print(f"    → ALSS not helpful for this array")
    
    return df


def main():
    """Run both scenarios."""
    print_header("ALSS ROLE IN MCM SCENARIOS: COMPREHENSIVE ANALYSIS")
    
    print("This analysis explores the interaction between:")
    print("  • ALSS: Adaptive Lag-Selective Shrinkage (your innovation)")
    print("  • MCM: Mutual Coupling Matrix effects")
    print("  • Array Geometry: Focus on arrays with MCM sensitivity")
    
    # Run Scenario 1
    df_s1 = run_scenario1_alss_with_mcm()
    
    # Run Scenario 2
    df_s2 = run_scenario2_alss_discrepancy_arrays()
    
    # Combined conclusions
    print_header("FINAL CONCLUSIONS")
    
    print("\n1. SCENARIO 1 (ALSS with MCM ON):")
    avg_improvement_s1 = df_s1['ALSS_Improvement_%'].mean()
    print(f"   - Average ALSS improvement under MCM: {avg_improvement_s1:+.1f}%")
    if avg_improvement_s1 > 5:
        print("   - Verdict: ✓ ALSS helps under MCM")
    else:
        print("   - Verdict: ≈ ALSS has minimal impact under MCM")
    
    print("\n2. SCENARIO 2 (ALSS bridging MCM gap):")
    avg_gap_s2 = df_s2['Gap_Reduction_%'].mean()
    print(f"   - Average gap reduction: {avg_gap_s2:.1f}%")
    if avg_gap_s2 > 50:
        print("   - Verdict: ✓ ALSS highly effective for MCM compensation")
    elif avg_gap_s2 > 25:
        print("   - Verdict: ✓ ALSS moderately effective")
    else:
        print("   - Verdict: ≈ ALSS provides limited compensation")
    
    print("\n3. PRACTICAL RECOMMENDATIONS:")
    
    if avg_improvement_s1 > 5 or avg_gap_s2 > 25:
        print("   ✓ Enable ALSS when using MCM-sensitive arrays")
        print("   ✓ ALSS provides meaningful performance recovery")
        print("   ✓ Computational cost is minimal (negligible overhead)")
    else:
        print("   • ALSS has limited impact on MCM-induced errors")
        print("   • Focus on alternative MCM compensation techniques")
        print("   • ALSS remains valuable for low-SNR scenarios")
    
    print("\n4. KEY INSIGHT:")
    print("   ALSS addresses NOISE variance, MCM introduces BIAS.")
    print("   → ALSS can help if noise dominates over coupling distortion")
    print("   → For severe coupling, dedicated MCM compensation needed")
    
    # Save results
    output_dir = os.path.join('results', 'summaries')
    os.makedirs(output_dir, exist_ok=True)
    
    df_s1.to_csv(os.path.join(output_dir, 'alss_scenario1_mcm_on.csv'), index=False)
    df_s2.to_csv(os.path.join(output_dir, 'alss_scenario2_discrepancy.csv'), index=False)
    
    print(f"\n✓ Results saved:")
    print(f"  • {os.path.join(output_dir, 'alss_scenario1_mcm_on.csv')}")
    print(f"  • {os.path.join(output_dir, 'alss_scenario2_discrepancy.csv')}")
    
    print_header("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
