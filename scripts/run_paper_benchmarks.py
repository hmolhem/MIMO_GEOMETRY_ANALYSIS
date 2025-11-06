# scripts/run_paper_benchmarks.py
"""
Paper-ready benchmarks with specified parameters:
- Geometry: Z5 (primary), Z4/ULA (secondary)
- ALSS: mode='zero', τ=1.0, ℓ₀=3
- Trials: 200-500 per point for tight confidence intervals
- Tolerances: Resolve if both peaks within ±1° and distinct by ≥0.5°
- Grid: 0.05° global scan + 0.01° local refine around peaks
- CIs: Bootstrap RMSE, Wilson binomial for resolve%
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import pandas as pd
from scipy import stats
import time
import warnings

# Import existing benchmark infrastructure
from core.radarpy.algorithms.coarray_music import estimate_doa_coarray_music
from core.radarpy.algorithms.crb import crb_pair_worst_deg
from core.analysis_scripts.run_benchmarks import (
    get_array_geometry, simulate_snapshots
)

def resolve_tolerance_check(est_doas, true_doas, position_tol_deg=1.0, separation_tol_deg=0.5):
    """
    Check if estimated DOAs are "resolved" based on paper criteria:
    - Both peaks within ±position_tol_deg of true values
    - Peaks separated by at least separation_tol_deg
    
    Returns: bool (True if resolved)
    """
    if len(est_doas) != len(true_doas):
        return False
    
    est_sorted = np.sort(est_doas)
    true_sorted = np.sort(true_doas)
    
    # Check position accuracy: each estimate within ±1° of truth
    position_ok = all(abs(e - t) <= position_tol_deg for e, t in zip(est_sorted, true_sorted))
    
    # Check separation: peaks distinct by ≥0.5°
    if len(est_sorted) >= 2:
        separation_ok = (est_sorted[1] - est_sorted[0]) >= separation_tol_deg
    else:
        separation_ok = True
    
    return position_ok and separation_ok

def local_refine_peaks(X, pos_phys, d_phys, wavelength, K, coarse_peaks, 
                       alss_enabled=True, alss_mode='zero', alss_tau=1.0, alss_coreL=3,
                       refine_window_deg=2.0, refine_step_deg=0.01):
    """
    Perform local refinement around coarse peaks with finer grid.
    
    Parameters:
    -----------
    coarse_peaks : array
        Initial peak estimates from coarse scan
    refine_window_deg : float
        ±window around each peak to refine (default ±2°)
    refine_step_deg : float
        Fine grid step (default 0.01°)
    
    Returns:
    --------
    refined_peaks : array
        Refined DOA estimates
    """
    # Build local search ranges around each coarse peak
    local_ranges = []
    for peak in coarse_peaks:
        local_min = peak - refine_window_deg
        local_max = peak + refine_window_deg
        local_ranges.append((local_min, local_max, refine_step_deg))
    
    # Run refined scan for each peak
    refined_peaks = []
    for local_range in local_ranges:
        # Use existing coarray MUSIC with fine grid
        doas_local, info = estimate_doa_coarray_music(
            X, pos_phys, d_phys, wavelength, K=1,  # Find 1 peak in local region
            scan_deg=local_range,
            alss_enabled=alss_enabled,
            alss_mode=alss_mode,
            alss_tau=alss_tau,
            alss_coreL=alss_coreL
        )
        if len(doas_local) > 0:
            refined_peaks.append(doas_local[0])
    
    return np.array(sorted(refined_peaks))

def run_single_trial(array_type, N, d, wavelength, doas_true, snapshots, snr_db, seed,
                     alss_enabled=True, alss_mode='zero', alss_tau=1.0, alss_coreL=3,
                     coarse_step=0.05, refine_step=0.01, refine_window=2.0,
                     position_tol=1.0, separation_tol=0.5):
    """
    Run single trial with two-stage grid refinement.
    
    Returns:
    --------
    dict with keys: rmse_deg, resolved, condition_number, num_virtual_sensors
    """
    K = len(doas_true)
    
    # Get array geometry
    _, pos_phys, _ = get_array_geometry(array_type, N, d, lambda_factor=wavelength/d)
    
    # Simulate snapshots
    X = simulate_snapshots(pos_phys, d, wavelength, doas_true, snapshots, snr_db, seed)
    
    # Stage 1: Coarse global scan (0.05°)
    coarse_scan = (-60, 60, coarse_step)
    try:
        doas_coarse, info = estimate_doa_coarray_music(
            X, pos_phys, d, wavelength, K=K,
            scan_deg=coarse_scan,
            alss_enabled=alss_enabled,
            alss_mode=alss_mode,
            alss_tau=alss_tau,
            alss_coreL=alss_coreL
        )
        
        # Stage 2: Local refinement around peaks (0.01°)
        if len(doas_coarse) == K:
            doas_est = local_refine_peaks(
                X, pos_phys, d, wavelength, K, doas_coarse,
                alss_enabled=alss_enabled, alss_mode=alss_mode,
                alss_tau=alss_tau, alss_coreL=alss_coreL,
                refine_window_deg=refine_window, refine_step_deg=refine_step
            )
        else:
            doas_est = doas_coarse  # Fallback if wrong number of peaks
        
        # Compute metrics
        if len(doas_est) == K:
            errors = np.sort(doas_est) - np.sort(doas_true)
            rmse = np.sqrt(np.mean(errors**2))
            resolved = resolve_tolerance_check(doas_est, doas_true, position_tol, separation_tol)
        else:
            rmse = 90.0  # Penalty for wrong number of peaks
            resolved = False
        
        # Extract condition number if available
        cond_num = info.get('Rv_cond', None) if info else None
        num_virtual = info.get('Mv', None) if info else None
        
        return {
            'rmse_deg': rmse,
            'resolved': resolved,
            'condition_number': cond_num,
            'num_virtual_sensors': num_virtual,
            'num_peaks_found': len(doas_est)
        }
        
    except Exception as e:
        warnings.warn(f"Trial failed: {e}")
        return {
            'rmse_deg': 90.0,
            'resolved': False,
            'condition_number': None,
            'num_virtual_sensors': None,
            'num_peaks_found': 0
        }

def wilson_binomial_ci(successes, trials, alpha=0.05):
    """
    Wilson score interval for binomial proportion.
    More accurate than normal approximation for small n or extreme p.
    
    Returns: (p_est, ci_lower, ci_upper)
    """
    if trials == 0:
        return 0.0, 0.0, 0.0
    
    p = successes / trials
    z = stats.norm.ppf(1 - alpha/2)  # 1.96 for 95% CI
    
    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2*trials)) / denominator
    margin = z * np.sqrt((p*(1-p) + z**2/(4*trials)) / trials) / denominator
    
    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)
    
    return p, ci_lower, ci_upper

def bootstrap_rmse_ci(rmse_values, n_bootstrap=1000, alpha=0.05):
    """
    Bootstrap confidence interval for RMSE.
    
    Returns: (rmse_mean, ci_lower, ci_upper)
    """
    rmse_array = np.array(rmse_values)
    n = len(rmse_array)
    
    if n < 10:  # Too few samples for bootstrap
        rmse_mean = np.mean(rmse_array)
        std_err = np.std(rmse_array, ddof=1) / np.sqrt(n)
        margin = stats.norm.ppf(1 - alpha/2) * std_err
        return rmse_mean, rmse_mean - margin, rmse_mean + margin
    
    # Bootstrap resampling
    bootstrap_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        resample = rng.choice(rmse_array, size=n, replace=True)
        bootstrap_means.append(np.mean(resample))
    
    bootstrap_means = np.array(bootstrap_means)
    ci_lower = np.percentile(bootstrap_means, 100 * alpha/2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
    
    return np.mean(rmse_array), ci_lower, ci_upper

def run_benchmark_sweep(array_type='Z5', N=7, trials=400,
                       snr_range=np.arange(0, 21, 5),
                       delta_range=np.array([10, 13, 20, 30, 45]),
                       snapshots=100, d=0.5, wavelength=1.0,
                       alss_enabled=True, alss_mode='zero', alss_tau=1.0, alss_coreL=3,
                       coarse_step=0.05, refine_step=0.01,
                       position_tol=1.0, separation_tol=0.5,
                       output_file=None):
    """
    Run comprehensive benchmark sweep with paper-ready parameters.
    """
    results = []
    
    total_runs = len(snr_range) * len(delta_range)
    run_counter = 0
    
    print(f"\n{'='*80}")
    print(f"Paper Benchmark Configuration:")
    print(f"{'='*80}")
    print(f"  Array Type:        {array_type} (N={N})")
    print(f"  ALSS:              {'ON' if alss_enabled else 'OFF'} (mode={alss_mode}, τ={alss_tau}, ℓ₀={alss_coreL})")
    print(f"  Trials per point:  {trials}")
    print(f"  Snapshots:         {snapshots}")
    print(f"  Grid:              {coarse_step}° coarse → {refine_step}° refined")
    print(f"  Tolerances:        ±{position_tol}° position, ≥{separation_tol}° separation")
    print(f"  SNR range:         {snr_range[0]} to {snr_range[-1]} dB ({len(snr_range)} points)")
    print(f"  Delta range:       {delta_range[0]}° to {delta_range[-1]}° ({len(delta_range)} points)")
    print(f"  Total scenarios:   {total_runs} (≈{total_runs * trials:,} trials)")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    for delta in delta_range:
        for snr in snr_range:
            run_counter += 1
            scenario_start = time.time()
            
            # DOAs: symmetric around 0°
            doas_true = np.array([-delta/2, delta/2])
            
            print(f"[{run_counter}/{total_runs}] SNR={snr:2d}dB, Δ={delta:5.1f}° ... ", end='', flush=True)
            
            # Run trials
            trial_results = []
            for t in range(trials):
                seed = int(1000 * delta + 100 * snr + t)
                result = run_single_trial(
                    array_type, N, d, wavelength, doas_true, snapshots, snr, seed,
                    alss_enabled, alss_mode, alss_tau, alss_coreL,
                    coarse_step, refine_step, 2.0,  # refine_window
                    position_tol, separation_tol
                )
                trial_results.append(result)
            
            # Aggregate results
            rmse_values = [r['rmse_deg'] for r in trial_results]
            resolved_count = sum(r['resolved'] for r in trial_results)
            
            # Compute CIs
            rmse_mean, rmse_ci_low, rmse_ci_high = bootstrap_rmse_ci(rmse_values, n_bootstrap=1000)
            resolve_pct, resolve_ci_low, resolve_ci_high = wilson_binomial_ci(resolved_count, trials)
            
            # Get condition number (average from successful trials)
            cond_nums = [r['condition_number'] for r in trial_results if r['condition_number'] is not None]
            avg_cond = np.mean(cond_nums) if cond_nums else None
            
            # Get CRB (using virtual array size for coarray MUSIC)
            _, pos_phys, _ = get_array_geometry(array_type, N, d, lambda_factor=wavelength/d)
            # For coarray MUSIC, use number of virtual sensors (approximate)
            if trial_results and trial_results[0]['num_virtual_sensors']:
                Mv = trial_results[0]['num_virtual_sensors']
            else:
                Mv = len(pos_phys)  # Fallback to physical sensors
            snr_linear = 10.0**(snr/10.0)
            crb_var = crb_pair_worst_deg(Mv, d, wavelength, snr_linear, doas_true, snapshots)
            crb_deg = np.sqrt(crb_var)
            
            scenario_time = time.time() - scenario_start
            
            results.append({
                'Array': array_type,
                'N': N,
                'SNR_dB': snr,
                'Delta_deg': delta,
                'Trials': trials,
                'RMSE_mean': rmse_mean,
                'RMSE_CI_low': rmse_ci_low,
                'RMSE_CI_high': rmse_ci_high,
                'Resolve_pct': resolve_pct * 100,
                'Resolve_CI_low': resolve_ci_low * 100,
                'Resolve_CI_high': resolve_ci_high * 100,
                'CRB_deg': crb_deg,
                'RMSE_over_CRB': rmse_mean / crb_deg if crb_deg > 0 else None,
                'Avg_Condition': avg_cond,
                'Time_sec': scenario_time
            })
            
            print(f"RMSE={rmse_mean:.3f}° [{rmse_ci_low:.3f}, {rmse_ci_high:.3f}], "
                  f"Resolve={resolve_pct*100:.1f}% [{resolve_ci_low*100:.1f}, {resolve_ci_high*100:.1f}], "
                  f"κ={avg_cond:.1f} ({scenario_time:.1f}s)")
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"Benchmark Complete! Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*80}\n")
    
    # Save results
    df = pd.DataFrame(results)
    if output_file:
        df.to_csv(output_file, index=False, float_format='%.6f')
        print(f"Results saved to: {output_file}")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Paper-ready MIMO DOA benchmarks with two-stage refinement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Z5 primary sweep (400 trials × 25 scenarios = 10,000 trials total)
  python scripts/run_paper_benchmarks.py --array Z5 --trials 400
  
  # Z4 comparison (200 trials for faster run)
  python scripts/run_paper_benchmarks.py --array Z4 --trials 200 --output results/bench/z4_paper.csv
  
  # ULA baseline
  python scripts/run_paper_benchmarks.py --array ULA --trials 300 --output results/bench/ula_paper.csv
  
  # Quick test (50 trials)
  python scripts/run_paper_benchmarks.py --trials 50 --snr-max 10
        """)
    
    parser.add_argument('--array', type=str, default='Z5',
                       choices=['Z5', 'Z4', 'ULA'],
                       help='Array type (default: Z5)')
    parser.add_argument('--N', type=int, default=7,
                       help='Number of physical sensors (default: 7)')
    parser.add_argument('--trials', type=int, default=400,
                       help='Trials per (SNR, Delta) point (default: 400)')
    parser.add_argument('--snr-min', type=int, default=0,
                       help='Minimum SNR in dB (default: 0)')
    parser.add_argument('--snr-max', type=int, default=20,
                       help='Maximum SNR in dB (default: 20)')
    parser.add_argument('--snr-step', type=int, default=5,
                       help='SNR step in dB (default: 5)')
    parser.add_argument('--deltas', type=str, default='10,13,20,30,45',
                       help='Comma-separated delta values in degrees (default: 10,13,20,30,45)')
    parser.add_argument('--snapshots', type=int, default=100,
                       help='Snapshots per trial (default: 100)')
    parser.add_argument('--alss-mode', type=str, default='zero',
                       choices=['zero', 'ar1'],
                       help='ALSS mode (default: zero)')
    parser.add_argument('--alss-tau', type=float, default=1.0,
                       help='ALSS shrinkage intensity τ (default: 1.0)')
    parser.add_argument('--alss-coreL', type=int, default=3,
                       help='ALSS protected core ℓ₀ (default: 3)')
    parser.add_argument('--no-alss', action='store_true',
                       help='Disable ALSS (baseline)')
    parser.add_argument('--coarse-step', type=float, default=0.05,
                       help='Coarse grid step in degrees (default: 0.05)')
    parser.add_argument('--refine-step', type=float, default=0.01,
                       help='Refinement grid step in degrees (default: 0.01)')
    parser.add_argument('--position-tol', type=float, default=1.0,
                       help='Position tolerance for resolution (degrees, default: 1.0)')
    parser.add_argument('--separation-tol', type=float, default=0.5,
                       help='Minimum separation for resolution (degrees, default: 0.5)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file (default: results/bench/{array}_paper_N{N}_T{trials}.csv)')
    
    args = parser.parse_args()
    
    # Parse deltas
    deltas = np.array([float(d) for d in args.deltas.split(',')])
    
    # Generate SNR range
    snr_range = np.arange(args.snr_min, args.snr_max + 1, args.snr_step)
    
    # Default output filename
    if args.output is None:
        alss_tag = 'baseline' if args.no_alss else f'alss_{args.alss_mode}'
        output_file = (f"results/bench/{args.array.lower()}_paper_N{args.N}_T{args.trials}_{alss_tag}.csv")
    else:
        output_file = args.output
    
    # Create output directory
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    df = run_benchmark_sweep(
        array_type=args.array,
        N=args.N,
        trials=args.trials,
        snr_range=snr_range,
        delta_range=deltas,
        snapshots=args.snapshots,
        d=0.5,  # Half-meter physical spacing
        wavelength=1.0,  # 1 meter wavelength → d = λ/2
        alss_enabled=not args.no_alss,
        alss_mode=args.alss_mode,
        alss_tau=args.alss_tau,
        alss_coreL=args.alss_coreL,
        coarse_step=args.coarse_step,
        refine_step=args.refine_step,
        position_tol=args.position_tol,
        separation_tol=args.separation_tol,
        output_file=output_file
    )
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    for delta in deltas:
        subset = df[df['Delta_deg'] == delta]
        print(f"\nΔ = {delta}°:")
        print(subset[['SNR_dB', 'RMSE_mean', 'Resolve_pct', 'RMSE_over_CRB']].to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"Complete results saved to: {output_file}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
