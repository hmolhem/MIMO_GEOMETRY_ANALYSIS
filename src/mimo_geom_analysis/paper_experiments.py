"""
Package-level runner for paper experiments (pilot refactor of
`core/analysis_scripts/run_paper_experiments.py`).

Provides `run_paper_experiments(argv=None)` for programmatic use and testing.
"""
from pathlib import Path
import sys
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import time
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import t as t_dist

# Import core functionality from core package
from core.radarpy.signal.doa_sim_core import simulate_snapshots
from core.radarpy.algorithms.coarray_music import estimate_doa_coarray_music
from core.radarpy.algorithms.crb import crb_pair_worst_deg
from core.radarpy.signal.mutual_coupling import generate_mcm


def get_array_positions(array_type: str, N: int = 7) -> np.ndarray:
    array_type = array_type.upper()
    if array_type == 'ULA':
        return np.arange(N, dtype=float)
    elif array_type == 'NESTED':
        N1 = (N + 1) // 2
        N2 = N // 2
        inner = np.arange(1, N1 + 1, dtype=float)
        outer = (N1 + 1) * np.arange(1, N2 + 1, dtype=float)
        return np.sort(np.concatenate([inner, outer]))
    elif array_type == 'Z1':
        if N == 10:
            return np.array([0, 1, 4, 9, 11, 13, 17, 25, 27, 34], dtype=float)
        elif N == 3:
            return np.array([0, 1, 4], dtype=float)
        else:
            return np.array([0, 1, 4] + list(range(9, 9 + N - 3)), dtype=float)
    elif array_type == 'Z4':
        if N == 7:
            return np.array([0, 5, 8, 11, 14, 17, 21], dtype=float)
        else:
            return np.array([0] + list(range(5, 5 + 3 * (N - 1), 3)), dtype=float)
    elif array_type == 'Z5':
        if N == 7:
            return np.array([0, 5, 8, 11, 14, 17, 21], dtype=float)
        else:
            positions = [0, 5, 8]
            for i in range(3, N):
                positions.append(positions[-1] + 3)
            return np.array(positions, dtype=float)
    elif array_type == 'Z6':
        if N == 7:
            return np.array([0, 1, 4, 8, 13, 17, 22], dtype=float)
        else:
            positions = [0, 1, 4]
            for i in range(3, N):
                positions.append(positions[-1] + 4 + (i - 3))
            return np.array(positions, dtype=float)
    else:
        raise ValueError(f"Unknown array type: {array_type}")


def compute_rmse(estimated: np.ndarray, true: np.ndarray) -> float:
    if len(estimated) != len(true):
        return np.inf
    cost_matrix = np.abs(true[:, np.newaxis] - estimated[np.newaxis, :])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    errors = true[row_ind] - estimated[col_ind]
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_resolution_rate(trials_results: List[Tuple[np.ndarray, float]],
                            true_doas: np.ndarray,
                            threshold: float = 3.0) -> float:
    K = len(true_doas)
    successes = 0
    total = len(trials_results)
    for est_doas, _ in trials_results:
        if len(est_doas) == K:
            rmse = compute_rmse(est_doas, true_doas)
            if rmse < threshold:
                successes += 1
    return 100.0 * successes / total if total > 0 else 0.0


def compute_crb_ratio(rmse: float, positions: np.ndarray, wavelength: float,
                     true_doas: np.ndarray, snr_db: float, snapshots: int) -> float:
    try:
        crb_deg = crb_pair_worst_deg(
            positions=positions,
            wavelength=wavelength,
            doas_deg=true_doas,
            snr_db=snr_db,
            M=snapshots,
            coupling_matrix=None
        )
        return float(rmse / crb_deg) if crb_deg > 0 else float('inf')
    except Exception:
        return float('inf')


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95):
    n = len(values)
    if n < 2:
        return float(np.mean(values)), float('nan'), float('nan')
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    sem = std / np.sqrt(n)
    alpha = 1 - confidence
    t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)
    ci_low = mean - t_crit * sem
    ci_high = mean + t_crit * sem
    return mean, ci_low, ci_high


def run_doa_trial(positions: np.ndarray,
                  true_doas: np.ndarray,
                  wavelength: float,
                  snr_db: float,
                  snapshots: int,
                  coupling_matrix: Optional[np.ndarray] = None,
                  alss_enabled: bool = False,
                  seed: Optional[int] = None):
    d_phys = 0.5
    positions_meters = positions * d_phys
    X, _, _ = simulate_snapshots(
        sensor_positions=positions_meters,
        wavelength=wavelength,
        doas_deg=true_doas,
        snr_db=snr_db,
        snapshots=snapshots,
        seed=seed,
        coupling_matrix=coupling_matrix
    )
    K = len(true_doas)
    start = time.time()
    doas_est, _ = estimate_doa_coarray_music(
        X=X,
        positions=positions,
        d_phys=d_phys,
        wavelength=wavelength,
        K=K,
        scan_deg=(-60, 60, 0.1),
        alss_enabled=alss_enabled,
        alss_tau=1.0 if alss_enabled else 0.0,
        alss_mode='zero'
    )
    runtime_ms = (time.time() - start) * 1000
    return doas_est, float(runtime_ms)


# For brevity, only implement scenario1 here (pilot). Other scenarios can be
# ported later following the same pattern.
def run_scenario1_baseline(arrays: List[str], snr_sweep: List[float], snapshot_sweep: List[int], trials: int, output_dir: str) -> pd.DataFrame:
    wavelength = 1.0
    true_doas = np.array([15.0, -20.0])
    N = 7
    results = []
    for array_type in arrays:
        positions = get_array_positions(array_type, N)
        for snr in snr_sweep:
            rmse_list = []
            runtime_list = []
            trials_data = []
            for trial in range(trials):
                seed = 42 + trial
                est_doas, runtime = run_doa_trial(positions, true_doas, wavelength, snr, 64, coupling_matrix=None, alss_enabled=False, seed=seed)
                rmse_list.append(compute_rmse(est_doas, true_doas))
                runtime_list.append(runtime)
                trials_data.append((est_doas, runtime))
            rmse_mean, rmse_ci_low, rmse_ci_high = compute_confidence_interval(np.array(rmse_list))
            crb_ratio = compute_crb_ratio(rmse_mean, positions, wavelength, true_doas, snr, 64)
            resolution_rate = compute_resolution_rate(trials_data, true_doas)
            runtime_mean = float(np.mean(runtime_list)) if runtime_list else 0.0
            results.append({
                'Scenario': 'Baseline',
                'Array': array_type,
                'SNR_dB': snr,
                'Snapshots': 64,
                'RMSE_deg': rmse_mean,
                'RMSE_CI_Low': rmse_ci_low,
                'RMSE_CI_High': rmse_ci_high,
                'RMSE_CRB_Ratio': crb_ratio,
                'Resolution_Rate_%': resolution_rate,
                'Runtime_ms': runtime_mean,
                'Trials': trials
            })
    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'scenario1_baseline.csv')
    results_df.to_csv(csv_path, index=False)
    return results_df


def run_paper_experiments(argv: Optional[List[str]] = None) -> str:
    ap = argparse.ArgumentParser(description='ALSS Paper Experiments - Minimal Metrics Set')
    ap.add_argument('--scenario', type=str, choices=['1', '3', '4', 'all'], required=True)
    ap.add_argument('--trials', type=int, default=500)
    ap.add_argument('--arrays', nargs='+', default=['ULA', 'Nested', 'Z1', 'Z4', 'Z5', 'Z6'])
    ap.add_argument('--output-dir', type=str, default='results/paper_experiments')
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args(args=argv)
    if args.test:
        args.trials = 2
        snr_sweep = [0, 5]
        snapshot_sweep = [32, 64]
    else:
        snr_sweep = [-5, 0, 5, 10, 15]
        snapshot_sweep = [32, 64, 128]
    if args.scenario == '1' or args.scenario == 'all':
        run_scenario1_baseline(arrays=args.arrays, snr_sweep=snr_sweep, snapshot_sweep=snapshot_sweep, trials=args.trials, output_dir=args.output_dir)
    return args.output_dir


if __name__ == '__main__':
    run_paper_experiments()
