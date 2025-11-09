# analysis_scripts/run_benchmarks.py
import argparse, os, time, json, csv
from pathlib import Path
import numpy as np
import pandas as pd

from sim.doa_sim_core import run_music, music_spectrum
from sim.array_manifold import steering_vector
from algorithms.spatial_music import estimate_doa_spatial_music
from algorithms.coarray_music import estimate_doa_coarray_music
from util.crb import crb_pair_worst_deg

# --- CRB Interpretation Helper ---
def crb_ratio_safestr(rmse_deg: float, crb_deg: float) -> str:
    """Format RMSE/CRB ratio with correct interpretation (RMSE is always >= CRB)."""
    if crb_deg is None or crb_deg <= 0:
        return "CRB N/A"
    ratio = rmse_deg / crb_deg
    return f"{ratio:.2f}× above CRB"

# --- Snapshot simulator (common to SpatialMUSIC & CoarrayMUSIC) ---
def _steering_matrix(positions, d_phys, wavelength, doas_deg):
    """A(θ): shape (N x K) for integer-grid positions * d_phys."""
    pos = np.asarray(positions, dtype=float) * float(d_phys)
    k = 2.0 * np.pi / float(wavelength)
    th = np.deg2rad(np.asarray(doas_deg, dtype=float))
    # a_n(θ) = exp(j k x_n sin θ)
    A = np.exp(1j * k * np.outer(pos, np.sin(th)))
    return A

def simulate_snapshots(positions, d_phys, wavelength, doas_deg, snapshots, snr_db, seed=None):
    """
    Returns X (N x M): X = A S + N, with unit source power and noise set by SNR per sensor.
    SNR definition: per-sensor, per-snapshot average (linear).
    """
    if seed is not None:
        rs = np.random.RandomState(seed)
    else:
        rs = np.random

    N = len(positions)
    K = len(doas_deg)
    M = int(snapshots)

    A = _steering_matrix(positions, d_phys, wavelength, doas_deg)  # (N x K)
    # Unit source power; complex Gaussian i.i.d.
    S = (rs.randn(K, M) + 1j * rs.randn(K, M)) / np.sqrt(2.0)

    # Signal power per sensor (average over sources + snapshots)
    sig = A @ S
    Ps = np.mean(np.abs(sig)**2)

    snr_lin = 10.0**(snr_db/10.0)
    # Noise power so that Ps / Pn = snr_lin
    Pn = Ps / max(snr_lin, 1e-12)
    Nn = (rs.randn(N, M) + 1j * rs.randn(N, M)) / np.sqrt(2.0) * np.sqrt(Pn)

    X = sig + Nn
    return X

# Adapters to pull sensor positions from your processors
# Assumes: geometry_processors.<name> exposes classes with 'sensor_positions' or a method
def get_array_geometry(name, N, d, lambda_factor=2.0):
    """
    Returns (array_name, positions_in_meters, wavelength_m)
    
    Parameters:
    -----------
    name : str
        Array type (ULA, Z4, Z5, Z6)
    N : int
        Number of sensors
    d : float
        Physical spacing in meters
    lambda_factor : float
        Wavelength = lambda_factor * d (default 2.0 for λ/2 spacing to avoid aliasing)
        
    Notes:
    ------
    - lambda_factor=2.0 gives d = λ/2 (standard half-wavelength spacing, no aliasing)
    - lambda_factor=1.0 gives d = λ (aliased for ULA, but sparse arrays more robust)
    """
    name = name.upper()
    wavelength = lambda_factor * d  # avoid spatial aliasing with λ/2 spacing by default

    if name == "ULA":
        pos = d * np.arange(N)
        arr_name = f"ULA(N={N})"
    elif name == "Z4":
        from geometry_processors.z4_processor import Z4ArrayProcessor
        z = Z4ArrayProcessor(N=N, d=d)
        z.run_full_analysis(verbose=False)  # ensure coarray computed
        pos = np.asarray(z.data.sensors_positions, dtype=float) * d
        arr_name = f"Z4(N={N})"
    elif name == "Z5":
        from geometry_processors.z5_processor import Z5ArrayProcessor
        z = Z5ArrayProcessor(N=N, d=d)
        z.run_full_analysis(verbose=False)  # ensure coarray computed
        pos = np.asarray(z.data.sensors_positions, dtype=float) * d
        arr_name = f"Z5(N={N})"
    elif name == "Z6":
        from geometry_processors.z6_processor import Z6ArrayProcessor
        z = Z6ArrayProcessor(N=N, d=d)
        z.run_full_analysis(verbose=False)  # ensure coarray computed
        pos = np.asarray(z.data.sensors_positions, dtype=float) * d
        arr_name = f"Z6(N={N})"
    else:
        raise ValueError(f"Unknown array name: {name}")
    
    # Aliasing check: warn if any adjacent spacing > λ/2
    dx = np.diff(np.sort(pos))
    max_dx = dx.max() if dx.size > 0 else 0
    if max_dx > wavelength / 2 + 1e-9:
        kd_max = 2 * np.pi * max_dx / wavelength
        print(f"[WARN] {arr_name}: Spatial aliasing possible! max(Δx) = {max_dx:.3f}m > λ/2 = {wavelength/2:.3f}m (kd = {kd_max:.2f})")
    
    # Report kd for ULA (for diagnostics)
    if name == "ULA" and N > 1:
        kd = 2 * np.pi * d / wavelength
        print(f"[INFO] {arr_name}: d = {d:.3f}m, λ = {wavelength:.3f}m, kd = {kd:.3f} rad (π = aliasing threshold)")
    
    return arr_name, pos, wavelength

def parse_list(s, cast=float):
    return [cast(x) for x in s.split(",") if x.strip()]

def run_alg(alg_name, X, positions, d_phys, wavelength, K, scan_deg=(-60, 60, 0.1), use_root=False):
    """
    Unified algorithm dispatcher for DOA estimation.
    
    Parameters:
    -----------
    alg_name : str
        Algorithm name: 'SpatialMUSIC' or 'CoarrayMUSIC'
    X : ndarray (N, M)
        Received signal matrix
    positions : ndarray
        Integer-grid sensor indices
    d_phys : float
        Physical grid spacing in meters
    wavelength : float
        Signal wavelength in meters
    K : int
        Number of sources to estimate
    scan_deg : tuple
        (theta_min, theta_max, theta_step) for angular scan
    use_root : bool
        If True and alg_name=='CoarrayMUSIC', use Root-MUSIC instead of grid search
        
    Returns:
    --------
    tuple: (doas_est, dbg) where dbg is dict with Mv and other info
    """
    if alg_name == "SpatialMUSIC":
        # Keep existing spatial MUSIC call
        doas_est = estimate_doa_spatial_music(X, positions, d_phys, wavelength, K, scan_deg=scan_deg)
        dbg = {"Mv": len(positions)}  # physical array size
        return doas_est, dbg
    elif alg_name == "CoarrayMUSIC":
        # Ask for debug to get virtual size Lv
        doas_est, P, thetas, dbg_full = estimate_doa_coarray_music(
            X, positions, d_phys, wavelength, K, scan_deg=scan_deg, return_debug=True, use_root=use_root
        )
        # dbg_full["Lv"] present from coarray builder; expose Mv for CSV
        dbg = {"Mv": int(dbg_full.get("Lv", 1))}
        return doas_est, dbg
    else:
        raise ValueError(f"Unknown algorithm: {alg_name}")

def main():
    ap = argparse.ArgumentParser(description="Run DOA benchmarks (MUSIC baseline).")
    ap.add_argument("--arrays", type=str, default="Z4,Z5,Z6,ULA")
    ap.add_argument("--N", type=int, default=7)
    ap.add_argument("--d", type=float, default=1.0)
    ap.add_argument("--lambda_factor", type=float, default=2.0,
                    help="Wavelength = lambda_factor * d (default 2.0 for λ/2 spacing, use 1.0 to test aliasing)")
    ap.add_argument("--algs", type=str, default="SpatialMUSIC",
                    help="Comma-separated algorithms: SpatialMUSIC,CoarrayMUSIC")
    ap.add_argument("--snr", type=str, default="-5,0,5,10,15")
    ap.add_argument("--snapshots", type=str, default="32,64,128")
    ap.add_argument("--k", type=str, default="2")
    ap.add_argument("--delta", type=str, default="2")  # spacing (deg) for 2-source tests
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", type=str, default="results/bench/bench_default.csv")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--angle_span", type=str, default="-60,60")
    ap.add_argument("--resolve_thresh", type=float, default=1.0)
    ap.add_argument("--save-crb", action="store_true",
                    help="Export CRB overlay rows to results/bench/crb_overlay.csv")
    ap.add_argument("--use-root", action="store_true",
                    help="Use Root-MUSIC for CoarrayMUSIC (polynomial-based, faster+sharper)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    arrays = [x.strip() for x in args.arrays.split(",")]
    algs = [x.strip() for x in args.algs.split(",") if x.strip()]
    snrs = parse_list(args.snr, float)
    snaps = parse_list(args.snapshots, int)
    ks = parse_list(args.k, int)
    deltas = parse_list(args.delta, float)
    span = parse_list(args.angle_span, float)
    th_lo, th_hi = span[0], span[1]

    rng = np.random.default_rng(args.seed)
    rows = []
    crb_rows = []

    for arr in arrays:
        arr_name, pos, wavelength = get_array_geometry(arr, args.N, args.d, args.lambda_factor)
        
        # Compute geometry/aliasing metrics
        k = 2.0 * np.pi / wavelength
        spacings = np.diff(np.sort(pos))
        max_dx = float(np.max(spacings)) if spacings.size else 0.0
        kd_ula = float(k * args.d)  # meaningful for ULA
        kd_max = float(k * max_dx)  # max inter-sensor spacing
        lambda2 = wavelength / 2.0
        maxdx_over_lambda2 = float(max_dx / lambda2) if lambda2 > 0 else np.inf
        
        # Get L and K_max from processor if available
        L, K_max = 0, 0
        try:
            if arr.upper() in ["Z4", "Z5", "Z6"]:
                # These were already computed in get_array_geometry
                from geometry_processors import z4_processor, z5_processor, z6_processor
                proc_map = {"Z4": z4_processor.Z4ArrayProcessor,
                           "Z5": z5_processor.Z5ArrayProcessor,
                           "Z6": z6_processor.Z6ArrayProcessor}
                if arr.upper() in proc_map:
                    proc = proc_map[arr.upper()](N=args.N, d=args.d)
                    proc.run_full_analysis(verbose=False)
                    summary = proc.data.performance_summary_table if hasattr(proc.data, 'performance_summary_table') else None
                    if summary is not None:
                        L = int(summary[summary['Metrics'] == 'Contiguous Segment Length (L, one-sided)']['Value'].iloc[0] if not summary[summary['Metrics'] == 'Contiguous Segment Length (L, one-sided)'].empty else 0)
                        K_max = int(summary[summary['Metrics'] == 'Maximum Detectable Sources (K_max)']['Value'].iloc[0] if not summary[summary['Metrics'] == 'Maximum Detectable Sources (K_max)'].empty else 0)
        except:
            pass
        
        # Get integer-grid positions for simulate_snapshots
        # pos is already in meters from get_array_geometry; convert back to grid indices
        sensor_positions_grid = np.asarray(pos / args.d, dtype=int)
        
        for K in ks:
            # For K=2, use symmetric pair around 0 with Δθ; for K>2, random uniform in span.
            for SNR in snrs:
                for M in snaps:
                    for t in range(args.trials):
                        if K == 2:
                            dth = deltas[t % len(deltas)]
                            doas_true = np.array([-dth/2.0, +dth/2.0])
                        else:
                            doas_true = rng.uniform(th_lo+2, th_hi-2, size=K)
                            doas_true.sort()

                        # Generate a fresh snapshot matrix once per trial/array condition
                        trial_seed = rng.integers(0, 1<<31)
                        t0_total = time.perf_counter()
                        X = simulate_snapshots(sensor_positions_grid, args.d, wavelength, doas_true, M, SNR, seed=trial_seed)
                        
                        # Run each algorithm on the same data
                        from sim.metrics import angle_rmse_deg, resolved_indicator
                        
                        for alg in algs:
                            try:
                                t0 = time.perf_counter()
                                use_root = args.use_root if hasattr(args, 'use_root') else False
                                doas_est, dbg = run_alg(alg, X, sensor_positions_grid, args.d, wavelength, K, scan_deg=(-60, 60, 0.1), use_root=use_root)
                                t1 = time.perf_counter()
                                
                                runtime_ms = 1000.0 * (t1 - t0)
                                runtime_total_ms = 1000.0 * (time.perf_counter() - t0_total)
                                
                                rmse, errs = angle_rmse_deg(doas_true, doas_est)
                                resolved = resolved_indicator(doas_true, doas_est, threshold_deg=args.resolve_thresh)
                                
                                Mv = int(dbg.get("Mv", len(pos)))

                                rows.append({
                                    "array": arr_name,
                                    "N": args.N,
                                    "d": args.d,
                                    "wavelength": wavelength,
                                    "lambda_factor": args.lambda_factor,
                                    "kd": kd_ula,
                                    "kd_max": kd_max,
                                    "max_dx_over_lambda2": maxdx_over_lambda2,
                                    "L": L,
                                    "K_max": K_max,
                                    "Mv": Mv,
                                    "alg": alg,
                                    "SNR_dB": SNR,
                                    "snapshots": M,
                                    "K": K,
                                    "delta_deg": dth if K == 2 else np.nan,
                                    "doas_true": ";".join(f"{x:.3f}" for x in doas_true),
                                    "doas_est":  ";".join(f"{x:.3f}" for x in doas_est),
                                    "rmse_deg": rmse,
                                    "resolved": int(resolved),
                                    "runtime_ms": runtime_ms,
                                    "runtime_total_ms": runtime_total_ms,
                                })
                            except Exception as e:
                                print(f"[ERROR] {alg} failed for {arr_name}: {e}")
                                import traceback
                                traceback.print_exc()

                        # light flush per ~1000 rows
                        if len(rows) % 1000 == 0:
                            df = pd.DataFrame(rows)
                            df.to_csv(args.out, index=False)
                    
                    # (Optional) CRB overlay file - compute once per (array, alg, SNR, M, K, delta) condition
                    if args.save_crb and K == 2:
                        # Only compute for 2-source case with known delta
                        for alg in algs:
                            snr_lin = 10.0**(SNR/10.0)
                            theta_pair = (doas_true[0], doas_true[1])
                            
                            # Use virtual size for CoarrayMUSIC; physical N for SpatialMUSIC
                            if alg == "CoarrayMUSIC":
                                Mv_for_alg = L + 1 if L > 0 else len(pos)
                            else:
                                Mv_for_alg = len(pos)
                            
                            try:
                                crb_deg = np.sqrt(crb_pair_worst_deg(Mv_for_alg, args.d, wavelength, snr_lin, theta_pair, M))
                                
                                crb_rows.append({
                                    "array": arr_name, "alg": alg,
                                    "N": args.N, "Mv": Mv_for_alg, "d": args.d, "wavelength": wavelength,
                                    "SNR_dB": SNR, "snapshots": M, "delta_deg": dth,
                                    "crb_deg": float(crb_deg)
                                })
                            except Exception as e:
                                print(f"[WARN] CRB computation failed for {arr_name}, {alg}: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[Saved] Bench CSV → {args.out}")
    
    # Save CRB overlay if requested
    if args.save_crb and crb_rows:
        crb_path = Path("results/bench/crb_overlay.csv")
        crb_path.parent.mkdir(parents=True, exist_ok=True)
        with crb_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(crb_rows[0].keys()))
            w.writeheader()
            w.writerows(crb_rows)
        print(f"[Saved] CRB overlay → {crb_path}")

if __name__ == "__main__":
    main()
