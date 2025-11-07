"""
DOA Estimation Demo with MIMO Arrays
=====================================

Demonstrates Direction of Arrival (DOA) estimation using MUSIC algorithm
with different MIMO array geometries.

Usage:
    python run_doa_demo.py --array z5 --N 7 --K 3 --angles -30 10 45 --SNR 10
    python run_doa_demo.py --array nested --N1 2 --N2 3 --K 2 --SNR 15
    python run_doa_demo.py --array ula --M 8 --K 3 --compare-snr
"""

import sys
import os
import argparse
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from doa_estimation.music import MUSICEstimator
from doa_estimation.metrics import DOAMetrics
from doa_estimation.visualization import (
    plot_doa_spectrum, plot_array_geometry, 
    plot_estimation_results, plot_music_comparison
)

# Import array processors
from geometry_processors.ula_processors import ULArrayProcessor
from geometry_processors.nested_processor import NestedArrayProcessor
from geometry_processors.z1_processor import Z1ArrayProcessor
from geometry_processors.z3_1_processor import Z3_1ArrayProcessor
from geometry_processors.z3_2_processor import Z3_2ArrayProcessor
from geometry_processors.z4_processor import Z4ArrayProcessor
from geometry_processors.z5_processor import Z5ArrayProcessor
from geometry_processors.z6_processor import Z6ArrayProcessor
from geometry_processors.tca_processor import TCAArrayProcessor
from geometry_processors.epca_processor import ePCAArrayProcessor


def create_array_processor(args):
    """Create array processor based on arguments."""
    array_type = args.array.lower()
    
    if array_type == 'ula':
        N = args.N if hasattr(args, 'N') and args.N else 8
        return ULArrayProcessor(N=N, d=args.d)
    
    elif array_type == 'nested':
        N1 = args.N1 if hasattr(args, 'N1') and args.N1 else 2
        N2 = args.N2 if hasattr(args, 'N2') and args.N2 else 3
        return NestedArrayProcessor(N1=N1, N2=N2, d=args.d)
    
    elif array_type == 'tca':
        M = args.M if hasattr(args, 'M') and args.M else 3
        N = args.N if hasattr(args, 'N') and args.N else 4
        return TCAArrayProcessor(M=M, N=N, d=args.d)
    
    elif array_type == 'epca':
        p1 = args.p1 if hasattr(args, 'p1') and args.p1 else 2
        p2 = args.p2 if hasattr(args, 'p2') and args.p2 else 3
        p3 = args.p3 if hasattr(args, 'p3') and args.p3 else 5
        return ePCAArrayProcessor(p1=p1, p2=p2, p3=p3, d=args.d)
    
    elif array_type == 'z1':
        N = args.N if hasattr(args, 'N') and args.N else 7
        return Z1ArrayProcessor(N=N, d=args.d)
    
    elif array_type == 'z3_1':
        N = args.N if hasattr(args, 'N') and args.N else 6
        return Z3_1ArrayProcessor(N=N, d=args.d)
    
    elif array_type == 'z3_2':
        N = args.N if hasattr(args, 'N') and args.N else 6
        return Z3_2ArrayProcessor(N=N, d=args.d)
    
    elif array_type == 'z4':
        N = args.N if hasattr(args, 'N') and args.N else 7
        return Z4ArrayProcessor(N=N, d=args.d)
    
    elif array_type == 'z5':
        N = args.N if hasattr(args, 'N') and args.N else 7
        return Z5ArrayProcessor(N=N, d=args.d)
    
    elif array_type == 'z6':
        N = args.N if hasattr(args, 'N') and args.N else 7
        return Z6ArrayProcessor(N=N, d=args.d)
    
    else:
        raise ValueError(f"Unknown array type: {array_type}")


def run_single_estimation(args):
    """Run single DOA estimation experiment."""
    print("\n" + "="*70)
    print("DOA ESTIMATION WITH MUSIC ALGORITHM")
    print("="*70)
    
    # Create array
    processor = create_array_processor(args)
    array_data = processor.run_full_analysis()
    
    print(f"\nArray: {array_data.name}")
    print(f"Physical Sensors: {array_data.num_sensors}")
    print(f"Sensor Positions: {array_data.sensors_positions}")
    
    # Get K_max from performance table
    perf = array_data.performance_summary_table
    # The table has columns 'Metrics' and 'Value'
    k_row = perf[perf['Metrics'].str.contains('K_max', case=False, na=False)]
    if not k_row.empty:
        K_max = int(k_row['Value'].iloc[0])
    else:
        K_max = array_data.num_sensors - 1  # Fallback
    print(f"\nMaximum Detectable Sources (K_max): {K_max}")
    
    # Check if K_sources is valid
    K_sources = args.K
    if K_sources > K_max:
        print(f"\n⚠ WARNING: Requested K={K_sources} exceeds K_max={K_max}")
        print(f"   Setting K={K_max}")
        K_sources = K_max
    
    # True angles
    if args.angles:
        true_angles = args.angles
    else:
        # Generate random angles
        np.random.seed(42)
        true_angles = sorted(np.random.uniform(-60, 60, K_sources))
    
    print(f"\nTrue DOA Angles: {[f'{a:.1f}°' for a in true_angles]}")
    
    # Create MUSIC estimator
    estimator = MUSICEstimator(
        sensor_positions=array_data.sensors_positions,
        wavelength=args.wavelength,
        angle_range=(-90, 90),
        angle_resolution=args.resolution
    )
    
    # Simulate signals
    print(f"\nSimulating signals (SNR={args.SNR} dB, {args.snapshots} snapshots)...")
    X = estimator.simulate_signals(
        true_angles=true_angles,
        SNR_dB=args.SNR,
        snapshots=args.snapshots,
        signal_type='random'
    )
    
    # Estimate DOA
    print("Running MUSIC algorithm...")
    estimated_angles, spectrum = estimator.estimate(
        X, K_sources=K_sources, return_spectrum=True
    )
    
    print(f"\nEstimated DOA Angles: {[f'{a:.1f}°' for a in estimated_angles]}")
    
    # Compute metrics
    metrics = DOAMetrics()
    rmse = metrics.compute_rmse(true_angles, estimated_angles)
    bias = metrics.compute_bias(true_angles, estimated_angles)
    mae = metrics.compute_mae(true_angles, estimated_angles)
    
    print(f"\nPerformance Metrics:")
    print(f"  RMSE:  {rmse:.3f}°")
    print(f"  MAE:   {mae:.3f}°")
    print(f"  Bias:  {bias:.3f}°")
    
    # Plot results
    if args.plot:
        print("\nGenerating plots...")
        
        # Array geometry
        plot_array_geometry(
            array_data.sensors_positions,
            array_name=array_data.name,
            wavelength=args.wavelength
        )
        
        # MUSIC spectrum
        plot_doa_spectrum(
            estimator.angles,
            spectrum,
            true_angles=true_angles,
            estimated_angles=estimated_angles,
            title=f"{array_data.name} - MUSIC DOA Estimation (SNR={args.SNR}dB)"
        )
        
        # Estimation results
        plot_estimation_results(
            true_angles,
            estimated_angles,
            rmse=rmse,
            title=f"{array_data.name} DOA Estimation"
        )
    
    print("\n" + "="*70)
    print("ESTIMATION COMPLETE")
    print("="*70 + "\n")


def run_snr_comparison(args):
    """Compare DOA performance across SNR values."""
    print("\n" + "="*70)
    print("DOA PERFORMANCE vs SNR")
    print("="*70)
    
    # Create array
    processor = create_array_processor(args)
    array_data = processor.run_full_analysis()
    
    print(f"\nArray: {array_data.name}")
    print(f"Physical Sensors: {array_data.num_sensors}")
    
    # Get K_max from performance table
    perf = array_data.performance_summary_table
    k_row = perf[perf['Metrics'].str.contains('K_max', case=False, na=False)]
    K_max = int(k_row['Value'].iloc[0]) if not k_row.empty else array_data.num_sensors - 1
    
    K_sources = min(args.K, K_max)
    
    # Generate true angles
    np.random.seed(42)
    true_angles = sorted(np.random.uniform(-60, 60, K_sources))
    print(f"True Angles: {[f'{a:.1f}°' for a in true_angles]}")
    
    # SNR range
    snr_values = np.arange(0, 21, 5)
    rmse_values = []
    
    # Create estimator
    estimator = MUSICEstimator(
        sensor_positions=array_data.sensors_positions,
        wavelength=args.wavelength,
        angle_resolution=args.resolution
    )
    
    print(f"\nRunning {len(snr_values)} experiments...")
    
    for snr in snr_values:
        print(f"  SNR = {snr} dB...", end=" ")
        
        # Monte Carlo trials
        trials = 20
        trial_rmse = []
        
        for trial in range(trials):
            X = estimator.simulate_signals(
                true_angles, SNR_dB=snr, 
                snapshots=args.snapshots
            )
            
            est_angles = estimator.estimate(X, K_sources=K_sources)
            
            metrics = DOAMetrics()
            rmse = metrics.compute_rmse(true_angles, est_angles)
            trial_rmse.append(rmse)
        
        avg_rmse = np.mean(trial_rmse)
        rmse_values.append(avg_rmse)
        print(f"RMSE = {avg_rmse:.3f}°")
    
    # Plot comparison
    results = {
        array_data.name: {
            'x_values': snr_values.tolist(),
            'rmse': rmse_values
        }
    }
    
    plot_music_comparison(
        results,
        metric='rmse',
        xlabel='SNR (dB)',
        save_path=f'results/plots/doa_snr_comparison_{args.array}.png'
    )
    
    print("\n" + "="*70)


def run_array_comparison(args):
    """Compare different array types."""
    print("\n" + "="*70)
    print("ARRAY COMPARISON FOR DOA ESTIMATION")
    print("="*70)
    
    # Arrays to compare
    array_configs = [
        ('ula', {'M': 6, 'd': args.d}),
        ('nested', {'N1': 2, 'N2': 3, 'd': args.d}),
        ('z5', {'N': 7, 'd': args.d}),
        ('z6', {'N': 7, 'd': args.d}),
    ]
    
    # True angles
    np.random.seed(42)
    true_angles = sorted(np.random.uniform(-60, 60, 3))
    
    print(f"\nTrue Angles: {[f'{a:.1f}°' for a in true_angles]}")
    print(f"SNR: {args.SNR} dB")
    print(f"Snapshots: {args.snapshots}")
    
    results = {}
    
    for array_name, config in array_configs:
        print(f"\n{array_name.upper()}:")
        
        # Create processor
        args_temp = argparse.Namespace(**config, array=array_name)
        processor = create_array_processor(args_temp)
        array_data = processor.run_full_analysis()
        
        # Get K_max - handle different data structures
        k_max_val = None
        if hasattr(array_data, 'performance_summary_table') and array_data.performance_summary_table is not None:
            perf = array_data.performance_summary_table
            k_row = perf[perf['Metrics'].str.contains('K_max', case=False, na=False)]
            k_max_val = int(k_row['Value'].iloc[0]) if not k_row.empty else None
        
        if k_max_val is None:
            # Fallback for processors that don't have standard table
            if hasattr(processor, 'generate_performance_summary'):
                summary = processor.generate_performance_summary()
                k_max_val = summary.get('Maximum Detectable Sources (K_max)', array_data.num_sensors - 1)
            else:
                k_max_val = array_data.num_sensors - 1
        
        print(f"  Sensors: {array_data.num_sensors}")
        print(f"  K_max: {k_max_val}")
        
        # Estimate
        estimator = MUSICEstimator(
            array_data.sensors_positions,
            wavelength=args.wavelength
        )
        
        X = estimator.simulate_signals(
            true_angles, SNR_dB=args.SNR, snapshots=args.snapshots
        )
        
        est_angles = estimator.estimate(X, K_sources=3)
        
        metrics = DOAMetrics()
        rmse = metrics.compute_rmse(true_angles, est_angles)
        
        print(f"  Estimated: {[f'{a:.1f}°' for a in est_angles]}")
        print(f"  RMSE: {rmse:.3f}°")
        
        results[f"{array_name.upper()} (N={array_data.num_sensors})"] = {
            'rmse': rmse,
            'estimated': est_angles
        }
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description="DOA Estimation Demo with MIMO Arrays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_doa_demo.py --array z5 --N 7 --K 3 --SNR 10 --plot
  python run_doa_demo.py --array nested --N1 2 --N2 3 --K 2 --compare-snr
  python run_doa_demo.py --compare-arrays --SNR 15
        """
    )
    
    # Array selection
    parser.add_argument("--array", type=str, default="z5",
                       choices=['ula', 'nested', 'tca', 'epca', 
                               'z1', 'z3_1', 'z3_2', 'z4', 'z5', 'z6'],
                       help="Array type")
    
    # Array parameters
    parser.add_argument("--N", type=int, help="Number of sensors (ULA, Z-arrays, TCA)")
    parser.add_argument("--M", type=int, help="TCA: First dimension")
    parser.add_argument("--N1", type=int, help="Nested: Inner subarray size")
    parser.add_argument("--N2", type=int, help="Nested: Outer subarray size")
    parser.add_argument("--p1", type=int, help="ePCA: First prime")
    parser.add_argument("--p2", type=int, help="ePCA: Second prime")
    parser.add_argument("--p3", type=int, help="ePCA: Third prime")
    parser.add_argument("--d", type=float, default=0.5, help="Sensor spacing")
    
    # Signal parameters
    parser.add_argument("--K", type=int, default=3, help="Number of sources")
    parser.add_argument("--angles", type=float, nargs='+', 
                       help="True DOA angles (degrees)")
    parser.add_argument("--SNR", type=float, default=10, help="SNR in dB")
    parser.add_argument("--snapshots", type=int, default=200, 
                       help="Number of snapshots")
    parser.add_argument("--wavelength", type=float, default=1.0, 
                       help="Signal wavelength")
    parser.add_argument("--resolution", type=float, default=0.5,
                       help="Angle resolution (degrees)")
    
    # Modes
    parser.add_argument("--plot", action="store_true", help="Show plots")
    parser.add_argument("--compare-snr", action="store_true",
                       help="Compare performance vs SNR")
    parser.add_argument("--compare-arrays", action="store_true",
                       help="Compare different arrays")
    
    args = parser.parse_args()
    
    # Run appropriate mode
    if args.compare_arrays:
        run_array_comparison(args)
    elif args.compare_snr:
        run_snr_comparison(args)
    else:
        run_single_estimation(args)


if __name__ == "__main__":
    main()
