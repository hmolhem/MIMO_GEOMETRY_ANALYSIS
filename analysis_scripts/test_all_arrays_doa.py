"""Comprehensive test of DOA estimation across ALL array geometries."""
import sys
import os
import numpy as np

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from doa_estimation.music import MUSICEstimator
from doa_estimation.metrics import DOAMetrics

# Import ALL array processors
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

print("="*70)
print("COMPREHENSIVE DOA TEST - ALL ARRAY GEOMETRIES")
print("="*70)

# Test configurations: (name, processor, test_K_sources)
# IMPORTANT: Use d=1.0 and wavelength=2.0 to ensure d/lambda = 0.5 (avoid spatial aliasing)
# Note: TCA and ePCA have issues with fractional spacing (d=0.5), so we use d=1.0
WAVELENGTH = 2.0  # Use 2.0 to get d/lambda = 0.5 with d=1.0

test_configs = [
    ("ULA (N=8)", ULArrayProcessor(N=8, d=1.0), 3),
    ("ULA (N=10)", ULArrayProcessor(N=10, d=1.0), 4),
    ("Nested (N1=2, N2=3)", NestedArrayProcessor(N1=2, N2=3, d=1.0), 2),
    ("Nested (N1=3, N2=4)", NestedArrayProcessor(N1=3, N2=4, d=1.0), 4),
    ("TCA (M=3, N=4)", TCAArrayProcessor(M=3, N=4, d=1.0), 3),
    ("TCA (M=4, N=5)", TCAArrayProcessor(M=4, N=5, d=1.0), 4),
    ("ePCA (2,3,5)", ePCAArrayProcessor(p1=2, p2=3, p3=5, N1=2, N2=2, N3=2, d=1.0), 3),
    ("Z1 (N=7)", Z1ArrayProcessor(N=7, d=1.0), 3),
    ("Z1 (N=10)", Z1ArrayProcessor(N=10, d=1.0), 4),
    ("Z3_1 (N=6)", Z3_1ArrayProcessor(N=6, d=1.0), 3),
    ("Z3_2 (N=6)", Z3_2ArrayProcessor(N=6, d=1.0), 3),
    ("Z3_2 (N=7)", Z3_2ArrayProcessor(N=7, d=1.0), 4),
    ("Z4 (N=7)", Z4ArrayProcessor(N=7, d=1.0), 3),
    ("Z5 (N=7)", Z5ArrayProcessor(N=7, d=1.0), 4),
    ("Z6 (N=7)", Z6ArrayProcessor(N=7, d=1.0), 4),
]

results = []
metrics_calc = DOAMetrics()

for name, processor, K_test in test_configs:
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"{'='*70}")
    
    try:
        # Run geometry analysis
        array_data = processor.run_full_analysis()
        
        # Extract K_max - handle different table formats
        K_max = None
        if hasattr(array_data, 'performance_summary_table') and array_data.performance_summary_table is not None:
            perf = array_data.performance_summary_table
            # Debug: show columns
            print(f"  DEBUG: Table columns = {list(perf.columns)}")
            
            # Check if it has 'Metrics' column (most arrays)
            if 'Metrics' in perf.columns:
                k_row = perf[perf['Metrics'].str.contains('K_max', case=False, na=False)]
                if not k_row.empty:
                    K_max = int(k_row['Value'].iloc[0])
            # Or check for direct K_max column (some arrays like TCA, ePCA)
            elif 'K_max' in perf.columns:
                K_max = int(perf['K_max'].iloc[0])
                print(f"  DEBUG: Extracted K_max={K_max} from direct column")
            # Or search in first column
            else:
                first_col = perf.columns[0]
                k_row = perf[perf[first_col].astype(str).str.contains('K_max', case=False, na=False)]
                if not k_row.empty:
                    K_max = int(perf.iloc[k_row.index[0], 1])
        
        if K_max is None:
            # Fallback for processors with different structure
            if hasattr(processor, 'generate_performance_summary'):
                summary = processor.generate_performance_summary()
                K_max = summary.get('Maximum Detectable Sources (K_max)', array_data.num_sensors - 1)
            else:
                K_max = array_data.num_sensors - 1
        
        print(f"  Physical Sensors: {array_data.num_sensors}")
        print(f"  Sensor Positions: {array_data.sensors_positions}")
        print(f"  K_max: {K_max}")
        print(f"  Testing with K={K_test} sources")
        
        # Check if K_test is valid
        if K_max is None or K_max == 0:
            print(f"  ✗ ERROR: Could not determine K_max for this array")
            print(f"     Table columns: {list(array_data.performance_summary_table.columns) if hasattr(array_data, 'performance_summary_table') else 'NO TABLE'}")
            results.append({
                'name': name,
                'N': array_data.num_sensors,
                'K_max': '?',
                'K_test': K_test,
                'rmse': 999,
                'mae': 999,
                'max_error': 999,
                'success': False,
                'error': 'K_max extraction failed'
            })
            continue
        
        if K_test > K_max:
            print(f"  ⚠ WARNING: K_test ({K_test}) > K_max ({K_max}), adjusting to K_max")
            K_test = K_max
        
        # Generate well-separated angles based on K
        if K_test == 2:
            true_angles = [-30, 30]
        elif K_test == 3:
            true_angles = [-40, 0, 40]
        elif K_test == 4:
            true_angles = [-50, -15, 20, 50]
        elif K_test == 5:
            true_angles = [-60, -25, 0, 30, 60]
        else:
            # Generate evenly spaced
            true_angles = list(np.linspace(-60, 60, K_test))
        
        print(f"  True angles: {true_angles}")
        
        # Create MUSIC estimator with proper wavelength
        estimator = MUSICEstimator(
            sensor_positions=array_data.sensors_positions,
            wavelength=WAVELENGTH,
            angle_range=(-90, 90),
            angle_resolution=0.5
        )
        
        # Simulate signals with high SNR for reliable testing
        X = estimator.simulate_signals(
            true_angles=true_angles,
            SNR_dB=25,
            snapshots=500,
            signal_type='random'
        )
        
        # Estimate DOA
        estimated_angles = estimator.estimate(X, K_sources=K_test)
        
        print(f"  Estimated: {[f'{a:.1f}°' for a in estimated_angles]}")
        
        # Compute metrics
        rmse = metrics_calc.compute_rmse(true_angles, estimated_angles)
        mae = metrics_calc.compute_mae(true_angles, estimated_angles)
        max_error = metrics_calc.compute_max_error(true_angles, estimated_angles)
        
        print(f"  RMSE: {rmse:.3f}°")
        print(f"  MAE:  {mae:.3f}°")
        print(f"  Max Error: {max_error:.3f}°")
        
        # Determine success
        success = rmse < 2.0  # Accept < 2° error as success
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  Status: {status}")
        
        results.append({
            'name': name,
            'N': array_data.num_sensors,
            'K_max': K_max,
            'K_test': K_test,
            'rmse': rmse,
            'mae': mae,
            'max_error': max_error,
            'success': success
        })
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'name': name,
            'N': '?',
            'K_max': '?',
            'K_test': K_test,
            'rmse': 999,
            'mae': 999,
            'max_error': 999,
            'success': False,
            'error': str(e)
        })

# Summary table
print("\n" + "="*70)
print("SUMMARY: DOA ESTIMATION ACROSS ALL ARRAY TYPES")
print("="*70)
print(f"\n{'Array':<25} {'N':<4} {'K_max':<6} {'K_test':<6} {'RMSE':<8} {'Status':<10}")
print("-"*70)

total_tests = len(results)
passed_tests = 0

for r in results:
    status_str = "✓ PASS" if r['success'] else "✗ FAIL"
    if 'error' in r:
        status_str = "✗ ERROR"
    else:
        if r['success']:
            passed_tests += 1
    
    print(f"{r['name']:<25} {str(r['N']):<4} {str(r['K_max']):<6} {str(r['K_test']):<6} "
          f"{r['rmse']:>6.3f}°  {status_str:<10}")

print("-"*70)
print(f"\nResults: {passed_tests}/{total_tests} tests passed ({100*passed_tests/total_tests:.1f}%)")

if passed_tests == total_tests:
    print("\n🎉 SUCCESS! DOA estimation works correctly for ALL array types!")
else:
    print(f"\n⚠ {total_tests - passed_tests} arrays failed - investigation needed")

print("\n" + "="*70)
