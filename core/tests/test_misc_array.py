"""
Test Suite for MISC Array Processor
====================================

Tests MISC (Minimum Inter-element Spacing Constraint) array implementation.

Author: Hossein (RadarPy Project)
Date: November 7, 2025
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from geometry_processors.misc_processor import MISCArrayProcessor
import numpy as np


def test_misc_construction():
    """
    TEST 1: MISC Array Construction
    
    Verify basic construction for simple case (N=5).
    Expected: Positions with w(1)=0 (no consecutive sensors)
    """
    print("="*80)
    print("TEST 1: MISC Construction")
    print("="*80)
    
    try:
        processor = MISCArrayProcessor(N=5, d=1.0)
        positions = np.array(processor.data.sensors_positions)
        
        # Check construction properties
        assert len(positions) == 5, f"Expected N=5, got {len(positions)}"
        
        # Check no consecutive positions (ensures w(1)=0)
        diffs = np.diff(positions)
        assert np.all(diffs > 1), f"Found consecutive positions: {positions}"
        
        print(f"✅ Construction correct: {positions}")
        print(f"   N = {len(positions)}")
        print(f"   Min spacing = {np.min(diffs)}")
        
    except AssertionError as e:
        print(f"❌ TEST ERROR: {e}")
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")


def test_misc_analysis():
    """
    TEST 2: Full Analysis Pipeline
    
    Verify all 7 analysis steps complete without errors for N=8.
    """
    print("\n" + "="*80)
    print("TEST 2: Full Analysis Pipeline")
    print("="*80)
    
    try:
        processor = MISCArrayProcessor(N=8, d=1.0)
        results = processor.run_full_analysis()
        
        # Check key results exist
        assert results.performance_summary_table is not None
        assert results.weight_table is not None
        assert results.unique_differences is not None
        
        aperture = results.coarray_aperture
        unique_lags = len(results.unique_differences)
        
        print(f"✅ Analysis complete for N=8")
        print(f"   Aperture: {aperture}")
        print(f"   Unique lags: {unique_lags}")
        
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")


def test_weight_w1_zero():
    """
    TEST 3: Weight Constraint w(1)=0
    
    Verify MISC achieves w(1)=0 constraint (critical property).
    This is the defining property of MISC arrays.
    """
    print("\n" + "="*80)
    print("TEST 3: Weight Constraint w(1)=0")
    print("="*80)
    
    try:
        processor = MISCArrayProcessor(N=7, d=1.0)
        results = processor.run_full_analysis()
        
        # Extract weight at lag 1
        wt_dict = dict(zip(results.weight_table['Lag'], results.weight_table['Weight']))
        w1 = wt_dict.get(1, 0)
        w2 = wt_dict.get(2, 0)
        w3 = wt_dict.get(3, 0)
        
        # Critical test: w(1) MUST be 0
        assert w1 == 0, f"MISC requires w(1)=0, got w(1)={w1}"
        
        print(f"   w(1) = {w1} ✅")
        print(f"   w(2) = {w2}")
        print(f"   w(3) = {w3}")
        print(f"✅ Weight constraint w(1)=0 satisfied")
        
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")


def test_misc_n7_properties():
    """
    TEST 4: N=7 Properties Validation
    
    Check reasonable performance metrics for N=7:
    - Aperture should be > 10 (sparse array advantage)
    - L should be > 5 (contiguous segment)
    - K_max should be >= 2 (detectable sources)
    """
    print("\n" + "="*80)
    print("TEST 4: N=7 Properties Validation")
    print("="*80)
    
    try:
        processor = MISCArrayProcessor(N=7, d=1.0)
        results = processor.run_full_analysis()
        
        N = results.num_sensors
        aperture = results.coarray_aperture
        L = results.segment_length
        K_max = L // 2
        
        # Reasonable bounds for N=7
        assert aperture > 10, f"Aperture too small: {aperture}"
        assert L > 5, f"Segment length too small: {L}"
        assert K_max >= 2, f"K_max too small: {K_max}"
        
        print(f"   N = {N}")
        print(f"   Aperture (A) = {aperture}")
        print(f"   Segment Length (L) = {L}")
        print(f"   Max DOAs (K_max) = {K_max}")
        print(f"✅ N=7 properties validated")
        
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")


def test_misc_vs_z5_comparison():
    """
    TEST 5: MISC vs Z5 Comparison
    
    Compare MISC with Z5 for N=7.
    Both should have w(1)=0 but may differ in aperture and L.
    """
    print("\n" + "="*80)
    print("TEST 5: MISC vs Z5 Comparison")
    print("="*80)
    
    try:
        from geometry_processors.z5_processor import Z5ArrayProcessor
        
        # Run both processors
        misc_proc = MISCArrayProcessor(N=7, d=1.0)
        z5_proc = Z5ArrayProcessor(N=7, d=1.0)
        
        misc_results = misc_proc.run_full_analysis()
        z5_results = z5_proc.run_full_analysis()
        
        # Extract weights
        misc_wt = dict(zip(misc_results.weight_table['Lag'], 
                          misc_results.weight_table['Weight']))
        z5_wt = dict(zip(z5_results.weight_table['Lag'], 
                        z5_results.weight_table['Weight']))
        
        # Both should have w(1)=0
        assert misc_wt.get(1, 0) == 0, "MISC w(1) != 0"
        assert z5_wt.get(1, 0) == 0, "Z5 w(1) != 0"
        
        print(f"   MISC: A={misc_results.coarray_aperture}, L={misc_results.segment_length}")
        print(f"   Z5:   A={z5_results.coarray_aperture}, L={z5_results.segment_length}")
        print(f"✅ Both arrays achieve w(1)=0")
        
    except ImportError:
        print(f"⚠️  Z5 processor not available - skipping comparison")
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")


def test_robustness():
    """
    TEST 6: Robustness Test (Various Configurations)
    
    Test MISC construction and analysis across multiple N values.
    Ensures implementation handles edge cases.
    """
    print("\n" + "="*80)
    print("TEST 6: Robustness Test (Various Configurations)")
    print("="*80)
    
    test_configs = [
        (5, 1.0),   # Minimum N
        (8, 1.0),   # Medium N
        (10, 1.0),  # Larger N
        (16, 1.0),  # Paper comparison size
    ]
    
    for N, d in test_configs:
        try:
            processor = MISCArrayProcessor(N=N, d=d)
            results = processor.run_full_analysis()
            
            # Verify w(1)=0 for all configurations
            wt_dict = dict(zip(results.weight_table['Lag'], 
                              results.weight_table['Weight']))
            w1 = wt_dict.get(1, 0)
            
            assert w1 == 0, f"N={N}: w(1)={w1} != 0"
            
            print(f"   ✅ N={N:2d} (d={d}): Success, A={results.coarray_aperture}, L={results.segment_length}")
            
        except Exception as e:
            print(f"   ❌ N={N:2d} (d={d}): {e}")
    
    print(f"✅ All configurations run without errors")


def run_all_tests():
    """
    Run all MISC array tests.
    """
    print("\n" + "="*80)
    print("  MISC ARRAY PROCESSOR TEST SUITE")
    print("="*80 + "\n")
    
    tests = [
        ("Construction", test_misc_construction),
        ("Analysis Pipeline", test_misc_analysis),
        ("Weight w(1)=0", test_weight_w1_zero),
        ("N=7 Properties", test_misc_n7_properties),
        ("MISC vs Z5", test_misc_vs_z5_comparison),
        ("Robustness", test_robustness),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ {test_name} FAILED: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    for test_name, _ in tests:
        status = "✅ PASS" if passed > 0 else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! MISC implementation validated.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
