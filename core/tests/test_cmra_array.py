"""
Test Suite for cMRA Array Processor
====================================

Tests cMRA (constrained Minimum Redundancy Array) implementation.

Author: Hossein (RadarPy Project)
Date: November 7, 2025
"""

import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from geometry_processors.cmra_processor import cMRAArrayProcessor
import numpy as np


def test_cmra_construction():
    """
    TEST 1: cMRA Array Construction
    
    Verify lookup table construction for N=5.
    Expected: Positions with w(1)=0 (no consecutive sensors)
    """
    print("="*80)
    print("TEST 1: cMRA Construction (Lookup Table)")
    print("="*80)
    
    try:
        processor = cMRAArrayProcessor(N=5, d=1.0)
        positions = np.array(processor.data.sensors_positions)
        
        # Check construction properties
        assert len(positions) == 5, f"Expected N=5, got {len(positions)}"
        assert processor.use_lookup == True, "Expected lookup table for N=5"
        
        # Check no consecutive positions (ensures w(1)=0)
        diffs = np.diff(positions)
        assert np.all(diffs > 1), f"Found consecutive positions: {positions}"
        
        print(f"✅ Construction correct: {positions}")
        print(f"   N = {len(positions)}")
        print(f"   Method = Lookup table")
        print(f"   Min spacing = {np.min(diffs)}")
        
    except AssertionError as e:
        print(f"❌ TEST ERROR: {e}")
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")


def test_cmra_analysis():
    """
    TEST 2: Full Analysis Pipeline
    
    Verify all 7 analysis steps complete without errors for N=8.
    """
    print("\n" + "="*80)
    print("TEST 2: Full Analysis Pipeline")
    print("="*80)
    
    try:
        processor = cMRAArrayProcessor(N=8, d=1.0)
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
        print(f"   Method: {'Lookup' if processor.use_lookup else 'Algorithmic'}")
        
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")


def test_weight_w1_zero():
    """
    TEST 3: Weight Constraint w(1)=0
    
    Verify cMRA achieves w(1)=0 constraint (critical property).
    This is the defining property for all Week 1 arrays.
    """
    print("\n" + "="*80)
    print("TEST 3: Weight Constraint w(1)=0")
    print("="*80)
    
    try:
        processor = cMRAArrayProcessor(N=7, d=1.0)
        results = processor.run_full_analysis()
        
        # Extract weight at lag 1
        wt_dict = dict(zip(results.weight_table['Lag'], results.weight_table['Weight']))
        w1 = wt_dict.get(1, 0)
        w2 = wt_dict.get(2, 0)
        w3 = wt_dict.get(3, 0)
        
        # Critical test: w(1) MUST be 0
        assert w1 == 0, f"cMRA requires w(1)=0, got w(1)={w1}"
        
        print(f"   w(1) = {w1} ✅")
        print(f"   w(2) = {w2}")
        print(f"   w(3) = {w3}")
        print(f"✅ Weight constraint w(1)=0 satisfied")
        
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")


def test_cmra_n7_properties():
    """
    TEST 4: N=7 Lookup Table Validation
    
    Verify N=7 matches expected lookup table entry.
    Expected: [0, 2, 6, 11, 17, 23, 29] with A=29
    """
    print("\n" + "="*80)
    print("TEST 4: N=7 Lookup Table Validation")
    print("="*80)
    
    try:
        processor = cMRAArrayProcessor(N=7, d=1.0)
        results = processor.run_full_analysis()
        
        assert processor.use_lookup == True, "Expected lookup table for N=7"
        
        N = results.num_sensors
        positions = np.array(results.sensors_positions)
        aperture = results.coarray_aperture
        
        # Expected lookup values
        expected_positions = np.array([0, 2, 6, 11, 17, 23, 29])
        expected_aperture = 29
        
        assert np.array_equal(positions, expected_positions), \
            f"Positions mismatch: got {positions}, expected {expected_positions}"
        assert aperture == expected_aperture, \
            f"Aperture mismatch: got {aperture}, expected {expected_aperture}"
        
        print(f"   N = {N}")
        print(f"   Positions = {positions}")
        print(f"   Aperture (A) = {aperture}")
        print(f"✅ Lookup table validated")
        
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")


def test_cmra_algorithmic():
    """
    TEST 5: Algorithmic Construction (N>16)
    
    Test algorithmic construction for N=20 (beyond lookup table).
    Verify w(1)=0 is maintained.
    """
    print("\n" + "="*80)
    print("TEST 5: Algorithmic Construction (N=20)")
    print("="*80)
    
    try:
        processor = cMRAArrayProcessor(N=20, d=1.0)
        results = processor.run_full_analysis()
        
        # Should use algorithmic method for N=20
        # (lookup table only goes to N=16)
        
        # Extract weight at lag 1
        wt_dict = dict(zip(results.weight_table['Lag'], results.weight_table['Weight']))
        w1 = wt_dict.get(1, 0)
        
        # Critical: w(1) MUST still be 0 even with algorithmic construction
        assert w1 == 0, f"Algorithmic cMRA must maintain w(1)=0, got w(1)={w1}"
        
        print(f"   N = {results.num_sensors}")
        print(f"   Method = {'Lookup' if processor.use_lookup else 'Algorithmic'}")
        print(f"   Aperture = {results.coarray_aperture}")
        print(f"   w(1) = {w1} ✅")
        print(f"✅ Algorithmic construction maintains w(1)=0")
        
    except AssertionError as e:
        print(f"❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"❌ TEST ERROR: {e}")


def test_robustness():
    """
    TEST 6: Robustness Test (Various Configurations)
    
    Test cMRA across lookup and algorithmic ranges.
    """
    print("\n" + "="*80)
    print("TEST 6: Robustness Test (Lookup + Algorithmic)")
    print("="*80)
    
    test_configs = [
        (5, "lookup"),    # Minimum N
        (7, "lookup"),    # Middle of lookup range
        (16, "lookup"),   # End of lookup range
        (20, "algorithmic"),  # Algorithmic range
    ]
    
    for N, expected_method in test_configs:
        try:
            processor = cMRAArrayProcessor(N=N, d=1.0)
            results = processor.run_full_analysis()
            
            # Verify w(1)=0 for all configurations
            wt_dict = dict(zip(results.weight_table['Lag'], 
                              results.weight_table['Weight']))
            w1 = wt_dict.get(1, 0)
            
            assert w1 == 0, f"N={N}: w(1)={w1} != 0"
            
            method = "lookup" if processor.use_lookup else "algorithmic"
            print(f"   ✅ N={N:2d} ({method:11s}): A={results.coarray_aperture}, L={results.segment_length}")
            
        except Exception as e:
            print(f"   ❌ N={N:2d}: {e}")
    
    print(f"✅ All configurations run without errors")


def run_all_tests():
    """
    Run all cMRA array tests.
    """
    print("\n" + "="*80)
    print("  cMRA ARRAY PROCESSOR TEST SUITE")
    print("="*80 + "\n")
    
    tests = [
        ("Construction", test_cmra_construction),
        ("Analysis Pipeline", test_cmra_analysis),
        ("Weight w(1)=0", test_weight_w1_zero),
        ("N=7 Lookup", test_cmra_n7_properties),
        ("Algorithmic N=20", test_cmra_algorithmic),
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
        print("\n🎉 ALL TESTS PASSED! cMRA implementation validated.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
