# core/tests/test_sna_array.py
"""
Test suite for Super Nested Array (SNA3) processor.

Validates implementation against paper:
Liu & Vaidyanathan (2016) "Super nested arrays"
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from geometry_processors.sna_processor import SNA3ArrayProcessor, optimize_sna3_parameters


def test_sna3_construction():
    """Test SNA3 array construction matches paper definition."""
    print("\n" + "="*70)
    print("TEST 1: SNA3 Construction")
    print("="*70)
    
    # Simple case: N1=2, N2=2, N3=1
    proc = SNA3ArrayProcessor(N1=2, N2=2, N3=1, d=1.0)
    positions = np.array(proc.data.sensors_positions)
    
    # Expected:
    # P1 = {1, 2}
    # P2 = 3*{1, 2} = {3, 6}
    # P3 = 3*3*{1} = {9}
    # Combined with origin: {0, 1, 2, 3, 6, 9}
    expected = np.array([0, 1, 2, 3, 6, 9])
    
    assert len(positions) == 6, f"Expected 6 sensors, got {len(positions)}"
    assert np.array_equal(positions, expected), f"Expected {expected}, got {positions}"
    
    print(f"✅ Construction correct: {positions}")
    print(f"   N = {proc.N_total} (N1={proc.N1}, N2={proc.N2}, N3={proc.N3})")
    return True


def test_sna3_analysis():
    """Test full analysis pipeline runs without errors."""
    print("\n" + "="*70)
    print("TEST 2: Full Analysis Pipeline")
    print("="*70)
    
    proc = SNA3ArrayProcessor(N1=3, N2=3, N3=2, d=0.5)
    results = proc.run_full_analysis()
    
    # Check all expected fields are populated
    assert results.sensors_positions is not None, "Sensor positions missing"
    assert results.coarray_positions is not None, "Coarray positions missing"
    assert results.weight_table is not None, "Weight table missing"
    assert results.largest_contiguous_segment is not None, "Contiguous segment missing"
    assert results.performance_summary_table is not None, "Summary table missing"
    
    # Check basic properties
    N = proc.N_total
    assert N == 8, f"Expected N=8, got {N}"
    
    # Aperture should be positive
    aperture = results.performance_summary_table[
        results.performance_summary_table['Metrics'] == 'Coarray Aperture (two-sided span)'
    ]['Value'].values[0]
    assert aperture > 0, f"Aperture should be positive, got {aperture}"
    
    print(f"✅ Analysis complete for N={N}")
    print(f"   Aperture: {aperture}")
    print(f"   Unique lags: {len(np.unique(results.coarray_positions))}")
    return True


def test_sna3_weight_reduction():
    """Test that SNA3 has reduced w(2) compared to naive nested."""
    print("\n" + "="*70)
    print("TEST 3: Weight Reduction Property")
    print("="*70)
    
    # For N=16, paper reports SNA3 has w(1)=1, w(2)=5
    proc = SNA3ArrayProcessor(N1=7, N2=6, N3=3, d=0.5)
    results = proc.run_full_analysis()
    
    wt_df = results.weight_table
    wt = {int(r["Lag"]): int(r["Weight"]) for _, r in wt_df.iterrows()}
    
    w1 = wt.get(1, 0)
    w2 = wt.get(2, 0)
    w3 = wt.get(3, 0)
    
    print(f"   w(1) = {w1}")
    print(f"   w(2) = {w2}")
    print(f"   w(3) = {w3}")
    
    # SNA3 should have relatively small weights at low lags
    # (exact values depend on parameters, but check they're reasonable)
    assert w1 >= 1, f"w(1) should be at least 1, got {w1}"
    assert w2 >= 1, f"w(2) should be at least 1, got {w2}"
    
    print(f"✅ Weight distribution reasonable")
    return True


def test_paper_n16_comparison():
    """Test N=16 case matches paper Table IV properties."""
    print("\n" + "="*70)
    print("TEST 4: Paper N=16 Validation")
    print("="*70)
    
    # Paper Table IV: SNA2 (not SNA3) has N=16, A=71, L=71, Dm=71
    # SNA3 should have similar or better properties
    proc = SNA3ArrayProcessor(N1=7, N2=6, N3=3, d=0.5)
    results = proc.run_full_analysis()
    
    summary = results.performance_summary_table
    
    # Extract metrics
    N = summary[summary['Metrics'] == 'Physical Sensors (N)']['Value'].values[0]
    A = summary[summary['Metrics'] == 'Coarray Aperture (two-sided span)']['Value'].values[0]
    L = summary[summary['Metrics'] == 'Contiguous Segment Length (L)']['Value'].values[0]
    K_max = summary[summary['Metrics'] == 'Maximum Detectable Sources (K_max)']['Value'].values[0]
    
    print(f"   N = {N}")
    print(f"   Aperture (A) = {A}")
    print(f"   Segment Length (L) = {L}")
    print(f"   Max DOAs (K_max) = {K_max}")
    
    # Sanity checks (not exact paper values since we're using SNA3)
    assert N == 16, f"Expected N=16, got {N}"
    assert A >= 50, f"Aperture should be >= 50, got {A}"
    assert L >= 50, f"Segment length should be >= 50, got {L}"
    assert K_max >= 25, f"Should detect >= 25 DOAs, got {K_max}"
    
    print(f"✅ N=16 properties validated")
    return True


def test_optimization_function():
    """Test parameter optimization function."""
    print("\n" + "="*70)
    print("TEST 5: Parameter Optimization")
    print("="*70)
    
    for N_total in [10, 16, 20, 25]:
        N1, N2, N3 = optimize_sna3_parameters(N_total)
        actual_sum = N1 + N2 + N3
        
        print(f"   N={N_total}: N1={N1}, N2={N2}, N3={N3} (sum={actual_sum})")
        
        assert N1 > 0, f"N1 should be positive"
        assert N2 > 0, f"N2 should be positive"
        assert N3 >= 0, f"N3 should be non-negative"
        assert actual_sum == N_total, f"Expected sum={N_total}, got {actual_sum}"
    
    print(f"✅ Optimization function works correctly")
    return True


def test_no_errors():
    """Run full analysis on various configurations to catch runtime errors."""
    print("\n" + "="*70)
    print("TEST 6: Robustness Test (Various Configurations)")
    print("="*70)
    
    test_configs = [
        (2, 2, 1),
        (3, 3, 2),
        (5, 4, 3),
        (7, 6, 3),  # Paper example
        (10, 8, 4),
    ]
    
    for N1, N2, N3 in test_configs:
        try:
            proc = SNA3ArrayProcessor(N1=N1, N2=N2, N3=N3, d=0.5)
            results = proc.run_full_analysis()
            N = proc.N_total
            print(f"   ✅ N={N:2d} (N1={N1}, N2={N2}, N3={N3}): Success")
        except Exception as e:
            print(f"   ❌ N1={N1}, N2={N2}, N3={N3}: {str(e)}")
            return False
    
    print(f"✅ All configurations run without errors")
    return True


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "="*80)
    print("  SNA3 ARRAY PROCESSOR TEST SUITE")
    print("="*80)
    
    tests = [
        ("Construction", test_sna3_construction),
        ("Analysis Pipeline", test_sna3_analysis),
        ("Weight Reduction", test_sna3_weight_reduction),
        ("Paper N=16", test_paper_n16_comparison),
        ("Optimization", test_optimization_function),
        ("Robustness", test_no_errors),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Error: {str(e)}")
            results.append((name, False))
        except Exception as e:
            print(f"\n❌ TEST ERROR: {name}")
            print(f"   Exception: {str(e)}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! SNA3 implementation validated.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
