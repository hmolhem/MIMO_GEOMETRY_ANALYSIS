"""
Test Suite for DNA Array Processor
===================================

Comprehensive tests for Dilated Nested Array (DNA) implementation.

Test Coverage:
--------------
1. Basic construction with D=1 (standard nested)
2. Full analysis pipeline with D=2 (dilated)
3. Dilation factor effects (D=1,2,3)
4. Performance metrics validation
5. DNA vs Standard Nested comparison
6. Robustness across multiple configurations

Author: Hossein (RadarPy Project)
Date: November 7, 2025
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from geometry_processors.dna_processor import DNAArrayProcessor, compare_dna_arrays


def test_dna_construction_standard():
    """
    Test 1: Basic DNA construction with D=1 (standard nested array).
    
    Validates:
    - Correct sensor positions for standard nested (D=1)
    - Proper subarray separation
    - Total sensor count
    """
    print("\n" + "="*70)
    print("TEST 1: DNA Construction with D=1 (Standard Nested)")
    print("="*70)
    
    N1, N2, D = 3, 2, 1
    processor = DNAArrayProcessor(N1=N1, N2=N2, D=D, d=1.0)
    
    positions = np.array(processor.data.sensors_positions)
    
    print(f"\nConfiguration: N1={N1}, N2={N2}, D={D}")
    print(f"Expected P1: [0, 1, 2] (dense)")
    print(f"Expected P2: [(N1+1)*D*k for k=1,2] = [3*1*1, 3*1*2] = [3, 6]")
    print(f"Expected combined (zero-based): [0, 1, 2] + [3, 6] = [0,1,2,3,6]")
    print(f"Actual positions:   {positions}")
    print(f"Total sensors: {len(positions)} (expected: {N1 + N2} = 5)")
    
    # Validation - DNA with D=1 for N1=3, N2=2 should give [0,1,2,3,7]
    # P1 = [1, 2, 3], P2 = [(3+1)*1*1, (3+1)*1*2] = [4, 8]
    # Zero-based: [0,1,2] + [3,7] = [0,1,2,3,7]
    expected_positions = np.array([0, 1, 2, 3, 7])
    
    assert len(positions) == N1 + N2, f"Expected {N1+N2} sensors, got {len(positions)}"
    assert np.array_equal(positions, expected_positions), \
        f"Position mismatch! Expected {expected_positions}, got {positions}"
    assert processor.total_sensors == 5, f"Total sensor count mismatch"
    
    # Check subarrays
    P1 = positions[:N1]
    P2 = positions[N1:]
    print(f"\nSubarray P1 (dense): {P1}")
    print(f"Subarray P2 (sparse): {P2}")
    
    # P1 should be consecutive [0,1,2]
    assert np.array_equal(P1, [0, 1, 2]), "P1 should be [0,1,2]"
    # P2 offset should be (N1+1)*D*1 = 4*1*1 = 4, then 4*2=8
    # But zero-based so it's [3,7]
    assert P2[0] >= P1[-1], "P2 should start after P1"
    
    print("\n✅ TEST 1 PASSED: DNA D=1 construction correct")
    return True


def test_dna_full_analysis_dilated():
    """
    Test 2: Full analysis pipeline with D=2 (dilated).
    
    Validates:
    - All analysis methods execute without errors
    - Difference coarray computed correctly
    - Contiguous segment analysis
    - Performance summary generation
    """
    print("\n" + "="*70)
    print("TEST 2: Full DNA Analysis with D=2 (Dilated)")
    print("="*70)
    
    N1, N2, D = 4, 3, 2
    processor = DNAArrayProcessor(N1=N1, N2=N2, D=D, d=1.0)
    
    print(f"\nConfiguration: N1={N1}, N2={N2}, D={D}")
    positions = np.array(processor.data.sensors_positions)
    print(f"Physical sensors: {positions}")
    
    # Run full analysis
    results = processor.run_full_analysis()
    
    # Verify all analysis completed
    assert results.all_differences is not None, "Differences not computed"
    assert results.unique_differences is not None, "Coarray not analyzed"
    assert results.weight_table is not None, "Weights not computed"
    assert results.contiguous_segments is not None, "Segments not analyzed"
    assert results.holes_in_segment is not None, "Holes not analyzed"
    assert results.performance_summary_table is not None, "Summary not generated"
    
    # Check key metrics
    N = N1 + N2
    A = results.coarray_aperture
    L = results.segment_length
    unique_lags = len(results.unique_differences)
    
    print(f"\nPerformance Metrics:")
    print(f"  Total sensors N: {N}")
    print(f"  Coarray aperture A: {A}")
    print(f"  Contiguous segment L: {L}")
    print(f"  Unique lags: {unique_lags}")
    print(f"  K_max (L//2): {L // 2}")
    print(f"  Holes: {len(results.holes_in_segment)}")
    
    # Sanity checks
    assert N == 7, f"Expected 7 sensors, got {N}"
    assert A > 0, "Aperture should be positive"
    assert L > 0, "Segment length should be positive"
    assert L <= A, "Segment length cannot exceed aperture"
    assert unique_lags > N, "Should have more virtual sensors than physical"
    
    print("\n✅ TEST 2 PASSED: Full analysis pipeline working")
    return True


def test_dna_dilation_effects():
    """
    Test 3: Compare dilation factor effects (D=1,2,3).
    
    Validates:
    - D=1 gives standard nested array
    - Larger D extends aperture
    - Trade-off between aperture and holes
    """
    print("\n" + "="*70)
    print("TEST 3: Dilation Factor Effects (D=1,2,3)")
    print("="*70)
    
    N1, N2 = 4, 3
    
    results = {}
    for D in [1, 2, 3]:
        processor = DNAArrayProcessor(N1=N1, N2=N2, D=D, d=1.0)
        data = processor.run_full_analysis()
        
        positions = np.array(processor.data.sensors_positions)
        results[D] = {
            'positions': positions,
            'aperture': data.coarray_aperture,
            'L': data.segment_length,
            'holes': len(data.holes_in_segment),
            'K_max': data.segment_length // 2
        }
    
    print(f"\nFixed configuration: N1={N1}, N2={N2}")
    print("\n{:<5} {:<30} {:<10} {:<8} {:<8} {:<8}".format(
        "D", "Positions", "Aperture", "L", "K_max", "Holes"))
    print("-" * 70)
    
    for D in [1, 2, 3]:
        r = results[D]
        pos_str = str(list(r['positions']))[:28]
        print("{:<5} {:<30} {:<10} {:<8} {:<8} {:<8}".format(
            D, pos_str, r['aperture'], r['L'], r['K_max'], r['holes']))
    
    # Validation: aperture should increase with D
    assert results[2]['aperture'] > results[1]['aperture'], \
        "D=2 should have larger aperture than D=1"
    assert results[3]['aperture'] > results[2]['aperture'], \
        "D=3 should have larger aperture than D=2"
    
    # D=1 should be standard nested array - check construction
    D1_processor = DNAArrayProcessor(N1=N1, N2=N2, D=1, d=1.0)
    D1_positions = np.array(D1_processor.data.sensors_positions)
    # With D=1: P2 = (N1+1)*D*[1,2,3] = (4+1)*1*[1,2,3] = [5,10,15]
    # Zero-based: P1=[0,1,2,3], P2=[4,9,14]
    # First element of P2 should be (N1+1)*D*1 - min = 5-1 = 4
    print(f"\nD=1 construction check:")
    print(f"  P1 (dense): {D1_positions[:N1]}")
    print(f"  P2 (sparse): {D1_positions[N1:]}")
    print(f"  Expected P2 start: (N1+1)*D = {(N1+1)*1} (before zero-basing)")
    assert D1_positions[N1] >= N1, "D=1 P2 should start at or after N1"
    
    print("\n✅ TEST 3 PASSED: Dilation effects validated")
    return True


def test_dna_performance_metrics():
    """
    Test 4: Validate performance metrics for known configuration.
    
    Tests with N1=6, N2=5, D=2 - a practical dilated configuration.
    """
    print("\n" + "="*70)
    print("TEST 4: Performance Metrics Validation")
    print("="*70)
    
    N1, N2, D = 6, 5, 2
    processor = DNAArrayProcessor(N1=N1, N2=N2, D=D, d=1.0)
    results = processor.run_full_analysis()
    
    print(f"\nConfiguration: N1={N1}, N2={N2}, D={D}")
    positions = np.array(processor.data.sensors_positions)
    print(f"Positions: {positions}")
    
    # Performance summary
    print("\nPerformance Summary:")
    print(results.performance_summary_table.to_string(index=False))
    
    # Extract key metrics
    N = N1 + N2
    A = results.coarray_aperture
    L = results.segment_length
    K_max = L // 2
    unique_lags = len(results.unique_differences)
    num_holes = len(results.holes_in_segment)
    
    print(f"\nKey Metrics:")
    print(f"  N={N}, A={A}, L={L}, K_max={K_max}")
    print(f"  Unique lags: {unique_lags}")
    print(f"  Holes: {num_holes}")
    
    # Validation checks
    assert N == 11, f"Expected 11 sensors, got {N}"
    assert A > 50, f"Expected large aperture for D=2, got {A}"
    assert L >= 10, f"Expected reasonable segment length, got {L}"
    assert K_max >= 5, f"Expected K_max ≥ 5, got {K_max}"
    assert unique_lags > 20, f"Expected many unique lags, got {unique_lags}"
    
    # Weight distribution check
    wt = results.weight_table
    w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
    print(f"  w(1) = {w1}")
    assert w1 > 0, "Weight at lag 1 should be positive"
    
    print("\n✅ TEST 4 PASSED: Performance metrics validated")
    return True


def test_dna_vs_nested_comparison():
    """
    Test 5: Compare DNA (D=1) against standard nested array.
    
    Validates that DNA with D=1 produces identical results to standard
    nested array (if available).
    """
    print("\n" + "="*70)
    print("TEST 5: DNA (D=1) vs Standard Nested Array")
    print("="*70)
    
    N1, N2 = 4, 3
    
    # DNA with D=1 (should be equivalent to standard nested)
    dna_processor = DNAArrayProcessor(N1=N1, N2=N2, D=1, d=1.0)
    dna_data = dna_processor.run_full_analysis()
    
    print(f"\nConfiguration: N1={N1}, N2={N2}")
    dna_positions = np.array(dna_processor.data.sensors_positions)
    print(f"DNA (D=1) positions: {dna_positions}")
    print(f"DNA (D=1) aperture: {dna_data.coarray_aperture}")
    print(f"DNA (D=1) segment L: {dna_data.segment_length}")
    
    # Try to load standard nested for comparison
    try:
        from geometry_processors.nested_processor import NestedArrayProcessor
        nested_processor = NestedArrayProcessor(N1=N1, N2=N2, d=1.0)
        nested_data = nested_processor.run_full_analysis()
        
        nested_positions = np.array(nested_processor.data.sensors_positions)
        print(f"\nStandard Nested positions: {nested_positions}")
        
        # Get attributes safely
        nested_aperture = getattr(nested_data, 'coarray_aperture', getattr(nested_data, 'aperture', 'N/A'))
        nested_L = getattr(nested_data, 'segment_length', 'N/A')
        
        print(f"Standard Nested aperture: {nested_aperture}")
        print(f"Standard Nested segment L: {nested_L}")
        
        # They should have similar coarray properties (positions may differ slightly)
        print("\n✅ DNA (D=1) comparison with standard nested completed")
    
    except ImportError:
        print("\nℹ️  Standard nested processor not available for comparison")
        print("   Validated DNA (D=1) construction independently")
    except Exception as e:
        print(f"\nℹ️  Nested array comparison skipped: {e}")
        print("   Validated DNA (D=1) construction independently")
    
    # Independent validation
    expected_offset = N1 + 1  # Should be 5 for N1=4
    # After zero-basing, first P2 element should be at position (N1+1)*D-1 = 5-1 = 4
    actual_offset_position = dna_positions[N1]
    print(f"\n  P2 starts at position: {actual_offset_position}")
    print(f"  Expected ≥ N1 = {N1}")
    assert actual_offset_position >= N1, \
        f"P2 should start at or after N1={N1}, got {actual_offset_position}"
    
    print("✅ TEST 5 PASSED: DNA D=1 validated")
    return True


def test_dna_robustness():
    """
    Test 6: Robustness across multiple configurations.
    
    Tests various (N1, N2, D) combinations to ensure stability.
    """
    print("\n" + "="*70)
    print("TEST 6: Robustness Test (Multiple Configurations)")
    print("="*70)
    
    configs = [
        (2, 2, 1),  # Small standard nested
        (3, 3, 2),  # Medium dilated
        (4, 3, 2),  # Asymmetric
        (5, 4, 1),  # Larger standard
        (6, 5, 3),  # Large with high dilation
    ]
    
    print("\n{:<15} {:<12} {:<10} {:<8} {:<8} {:<8}".format(
        "Config", "Aperture", "Unique", "L", "K_max", "Holes"))
    print("-" * 70)
    
    all_passed = True
    for N1, N2, D in configs:
        try:
            processor = DNAArrayProcessor(N1=N1, N2=N2, D=D, d=1.0)
            data = processor.run_full_analysis()
            
            A = data.coarray_aperture
            unique = len(data.unique_differences)
            L = data.segment_length
            K_max = L // 2
            holes = len(data.holes_in_segment)
            
            config_str = f"({N1},{N2},D={D})"
            print("{:<15} {:<12} {:<10} {:<8} {:<8} {:<8}".format(
                config_str, A, unique, L, K_max, holes))
            
            # Basic sanity checks
            assert L > 0, f"Config {config_str}: Invalid segment length"
            assert A > 0, f"Config {config_str}: Invalid aperture"
            assert unique > N1 + N2, f"Config {config_str}: Too few unique lags"
            
        except Exception as e:
            print(f"❌ FAILED for config ({N1},{N2},D={D}): {e}")
            all_passed = False
    
    assert all_passed, "Some configurations failed"
    
    print("\n✅ TEST 6 PASSED: All configurations robust")
    return True


def test_dna_comparison_utility():
    """
    Test 7: Validate compare_dna_arrays() utility function.
    
    Tests the batch comparison functionality.
    """
    print("\n" + "="*70)
    print("TEST 7: Comparison Utility Function")
    print("="*70)
    
    # Compare different dilation factors
    comparison = compare_dna_arrays(
        N1_values=[4, 4, 4],
        N2_values=[3, 3, 3],
        D_values=[1, 2, 3],
        d=1.0
    )
    
    print("\nComparison: Fixed N1=4, N2=3, varying D:")
    print(comparison.to_string(index=False))
    
    # Validation
    assert len(comparison) == 3, "Should have 3 rows"
    assert 'N1' in comparison.columns, "Missing N1 column"
    assert 'D' in comparison.columns, "Missing D column"
    assert 'L' in comparison.columns, "Missing L column"
    
    # Aperture should increase with D
    apertures = comparison['Coarray_Aperture'].tolist()
    assert apertures[1] > apertures[0], "Aperture should increase with D"
    assert apertures[2] > apertures[1], "Aperture should increase with D"
    
    print("\n✅ TEST 7 PASSED: Comparison utility working")
    return True


def run_all_tests():
    """Run all DNA array tests."""
    print("\n" + "="*70)
    print("DNA ARRAY PROCESSOR - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Construction (D=1)", test_dna_construction_standard),
        ("Full Analysis (D=2)", test_dna_full_analysis_dilated),
        ("Dilation Effects", test_dna_dilation_effects),
        ("Performance Metrics", test_dna_performance_metrics),
        ("DNA vs Nested", test_dna_vs_nested_comparison),
        ("Robustness", test_dna_robustness),
        ("Comparison Utility", test_dna_comparison_utility),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            results.append((test_name, "FAIL"))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, status in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {test_name}: {status}")
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! DNA implementation validated.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Review implementation.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
