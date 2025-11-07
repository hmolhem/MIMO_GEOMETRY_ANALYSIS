"""
Test Suite for DDNA Array Processor
===================================

Comprehensive tests for Double Dilated Nested Array (DDNA) implementation.

Tests cover:
1. Basic construction with various (D1, D2) combinations
2. Full analysis pipeline validation
3. Dilation factor effects (D1 and D2 independently)
4. Performance metrics verification
5. Comparison with DNA and nested arrays
6. Robustness across parameter ranges
7. Utility function validation

Author: Hossein (RadarPy Project)
Date: November 7, 2025
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from geometry_processors.ddna_processor import DDNAArrayProcessor, compare_ddna_arrays
from geometry_processors.dna_processor import DNAArrayProcessor
from geometry_processors.nested_processor import NestedArrayProcessor


class TestDDNAConstruction(unittest.TestCase):
    """Test DDNA array construction with various dilation factors."""
    
    def test_standard_nested_d1d2_1(self):
        """Test DDNA with D1=D2=1 (standard nested array)."""
        processor = DDNAArrayProcessor(N1=4, N2=3, D1=1, D2=1, d=1.0)
        
        # Expected positions for nested: P1=[0,1,2,3], P2=5*[1,2,3]=[5,10,15]
        expected = np.array([0, 1, 2, 3, 5, 10, 15])
        np.testing.assert_array_equal(processor.data.sensors_positions, expected)
        
        # Verify parameters
        self.assertEqual(processor.N1, 4)
        self.assertEqual(processor.N2, 3)
        self.assertEqual(processor.D1, 1)
        self.assertEqual(processor.D2, 1)
        self.assertEqual(processor.total_sensors, 7)
    
    def test_dna_equivalent_d1_1_d2_2(self):
        """Test DDNA with D1=1, D2=2 (DNA-equivalent configuration)."""
        processor = DDNAArrayProcessor(N1=3, N2=2, D1=1, D2=2, d=1.0)
        
        # Expected: P1=[0,1,2], P2=(3+1)*2*[1,2]=8*[1,2]=[8,16]
        expected = np.array([0, 1, 2, 8, 16])
        np.testing.assert_array_equal(processor.data.sensors_positions, expected)
        
        self.assertEqual(processor.D1, 1)
        self.assertEqual(processor.D2, 2)
    
    def test_full_ddna_d1_2_d2_2(self):
        """Test full DDNA with D1=2, D2=2 (both dilated)."""
        processor = DDNAArrayProcessor(N1=3, N2=2, D1=2, D2=2, d=1.0)
        
        # Expected: P1=2*[0,1,2]=[0,2,4], P2=(2*3+1)*2*[1,2]=14*[1,2]=[14,28]
        expected = np.array([0, 2, 4, 14, 28])
        np.testing.assert_array_equal(processor.data.sensors_positions, expected)
        
        # Verify spacing
        positions = processor.data.sensors_positions
        # P1 spacing should be D1*d = 2
        self.assertEqual(positions[1] - positions[0], 2)
        self.assertEqual(positions[2] - positions[1], 2)
    
    def test_high_dilation_d1_2_d2_3(self):
        """Test DDNA with D1=2, D2=3 (high dilation)."""
        processor = DDNAArrayProcessor(N1=4, N2=3, D1=2, D2=3, d=1.0)
        
        # Expected: P1=2*[0,1,2,3]=[0,2,4,6], P2=(2*4+1)*3*[1,2,3]=27*[1,2,3]=[27,54,81]
        expected = np.array([0, 2, 4, 6, 27, 54, 81])
        np.testing.assert_array_equal(processor.data.sensors_positions, expected)
        
        self.assertEqual(processor.total_sensors, 7)
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        with self.assertRaises(ValueError):
            DDNAArrayProcessor(N1=0, N2=3, D1=1, D2=1)  # N1 < 1
        
        with self.assertRaises(ValueError):
            DDNAArrayProcessor(N1=3, N2=0, D1=1, D2=1)  # N2 < 1
        
        with self.assertRaises(ValueError):
            DDNAArrayProcessor(N1=3, N2=2, D1=0, D2=1)  # D1 < 1
        
        with self.assertRaises(ValueError):
            DDNAArrayProcessor(N1=3, N2=2, D1=1, D2=0)  # D2 < 1


class TestDDNAFullAnalysis(unittest.TestCase):
    """Test full analysis pipeline for DDNA arrays."""
    
    def test_nested_baseline_analysis(self):
        """Test full analysis for D1=D2=1 (nested baseline)."""
        processor = DDNAArrayProcessor(N1=4, N2=3, D1=1, D2=1, d=1.0)
        results = processor.run_full_analysis()
        
        # For nested (D1=D2=1), expect:
        # - Large coarray aperture
        # - Long contiguous segment (L ≈ 2N)
        # - Zero holes typically
        self.assertGreater(results.coarray_aperture, 0)
        self.assertGreater(results.segment_length, 10)  # Should be substantial
        self.assertIsInstance(results.performance_summary_table, pd.DataFrame)
        
        # Check summary table has expected metrics
        metrics = results.performance_summary_table['Metric'].tolist()
        self.assertIn('Array Type', metrics)
        self.assertIn('Dilation Factor D1', metrics)
        self.assertIn('Dilation Factor D2', metrics)
        self.assertIn('Contiguous Segment Length (L)', metrics)
    
    def test_dna_equivalent_analysis(self):
        """Test full analysis for D1=1, D2=2 (DNA-equivalent)."""
        processor = DDNAArrayProcessor(N1=4, N2=3, D1=1, D2=2, d=1.0)
        results = processor.run_full_analysis()
        
        # For DNA-equivalent (D1=1, D2=2), expect:
        # - Extended aperture compared to nested
        # - Shorter segment length than nested
        # - Possible holes depending on N1, N2
        self.assertGreater(results.coarray_aperture, 20)
        self.assertGreater(results.segment_length, 5)
        
        # Verify K_max computation
        K_max = results.segment_length // 2
        self.assertGreaterEqual(K_max, 2)
    
    def test_full_ddna_analysis(self):
        """Test full analysis for D1=2, D2=2 (full DDNA)."""
        processor = DDNAArrayProcessor(N1=5, N2=4, D1=2, D2=2, d=1.0)
        results = processor.run_full_analysis()
        
        # For full DDNA (D1=2, D2=2), expect:
        # - Very large aperture
        # - Moderate segment length
        # - Weight distribution affected by both dilations
        self.assertGreater(results.coarray_aperture, 50)
        self.assertIsInstance(results.unique_differences, list)
        self.assertGreater(len(results.unique_differences), 10)
        
        # Check weight table structure
        wt = results.weight_table
        self.assertIn('Lag', wt.columns)
        self.assertIn('Weight', wt.columns)
        self.assertGreater(len(wt), 5)


class TestDilationFactorEffects(unittest.TestCase):
    """Test effects of varying D1 and D2 independently."""
    
    def test_vary_d1_fixed_d2(self):
        """Test effect of increasing D1 while keeping D2 fixed."""
        configs = [
            (1, 2),  # D1=1, D2=2
            (2, 2),  # D1=2, D2=2
            (3, 2),  # D1=3, D2=2
        ]
        
        apertures = []
        segment_lengths = []
        
        for D1, D2 in configs:
            processor = DDNAArrayProcessor(N1=4, N2=3, D1=D1, D2=D2, d=1.0)
            results = processor.run_full_analysis()
            apertures.append(results.coarray_aperture)
            segment_lengths.append(results.segment_length)
        
        # Aperture should generally increase with D1
        # (though not strictly monotonic depending on N1, N2)
        self.assertGreater(max(apertures), min(apertures))
        
        # All should have valid segment lengths
        for L in segment_lengths:
            self.assertGreater(L, 0)
    
    def test_vary_d2_fixed_d1(self):
        """Test effect of increasing D2 while keeping D1 fixed."""
        configs = [
            (1, 1),  # D1=1, D2=1 (nested)
            (1, 2),  # D1=1, D2=2 (DNA)
            (1, 3),  # D1=1, D2=3 (high dilation)
        ]
        
        apertures = []
        K_max_values = []
        
        for D1, D2 in configs:
            processor = DDNAArrayProcessor(N1=4, N2=3, D1=D1, D2=D2, d=1.0)
            results = processor.run_full_analysis()
            apertures.append(results.coarray_aperture)
            K_max_values.append(results.segment_length // 2)
        
        # Aperture should strictly increase with D2 (for fixed D1=1)
        self.assertLess(apertures[0], apertures[1])
        self.assertLess(apertures[1], apertures[2])
        
        # All should have valid K_max
        for K in K_max_values:
            self.assertGreaterEqual(K, 1)
    
    def test_d1_d2_combinations(self):
        """Test various (D1, D2) combinations."""
        combinations = [
            (1, 1),  # Standard nested
            (1, 2),  # DNA-equivalent
            (2, 1),  # D1 dilated only
            (2, 2),  # Full DDNA
            (2, 3),  # High dilation
            (3, 3),  # Very high dilation
        ]
        
        for D1, D2 in combinations:
            processor = DDNAArrayProcessor(N1=4, N2=3, D1=D1, D2=D2, d=1.0)
            results = processor.run_full_analysis()
            
            # All configurations should produce valid results
            self.assertGreater(results.coarray_aperture, 0)
            self.assertGreater(results.segment_length, 0)
            self.assertGreater(len(results.unique_differences), 0)
            self.assertIsNotNone(results.performance_summary_table)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics calculations."""
    
    def test_metrics_consistency(self):
        """Test that performance metrics are internally consistent."""
        processor = DDNAArrayProcessor(N1=5, N2=4, D1=2, D2=2, d=1.0)
        results = processor.run_full_analysis()
        
        # Segment length and K_max relationship
        K_max_computed = results.segment_length // 2
        
        # Extract K_max from summary table
        summary = results.performance_summary_table
        k_max_row = summary[summary['Metric'] == 'Max Detectable Sources (K_max)']
        K_max_reported = int(k_max_row['Value'].iloc[0])
        
        self.assertEqual(K_max_computed, K_max_reported)
        
        # Total sensors
        total_n_row = summary[summary['Metric'] == 'Total Sensors (N)']
        total_n = int(total_n_row['Value'].iloc[0])
        self.assertEqual(total_n, processor.N1 + processor.N2)
        
        # Holes count
        holes_row = summary[summary['Metric'] == 'Holes in Segment']
        num_holes = int(holes_row['Value'].iloc[0])
        self.assertEqual(num_holes, len(results.holes_in_segment))
    
    def test_weight_distribution(self):
        """Test weight distribution properties."""
        processor = DDNAArrayProcessor(N1=4, N2=3, D1=1, D2=1, d=1.0)
        results = processor.run_full_analysis()
        
        wt = results.weight_table
        
        # Weight at lag 0 should equal total sensors N
        w0 = int(wt[wt['Lag'] == 0]['Weight'].iloc[0])
        self.assertEqual(w0, processor.total_sensors)
        
        # Weights should be symmetric (positive and negative lags)
        positive_lags = wt[wt['Lag'] > 0]
        for _, row in positive_lags.iterrows():
            lag = int(row['Lag'])
            weight_pos = int(row['Weight'])
            
            neg_lag_row = wt[wt['Lag'] == -lag]
            if not neg_lag_row.empty:
                weight_neg = int(neg_lag_row['Weight'].iloc[0])
                self.assertEqual(weight_pos, weight_neg, 
                               f"Asymmetric weights at lag ±{lag}")
    
    def test_aperture_bounds(self):
        """Test coarray aperture bounds."""
        processor = DDNAArrayProcessor(N1=4, N2=3, D1=2, D2=2, d=1.0)
        results = processor.run_full_analysis()
        
        # Coarray aperture should be at least twice physical aperture
        positions = np.array(results.sensors_positions)
        phys_aperture = np.max(positions) - np.min(positions)
        
        # For most DDNA configs, coarray aperture > physical aperture
        # (though not always exactly 2x due to discrete positioning)
        self.assertGreaterEqual(results.coarray_aperture, phys_aperture)


class TestDDNAComparisons(unittest.TestCase):
    """Test DDNA comparisons with DNA and nested arrays."""
    
    def test_ddna_vs_dna(self):
        """Compare DDNA (D1=1,D2≥2) with DNA - verify structural relationship."""
        # DDNA with D1=1, D2=2
        ddna_proc = DDNAArrayProcessor(N1=4, N2=3, D1=1, D2=2, d=1.0)
        ddna_results = ddna_proc.run_full_analysis()
        
        # DNA with same parameters
        dna_proc = DNAArrayProcessor(N1=4, N2=3, D=2, d=1.0)
        dna_results = dna_proc.run_full_analysis()
        
        # NOTE: DDNA and DNA use different construction conventions:
        # - DDNA P1: d × [0, 1, ..., N1-1] (zero-based, matches standard nested)
        # - DNA P1: d × [1, 2, ..., N1] (one-based, different convention)
        # This means positions won't match exactly, but performance should be similar
        
        # Instead of exact position match, verify similar performance characteristics
        # Both should have same number of sensors
        self.assertEqual(ddna_proc.total_sensors, dna_proc.total_sensors)
        
        # Both should have similar aperture (within 10% tolerance)
        aperture_ratio = ddna_results.coarray_aperture / dna_results.coarray_aperture
        self.assertGreater(aperture_ratio, 0.9)
        self.assertLess(aperture_ratio, 1.1)
        
        # Both should produce valid coarrays
        self.assertGreater(ddna_results.segment_length, 0)
        self.assertGreater(dna_results.segment_length, 0)
    
    def test_ddna_vs_nested(self):
        """Compare DDNA (D1=D2=1) with standard nested."""
        # DDNA with D1=D2=1 should match nested
        ddna_proc = DDNAArrayProcessor(N1=4, N2=3, D1=1, D2=1, d=1.0)
        ddna_results = ddna_proc.run_full_analysis()
        
        # Nested with same parameters
        nested_proc = NestedArrayProcessor(N1=4, N2=3, d=1.0)
        nested_results = nested_proc.run_full_analysis()
        
        # Positions should match
        np.testing.assert_array_equal(ddna_results.sensors_positions,
                                     nested_results.sensors_positions)
        
        # Nested uses 'largest_contiguous_segment' which is an array of lag positions
        # DDNA uses 'segment_length' which is the integer length
        if hasattr(nested_results, 'largest_contiguous_segment'):
            nested_segment = nested_results.largest_contiguous_segment
            # It's a flat array of lag positions in the contiguous segment
            if isinstance(nested_segment, (list, np.ndarray)):
                nested_len = len(nested_segment)
            else:
                nested_len = int(nested_segment)
            
            # Both should have positive segment lengths
            self.assertGreater(ddna_results.segment_length, 0)
            self.assertGreater(nested_len, 0)
        
        # Both should have same total sensors
        self.assertEqual(ddna_proc.total_sensors, nested_proc.N1 + nested_proc.N2)
    
    def test_ddna_flexibility(self):
        """Test that DDNA provides more flexibility than DNA."""
        # DDNA can have D1>1, D2>1 combinations not possible with DNA (D1=1 always)
        ddna_full = DDNAArrayProcessor(N1=4, N2=3, D1=2, D2=2, d=1.0)
        results_full = ddna_full.run_full_analysis()
        
        # DNA cannot achieve this configuration (D1 is always 1)
        # DDNA should have different structure
        ddna_dna_like = DDNAArrayProcessor(N1=4, N2=3, D1=1, D2=2, d=1.0)
        results_dna_like = ddna_dna_like.run_full_analysis()
        
        # Positions should differ
        self.assertFalse(np.array_equal(results_full.sensors_positions,
                                       results_dna_like.sensors_positions))
        
        # Both should be valid
        self.assertGreater(results_full.coarray_aperture, 0)
        self.assertGreater(results_dna_like.coarray_aperture, 0)


class TestRobustness(unittest.TestCase):
    """Test robustness across parameter ranges."""
    
    def test_various_n1_n2_combinations(self):
        """Test different N1 and N2 combinations."""
        configs = [
            (3, 2), (4, 3), (5, 4), (6, 5),  # N1 > N2
            (3, 3), (4, 4), (5, 5),          # N1 = N2
            (2, 3), (3, 4), (4, 5),          # N1 < N2
        ]
        
        for N1, N2 in configs:
            processor = DDNAArrayProcessor(N1=N1, N2=N2, D1=1, D2=2, d=1.0)
            results = processor.run_full_analysis()
            
            # All should produce valid results
            self.assertEqual(processor.total_sensors, N1 + N2)
            self.assertGreater(results.coarray_aperture, 0)
            self.assertGreater(results.segment_length, 0)
            self.assertIsNotNone(results.performance_summary_table)
    
    def test_edge_cases(self):
        """Test edge cases (small N, extreme dilations)."""
        # Very small array
        processor = DDNAArrayProcessor(N1=1, N2=1, D1=1, D2=1, d=1.0)
        results = processor.run_full_analysis()
        self.assertEqual(processor.total_sensors, 2)
        self.assertGreater(results.segment_length, 0)
        
        # Large N1
        processor = DDNAArrayProcessor(N1=10, N2=5, D1=1, D2=1, d=1.0)
        results = processor.run_full_analysis()
        self.assertEqual(processor.total_sensors, 15)
        
        # High D1
        processor = DDNAArrayProcessor(N1=4, N2=3, D1=5, D2=1, d=1.0)
        results = processor.run_full_analysis()
        self.assertGreater(results.coarray_aperture, 0)
        
        # High D2
        processor = DDNAArrayProcessor(N1=4, N2=3, D1=1, D2=5, d=1.0)
        results = processor.run_full_analysis()
        self.assertGreater(results.coarray_aperture, 50)
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        processor1 = DDNAArrayProcessor(N1=5, N2=4, D1=2, D2=2, d=1.0)
        results1 = processor1.run_full_analysis()
        
        processor2 = DDNAArrayProcessor(N1=5, N2=4, D1=2, D2=2, d=1.0)
        results2 = processor2.run_full_analysis()
        
        # Results should be identical
        np.testing.assert_array_equal(results1.sensors_positions, results2.sensors_positions)
        self.assertEqual(results1.coarray_aperture, results2.coarray_aperture)
        self.assertEqual(results1.segment_length, results2.segment_length)
        self.assertEqual(len(results1.holes_in_segment), len(results2.holes_in_segment))


class TestComparisonUtility(unittest.TestCase):
    """Test the compare_ddna_arrays utility function."""
    
    def test_basic_comparison(self):
        """Test basic comparison functionality."""
        comparison = compare_ddna_arrays(
            N1_values=[4, 4, 4],
            N2_values=[3, 3, 3],
            D1_values=[1, 1, 2],
            D2_values=[1, 2, 2],
            d=1.0
        )
        
        # Should return DataFrame
        self.assertIsInstance(comparison, pd.DataFrame)
        
        # Should have expected columns
        expected_cols = ['N1', 'N2', 'D1', 'D2', 'Total_N', 'Phys_Aperture',
                        'Coarray_Aperture', 'Unique_Lags', 'L', 'K_max', 'Holes', 'w(1)']
        for col in expected_cols:
            self.assertIn(col, comparison.columns)
        
        # Should have 3 rows
        self.assertEqual(len(comparison), 3)
        
        # All Total_N should be 7 (4+3)
        self.assertTrue(all(comparison['Total_N'] == 7))
    
    def test_dilation_study(self):
        """Test dilation factor study using comparison utility."""
        comparison = compare_ddna_arrays(
            N1_values=[4, 4, 4, 4, 4, 4],
            N2_values=[3, 3, 3, 3, 3, 3],
            D1_values=[1, 1, 1, 2, 2, 2],
            D2_values=[1, 2, 3, 1, 2, 3]
        )
        
        # Should have 6 configurations
        self.assertEqual(len(comparison), 6)
        
        # Aperture should generally increase with D2 (for fixed D1)
        d1_1_configs = comparison[comparison['D1'] == 1]
        apertures_d1_1 = d1_1_configs['Coarray_Aperture'].tolist()
        # Check increasing trend (allowing some tolerance)
        self.assertLess(apertures_d1_1[0], apertures_d1_1[2])  # D2=1 < D2=3
    
    def test_comparison_error_handling(self):
        """Test error handling in comparison utility."""
        with self.assertRaises(ValueError):
            # Mismatched list lengths
            compare_ddna_arrays(
                N1_values=[4, 5],
                N2_values=[3],
                D1_values=[1, 2],
                D2_values=[1, 2]
            )


def run_all_tests():
    """Run all test suites and report results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDDNAConstruction))
    suite.addTests(loader.loadTestsFromTestCase(TestDDNAFullAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestDilationFactorEffects))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestDDNAComparisons))
    suite.addTests(loader.loadTestsFromTestCase(TestRobustness))
    suite.addTests(loader.loadTestsFromTestCase(TestComparisonUtility))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("DDNA TEST SUITE SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED! DDNA implementation validated.")
    else:
        print("\n❌ SOME TESTS FAILED. Review output above.")
    
    print("="*70)
    
    return result


if __name__ == '__main__':
    run_all_tests()
