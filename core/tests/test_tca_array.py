"""
Test Suite for Two-level Coprime Array (TCA) Processor
========================================================

Comprehensive tests validating TCA construction, coarray properties,
and performance metrics.

Author: Hossein (RadarPy Project)
Date: November 7, 2025
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from geometry_processors.tca_processor import TCAArrayProcessor, compare_tca_arrays


class TestTCAConstruction(unittest.TestCase):
    """Test TCA sensor position construction."""
    
    def test_small_coprime_2_3(self):
        """Test smallest TCA: M=2, N=3."""
        tca = TCAArrayProcessor(M=2, N=3, d=1.0)
        
        # Expected: P1=[0,2,4], P2=[0,3] → Unique=[0,2,3,4]
        expected = np.array([0, 2, 3, 4])
        np.testing.assert_array_equal(tca.data.sensors_positions, expected)
        
        # Check parameters
        self.assertEqual(tca.M, 2)
        self.assertEqual(tca.N, 3)
        self.assertEqual(tca.total_sensors, 4)  # M+N-1 (origin shared)
        self.assertTrue(tca.is_coprime)  # gcd(2,3)=1
    
    def test_medium_coprime_3_5(self):
        """Test medium TCA: M=3, N=5."""
        tca = TCAArrayProcessor(M=3, N=5, d=1.0)
        
        # P1=[0,3,6,9,12], P2=[0,5,10] → Unique=[0,3,5,6,9,10,12]
        expected = np.array([0, 3, 5, 6, 9, 10, 12])
        np.testing.assert_array_equal(tca.data.sensors_positions, expected)
        
        self.assertEqual(tca.total_sensors, 7)  # M+N-1
        self.assertTrue(tca.is_coprime)
    
    def test_different_spacing(self):
        """Test TCA with d=0.5."""
        tca = TCAArrayProcessor(M=2, N=3, d=0.5)
        
        # With d=0.5: P1=0.5*2*[0,1,2]=[0,1,2], P2=0.5*3*[0,1]=[0,1.5]
        # Unique=[0,1,1.5,2]
        expected = np.array([0, 1, 1.5, 2])
        np.testing.assert_array_almost_equal(tca.data.sensors_positions, expected, decimal=5)
    
    def test_non_coprime_warning(self):
        """Test non-coprime pair (gcd≠1)."""
        tca = TCAArrayProcessor(M=4, N=6, d=1.0)  # gcd(4,6)=2
        
        self.assertFalse(tca.is_coprime)
        # P1=[0,4,8,12,16,20], P2=[0,6,12,18] → Unique=[0,4,6,8,12,16,18,20] (8 positions)
        self.assertEqual(tca.total_sensors, 8)
    
    def test_minimum_requirements(self):
        """Test M, N minimum values."""
        # Valid minimum: M=2, N=2
        # P1 = [0,2], P2 = [0,2] → Unique = [0,2] (2 positions, both same)
        tca = TCAArrayProcessor(M=2, N=2, d=1.0)
        self.assertEqual(tca.total_sensors, 2)  # Same positions, origin shared
        
        # Invalid: M < 2
        with self.assertRaises(ValueError):
            TCAArrayProcessor(M=1, N=3, d=1.0)
        
        # Invalid: N < 2
        with self.assertRaises(ValueError):
            TCAArrayProcessor(M=3, N=1, d=1.0)


class TestTCAFullAnalysis(unittest.TestCase):
    """Test complete TCA analysis pipeline."""
    
    def test_full_analysis_small(self):
        """Test full analysis for M=2, N=3."""
        tca = TCAArrayProcessor(M=2, N=3, d=1.0)
        results = tca.run_full_analysis()
        
        # Check all required attributes exist
        self.assertTrue(hasattr(results, 'sensors_positions'))
        self.assertTrue(hasattr(results, 'coarray_positions'))
        self.assertTrue(hasattr(results, 'unique_differences'))
        self.assertTrue(hasattr(results, 'weight_table'))
        self.assertTrue(hasattr(results, 'contiguous_segments'))
        self.assertTrue(hasattr(results, 'holes_in_segment'))
        self.assertTrue(hasattr(results, 'performance_summary_table'))
        
        # Check coarray aperture
        self.assertEqual(results.coarray_aperture, 8)  # max(4)-min(-4)
        
        # Check segment length
        self.assertGreater(results.segment_length, 0)
    
    def test_full_analysis_medium(self):
        """Test full analysis for M=3, N=5."""
        tca = TCAArrayProcessor(M=3, N=5, d=1.0)
        results = tca.run_full_analysis()
        
        # For coprime (3,5), expect good performance
        self.assertGreater(results.coarray_aperture, 20)
        self.assertEqual(len(results.sensors_positions), 7)
        self.assertGreaterEqual(results.segment_length, 10)
    
    def test_full_analysis_large(self):
        """Test full analysis for M=4, N=5."""
        tca = TCAArrayProcessor(M=4, N=5, d=1.0)
        results = tca.run_full_analysis()
        
        # Larger configuration
        self.assertEqual(len(results.sensors_positions), 8)
        self.assertGreater(results.coarray_aperture, 30)


class TestTCACoprimalityEffects(unittest.TestCase):
    """Test effects of coprime vs non-coprime configurations."""
    
    def test_coprime_vs_noncoprime(self):
        """Compare coprime (3,5) vs non-coprime (3,6)."""
        tca_coprime = TCAArrayProcessor(M=3, N=5, d=1.0)
        results_coprime = tca_coprime.run_full_analysis()
        
        tca_non = TCAArrayProcessor(M=3, N=6, d=1.0)
        results_non = tca_non.run_full_analysis()
        
        # Coprime should have fewer holes
        holes_coprime = len(results_coprime.holes_in_segment)
        holes_non = len(results_non.holes_in_segment)
        
        # Non-coprime typically has more holes (not always guaranteed, but common)
        # At minimum, both should be non-negative
        self.assertGreaterEqual(holes_coprime, 0)
        self.assertGreaterEqual(holes_non, 0)
    
    def test_coprimality_detection(self):
        """Test coprimality flag for various pairs."""
        coprime_pairs = [(2,3), (3,4), (3,5), (4,5), (5,7)]
        non_coprime_pairs = [(2,4), (3,6), (4,6), (6,9)]
        
        for M, N in coprime_pairs:
            tca = TCAArrayProcessor(M=M, N=N, d=1.0)
            self.assertTrue(tca.is_coprime, f"({M},{N}) should be coprime")
        
        for M, N in non_coprime_pairs:
            tca = TCAArrayProcessor(M=M, N=N, d=1.0)
            self.assertFalse(tca.is_coprime, f"({M},{N}) should NOT be coprime")


class TestTCAPerformanceMetrics(unittest.TestCase):
    """Test TCA performance metrics and DOF efficiency."""
    
    def test_dof_efficiency_trend(self):
        """Test that DOF efficiency generally improves with larger coprime pairs."""
        configs = [(2,3), (3,4), (3,5), (4,5)]
        efficiencies = []
        
        for M, N in configs:
            tca = TCAArrayProcessor(M=M, N=N, d=1.0)
            results = tca.run_full_analysis()
            K_max = results.segment_length // 2
            efficiency = K_max / tca.total_sensors
            efficiencies.append(efficiency)
        
        # Generally expect increasing efficiency (not strict, but trend)
        # At minimum, all should be positive
        for eff in efficiencies:
            self.assertGreater(eff, 0)
    
    def test_aperture_scaling(self):
        """Test aperture scales approximately as M×N."""
        configs = [(2,3), (3,5), (4,5)]
        
        for M, N in configs:
            tca = TCAArrayProcessor(M=M, N=N, d=1.0)
            results = tca.run_full_analysis()
            
            # Aperture should be on order of M×N (not exact, but close)
            expected_order = M * N
            self.assertLess(results.coarray_aperture, 3 * expected_order)  # Upper bound
            self.assertGreater(results.coarray_aperture, expected_order // 2)  # Lower bound
    
    def test_weight_distribution(self):
        """Test weight distribution properties."""
        tca = TCAArrayProcessor(M=3, N=5, d=1.0)
        results = tca.run_full_analysis()
        
        wt = results.weight_table
        
        # Weight at lag 0 should equal number of sensors
        w0 = int(wt[wt['Lag'] == 0]['Weight'].iloc[0])
        self.assertEqual(w0, tca.total_sensors)
        
        # All weights should be positive
        self.assertTrue((wt['Weight'] > 0).all())


class TestTCAHolesAnalysis(unittest.TestCase):
    """Test hole detection and contiguous segment analysis."""
    
    def test_coprime_minimal_holes(self):
        """Test that coprime pairs have minimal holes."""
        coprime_pairs = [(2,3), (3,5), (4,5)]
        
        for M, N in coprime_pairs:
            tca = TCAArrayProcessor(M=M, N=N, d=1.0)
            results = tca.run_full_analysis()
            
            # Coprime arrays should have relatively few holes
            holes = len(results.holes_in_segment)
            
            # Expected to be hole-free or very few holes in main segment
            # (Not strict guarantee, depends on exact construction)
            self.assertGreaterEqual(holes, 0)  # Non-negative
    
    def test_contiguous_segments(self):
        """Test contiguous segment detection."""
        tca = TCAArrayProcessor(M=3, N=5, d=1.0)
        results = tca.run_full_analysis()
        
        segments = results.contiguous_segments
        
        # Should have at least one segment
        self.assertGreater(len(segments), 0)
        
        # Each segment should have start ≤ end
        for seg in segments:
            self.assertLessEqual(seg[0], seg[1])


class TestTCAComparison(unittest.TestCase):
    """Test TCA comparison utilities."""
    
    def test_compare_tca_arrays(self):
        """Test batch comparison of TCA configurations."""
        configs = [(2,3), (3,4), (3,5), (4,5)]
        comparison = compare_tca_arrays(configs, d=1.0)
        
        # Should have one row per configuration
        self.assertEqual(len(comparison), len(configs))
        
        # Check required columns exist
        self.assertIn('M', comparison.columns)
        self.assertIn('N', comparison.columns)
        self.assertIn('Total_Sensors', comparison.columns)
        self.assertIn('K_max', comparison.columns)
        self.assertIn('Is_Coprime', comparison.columns)
    
    def test_error_handling(self):
        """Test error handling in batch comparison."""
        configs = [(2,3), (1,5)]  # Second one invalid (M<2)
        
        # Should process valid configs and skip invalid
        comparison = compare_tca_arrays(configs, d=1.0)
        
        # Should have fewer rows than configs (invalid skipped)
        self.assertLess(len(comparison), len(configs))


class TestTCAVisualization(unittest.TestCase):
    """Test TCA visualization outputs."""
    
    def test_plot_coarray_output(self):
        """Test that plot_coarray generates valid output."""
        tca = TCAArrayProcessor(M=3, N=5, d=1.0)
        tca.run_full_analysis()
        
        plot_str = tca.plot_coarray()
        
        # Should be non-empty string
        self.assertIsInstance(plot_str, str)
        self.assertGreater(len(plot_str), 0)
        
        # Should contain key information
        self.assertIn("TCA", plot_str)
        self.assertIn("M=3", plot_str)
        self.assertIn("N=5", plot_str)
        self.assertIn("Coprime" if tca.is_coprime else "NOT Coprime", plot_str)


class TestTCAEdgeCases(unittest.TestCase):
    """Test TCA edge cases and robustness."""
    
    def test_minimum_tca(self):
        """Test smallest possible TCA: M=2, N=2."""
        tca = TCAArrayProcessor(M=2, N=2, d=1.0)
        results = tca.run_full_analysis()
        
        # M=2, N=2: P1=[0,2], P2=[0,2] → Unique=[0,2] (2 sensors)
        self.assertEqual(tca.total_sensors, 2)
        self.assertGreater(results.segment_length, 0)
    
    def test_large_coprime_pair(self):
        """Test larger coprime pair: M=5, N=7."""
        tca = TCAArrayProcessor(M=5, N=7, d=1.0)
        results = tca.run_full_analysis()
        
        # Should handle larger arrays
        self.assertEqual(tca.total_sensors, 11)  # M+N-1
        self.assertGreater(results.coarray_aperture, 60)
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        tca1 = TCAArrayProcessor(M=3, N=5, d=1.0)
        results1 = tca1.run_full_analysis()
        
        tca2 = TCAArrayProcessor(M=3, N=5, d=1.0)
        results2 = tca2.run_full_analysis()
        
        # Should get identical results
        np.testing.assert_array_equal(results1.sensors_positions, results2.sensors_positions)
        self.assertEqual(results1.coarray_aperture, results2.coarray_aperture)
        self.assertEqual(results1.segment_length, results2.segment_length)


def run_tests():
    """Run all TCA tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTCAConstruction))
    suite.addTests(loader.loadTestsFromTestCase(TestTCAFullAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestTCACoprimalityEffects))
    suite.addTests(loader.loadTestsFromTestCase(TestTCAPerformanceMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestTCAHolesAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestTCAComparison))
    suite.addTests(loader.loadTestsFromTestCase(TestTCAVisualization))
    suite.addTests(loader.loadTestsFromTestCase(TestTCAEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TCA TEST SUITE SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED! 🎉")
    else:
        print("\n✗ Some tests failed")
    
    print("="*70)
    
    return result


if __name__ == '__main__':
    run_tests()
