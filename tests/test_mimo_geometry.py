"""
Unit Tests for MIMO Geometry Analysis Package
"""

import unittest
import numpy as np
from mimo_geometry import AntennaArray, GeometryAnalyzer, ArrayVisualizer


class TestAntennaArray(unittest.TestCase):
    """Test cases for AntennaArray class"""
    
    def test_create_ula(self):
        """Test ULA creation"""
        ula = AntennaArray.create_ula(num_elements=8, spacing=0.5)
        self.assertEqual(ula.num_elements, 8)
        self.assertEqual(ula.array_type, "ULA")
        
        # Check that elements are along x-axis
        self.assertTrue(np.allclose(ula.positions[:, 1], 0))
        self.assertTrue(np.allclose(ula.positions[:, 2], 0))
        
        # Check spacing
        for i in range(7):
            dist = np.linalg.norm(ula.positions[i+1] - ula.positions[i])
            self.assertAlmostEqual(dist, 0.5, places=5)
    
    def test_create_ura(self):
        """Test URA creation"""
        ura = AntennaArray.create_ura(rows=3, cols=4, spacing=(0.5, 0.5))
        self.assertEqual(ura.num_elements, 12)
        self.assertEqual(ura.array_type, "URA")
        
        # All elements should be in z=0 plane
        self.assertTrue(np.allclose(ura.positions[:, 2], 0))
    
    def test_create_uca(self):
        """Test UCA creation"""
        uca = AntennaArray.create_uca(num_elements=8, radius=1.0)
        self.assertEqual(uca.num_elements, 8)
        self.assertEqual(uca.array_type, "UCA")
        
        # All elements should be at distance 'radius' from center
        center = uca.center
        for pos in uca.positions:
            dist = np.linalg.norm(pos[:2] - center[:2])  # Distance in xy plane
            self.assertAlmostEqual(dist, 1.0, places=5)
    
    def test_custom_array(self):
        """Test custom array creation"""
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        array = AntennaArray(positions, array_type="custom")
        self.assertEqual(array.num_elements, 3)
        self.assertEqual(array.array_type, "custom")
    
    def test_invalid_positions(self):
        """Test that invalid positions raise an error"""
        with self.assertRaises(ValueError):
            # 2D positions should raise error
            AntennaArray(np.array([[0, 0], [1, 1]]))
    
    def test_center_calculation(self):
        """Test array center calculation"""
        positions = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]])
        array = AntennaArray(positions)
        expected_center = np.array([1, 1, 0])
        np.testing.assert_array_almost_equal(array.center, expected_center)
    
    def test_translate(self):
        """Test array translation"""
        array = AntennaArray.create_ula(num_elements=4, spacing=1.0)
        original_center = array.center.copy()
        
        offset = np.array([1, 2, 3])
        array.translate(offset)
        
        expected_center = original_center + offset
        np.testing.assert_array_almost_equal(array.center, expected_center)
    
    def test_rotate(self):
        """Test array rotation"""
        # Create array along x-axis
        array = AntennaArray.create_ula(num_elements=2, spacing=1.0)
        
        # Rotate 90 degrees around z-axis
        array.rotate(np.pi / 2, axis='z')
        
        # After rotation, should be along y-axis
        # First element at origin should still be at origin (approx)
        self.assertAlmostEqual(np.linalg.norm(array.positions[0] - array.center), 0.5, places=5)
    
    def test_get_distances(self):
        """Test pairwise distance calculation"""
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        array = AntennaArray(positions)
        
        distances = array.get_distances()
        
        # Check diagonal is zero
        np.testing.assert_array_almost_equal(np.diag(distances), np.zeros(3))
        
        # Check specific distances
        self.assertAlmostEqual(distances[0, 1], 1.0)
        self.assertAlmostEqual(distances[0, 2], 1.0)
        self.assertAlmostEqual(distances[1, 2], np.sqrt(2), places=5)
    
    def test_get_angles(self):
        """Test angle calculation"""
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        array = AntennaArray(positions)
        
        angles = array.get_angles(reference_idx=0)
        
        # Angle to element 1 (along x-axis) should be 0
        self.assertAlmostEqual(angles[1], 0.0)
        
        # Angle to element 2 (along y-axis) should be pi/2
        self.assertAlmostEqual(angles[2], np.pi / 2, places=5)


class TestGeometryAnalyzer(unittest.TestCase):
    """Test cases for GeometryAnalyzer class"""
    
    def test_compute_array_aperture(self):
        """Test aperture calculation"""
        array = AntennaArray.create_ula(num_elements=5, spacing=1.0)
        analyzer = GeometryAnalyzer(array)
        
        aperture = analyzer.compute_array_aperture()
        
        # For 5 elements with spacing 1.0, aperture should be 4.0
        self.assertAlmostEqual(aperture, 4.0, places=5)
    
    def test_compute_element_spacing_stats(self):
        """Test spacing statistics"""
        array = AntennaArray.create_ula(num_elements=5, spacing=1.0)
        analyzer = GeometryAnalyzer(array)
        
        stats = analyzer.compute_element_spacing_stats()
        
        # Min spacing should be 1.0 (adjacent elements)
        self.assertAlmostEqual(stats['min_spacing'], 1.0, places=5)
        
        # Max spacing should be 4.0 (first to last)
        self.assertAlmostEqual(stats['max_spacing'], 4.0, places=5)
    
    def test_compute_spatial_correlation(self):
        """Test spatial correlation matrix"""
        array = AntennaArray.create_ula(num_elements=4, spacing=0.5)
        analyzer = GeometryAnalyzer(array)
        
        correlation = analyzer.compute_spatial_correlation(angle_spread=10.0)
        
        # Diagonal should be 1 (perfect self-correlation)
        np.testing.assert_array_almost_equal(
            np.abs(np.diag(correlation)), 
            np.ones(4)
        )
        
        # Matrix should be Hermitian
        np.testing.assert_array_almost_equal(
            correlation, 
            correlation.conj().T
        )
    
    def test_estimate_channel_capacity(self):
        """Test channel capacity estimation"""
        array = AntennaArray.create_ula(num_elements=4, spacing=0.5)
        analyzer = GeometryAnalyzer(array)
        
        capacity = analyzer.estimate_channel_capacity(snr_db=10.0, angle_spread=10.0)
        
        # Capacity should be positive
        self.assertGreater(capacity, 0)
        
        # Higher SNR should give higher capacity
        capacity_high_snr = analyzer.estimate_channel_capacity(snr_db=20.0, angle_spread=10.0)
        self.assertGreater(capacity_high_snr, capacity)
    
    def test_compute_array_factor(self):
        """Test array factor computation"""
        array = AntennaArray.create_ula(num_elements=4, spacing=0.5)
        analyzer = GeometryAnalyzer(array)
        
        angles = np.array([0, 45, 90, 180])
        af = analyzer.compute_array_factor(angles)
        
        # Array factor should be normalized (max = 1)
        self.assertAlmostEqual(np.max(af), 1.0, places=5)
        
        # All values should be non-negative
        self.assertTrue(np.all(af >= 0))
    
    def test_analyze_array(self):
        """Test comprehensive array analysis"""
        array = AntennaArray.create_ula(num_elements=8, spacing=0.5)
        analyzer = GeometryAnalyzer(array)
        
        analysis = analyzer.analyze_array()
        
        # Check that all expected keys are present
        expected_keys = [
            'array_type', 'num_elements', 'array_center', 'aperture',
            'min_spacing', 'max_spacing', 'mean_spacing', 'std_spacing',
            'capacity_snr_0dB', 'capacity_snr_10dB', 'capacity_snr_20dB', 'capacity_snr_30dB'
        ]
        
        for key in expected_keys:
            self.assertIn(key, analysis)


class TestArrayVisualizer(unittest.TestCase):
    """Test cases for ArrayVisualizer class"""
    
    def test_visualizer_creation(self):
        """Test visualizer initialization"""
        array = AntennaArray.create_ula(num_elements=4, spacing=0.5)
        visualizer = ArrayVisualizer(array)
        
        self.assertIsNotNone(visualizer.array)
        self.assertIsNotNone(visualizer.analyzer)
    
    def test_plot_array_2d(self):
        """Test 2D plotting"""
        array = AntennaArray.create_ula(num_elements=4, spacing=0.5)
        visualizer = ArrayVisualizer(array)
        
        # Test different planes
        for plane in ['xy', 'xz', 'yz']:
            fig, ax = visualizer.plot_array_2d(plane=plane)
            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
            import matplotlib.pyplot as plt
            plt.close(fig)
    
    def test_plot_array_3d(self):
        """Test 3D plotting"""
        array = AntennaArray.create_uca(num_elements=8, radius=1.0)
        visualizer = ArrayVisualizer(array)
        
        fig, ax = visualizer.plot_array_3d()
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_invalid_plane(self):
        """Test that invalid plane raises error"""
        array = AntennaArray.create_ula(num_elements=4, spacing=0.5)
        visualizer = ArrayVisualizer(array)
        
        with self.assertRaises(ValueError):
            visualizer.plot_array_2d(plane='invalid')


if __name__ == '__main__':
    unittest.main()
