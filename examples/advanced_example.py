"""
Advanced Example: Custom Array Configuration and Analysis

This example demonstrates creating custom arrays and performing detailed analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from mimo_geometry import AntennaArray, GeometryAnalyzer, ArrayVisualizer


def main():
    print("=" * 60)
    print("MIMO Geometry Analysis - Advanced Example")
    print("=" * 60)
    print()
    
    # Create a custom L-shaped array
    print("Creating custom L-shaped array...")
    positions = []
    
    # Horizontal arm
    for i in range(5):
        positions.append([i * 0.5, 0, 0])
    
    # Vertical arm (excluding origin to avoid duplication)
    for i in range(1, 5):
        positions.append([0, i * 0.5, 0])
    
    custom_array = AntennaArray(np.array(positions), array_type="L-shaped")
    print(f"✓ Created {custom_array}")
    print()
    
    # Detailed analysis
    analyzer = GeometryAnalyzer(custom_array)
    
    print("Performing detailed analysis...")
    print("-" * 60)
    
    # Basic geometry
    print(f"\nGeometry Properties:")
    print(f"  Number of elements: {custom_array.num_elements}")
    print(f"  Array center: {custom_array.center}")
    print(f"  Aperture: {analyzer.compute_array_aperture():.3f} wavelengths")
    
    # Spacing statistics
    spacing_stats = analyzer.compute_element_spacing_stats()
    print(f"\nSpacing Statistics:")
    print(f"  Min: {spacing_stats['min_spacing']:.3f} wavelengths")
    print(f"  Max: {spacing_stats['max_spacing']:.3f} wavelengths")
    print(f"  Mean: {spacing_stats['mean_spacing']:.3f} wavelengths")
    print(f"  Std: {spacing_stats['std_spacing']:.3f} wavelengths")
    
    # Capacity analysis at different SNRs and angle spreads
    print(f"\nChannel Capacity Analysis:")
    for snr in [0, 10, 20, 30]:
        for spread in [5, 10, 20]:
            capacity = analyzer.estimate_channel_capacity(snr, spread)
            print(f"  SNR={snr:2d}dB, Spread={spread:2d}°: {capacity:.2f} bits/s/Hz")
    
    print()
    print("=" * 60)
    print("Creating comprehensive visualizations...")
    print("=" * 60)
    
    # Create visualizer
    visualizer = ArrayVisualizer(custom_array)
    
    # Summary plot
    fig = visualizer.create_summary_plot(figsize=(15, 10))
    plt.savefig('example_custom_summary.png', dpi=150, bbox_inches='tight')
    print("✓ Saved example_custom_summary.png")
    plt.close()
    
    # Spatial correlation heatmap
    fig, ax = visualizer.plot_spatial_correlation(angle_spread=10.0)
    plt.savefig('example_custom_correlation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved example_custom_correlation.png")
    plt.close()
    
    # Capacity vs SNR
    fig, ax = visualizer.plot_capacity_vs_snr(snr_range=(-10, 30), angle_spread=10.0)
    plt.savefig('example_custom_capacity.png', dpi=150, bbox_inches='tight')
    print("✓ Saved example_custom_capacity.png")
    plt.close()
    
    print()
    print("Analysis complete!")
    
    # Demonstrate array transformations
    print()
    print("=" * 60)
    print("Demonstrating array transformations...")
    print("=" * 60)
    
    # Create a simple ULA for transformation demo
    test_array = AntennaArray.create_ula(num_elements=4, spacing=0.5)
    print(f"\nOriginal array center: {test_array.center}")
    
    # Translate
    test_array.translate(np.array([1.0, 2.0, 0.5]))
    print(f"After translation: {test_array.center}")
    
    # Rotate
    test_array.rotate(np.pi / 4, axis='z')
    print(f"After rotation (45° around Z): {test_array.center}")
    
    print()
    print("Transformation demo complete!")


if __name__ == "__main__":
    main()
