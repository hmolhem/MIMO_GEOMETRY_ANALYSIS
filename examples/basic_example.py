"""
Basic Example: Analyzing Different Array Configurations

This example demonstrates how to create and analyze different MIMO array types.
"""

import numpy as np
import matplotlib.pyplot as plt
from mimo_geometry import AntennaArray, GeometryAnalyzer, ArrayVisualizer


def main():
    print("=" * 60)
    print("MIMO Geometry Analysis - Basic Example")
    print("=" * 60)
    print()
    
    # Create different array types
    print("Creating array configurations...")
    
    # 1. Uniform Linear Array (ULA)
    ula = AntennaArray.create_ula(num_elements=8, spacing=0.5)
    print(f"✓ Created {ula}")
    
    # 2. Uniform Rectangular Array (URA)
    ura = AntennaArray.create_ura(rows=4, cols=4, spacing=(0.5, 0.5))
    print(f"✓ Created {ura}")
    
    # 3. Uniform Circular Array (UCA)
    uca = AntennaArray.create_uca(num_elements=8, radius=1.0)
    print(f"✓ Created {uca}")
    
    print()
    
    # Analyze each array
    arrays = [("ULA", ula), ("URA", ura), ("UCA", uca)]
    
    for name, array in arrays:
        print(f"\n{name} Analysis:")
        print("-" * 40)
        
        analyzer = GeometryAnalyzer(array)
        analysis = analyzer.analyze_array()
        
        print(f"  Elements: {analysis['num_elements']}")
        print(f"  Aperture: {analysis['aperture']:.3f} wavelengths")
        print(f"  Min spacing: {analysis['min_spacing']:.3f} wavelengths")
        print(f"  Max spacing: {analysis['max_spacing']:.3f} wavelengths")
        print(f"  Mean spacing: {analysis['mean_spacing']:.3f} wavelengths")
        print(f"  Capacity @ 10dB SNR: {analysis['capacity_snr_10dB']:.2f} bits/s/Hz")
    
    print()
    print("=" * 60)
    print("Creating visualizations...")
    print("=" * 60)
    
    # Visualize each array
    for name, array in arrays:
        visualizer = ArrayVisualizer(array)
        
        # Create 2D plot
        fig, ax = visualizer.plot_array_2d(plane='xy')
        plt.savefig(f'example_{name.lower()}_2d.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved example_{name.lower()}_2d.png")
        plt.close()
        
        # Create 3D plot
        fig, ax = visualizer.plot_array_3d()
        plt.savefig(f'example_{name.lower()}_3d.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved example_{name.lower()}_3d.png")
        plt.close()
        
        # Create array factor plot
        fig, axes = visualizer.plot_array_factor()
        plt.savefig(f'example_{name.lower()}_pattern.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved example_{name.lower()}_pattern.png")
        plt.close()
    
    print()
    print("All visualizations saved successfully!")


if __name__ == "__main__":
    main()
