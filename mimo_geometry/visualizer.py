"""
Visualizer Module

Provides visualization tools for MIMO antenna arrays.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional
from .antenna_array import AntennaArray
from .geometry_analyzer import GeometryAnalyzer


class ArrayVisualizer:
    """
    Visualizes MIMO antenna array configurations and analysis results.
    """
    
    def __init__(self, array: AntennaArray):
        """
        Initialize the visualizer.
        
        Args:
            array: AntennaArray to visualize
        """
        self.array = array
        self.analyzer = GeometryAnalyzer(array)
    
    def plot_array_2d(self, 
                      plane: str = 'xy',
                      show_labels: bool = True,
                      figsize: tuple = (8, 6)):
        """
        Plot 2D projection of the antenna array.
        
        Args:
            plane: Plane to project onto ('xy', 'xz', or 'yz')
            show_labels: Whether to show antenna labels
            figsize: Figure size
            
        Returns:
            matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        positions = self.array.positions
        
        if plane == 'xy':
            x, y = positions[:, 0], positions[:, 1]
            xlabel, ylabel = 'X (wavelengths)', 'Y (wavelengths)'
        elif plane == 'xz':
            x, y = positions[:, 0], positions[:, 2]
            xlabel, ylabel = 'X (wavelengths)', 'Z (wavelengths)'
        elif plane == 'yz':
            x, y = positions[:, 1], positions[:, 2]
            xlabel, ylabel = 'Y (wavelengths)', 'Z (wavelengths)'
        else:
            raise ValueError("Plane must be 'xy', 'xz', or 'yz'")
        
        ax.scatter(x, y, c='blue', s=100, marker='o', edgecolors='black', linewidth=2)
        
        if show_labels:
            for i, (xi, yi) in enumerate(zip(x, y)):
                ax.annotate(f'{i}', (xi, yi), xytext=(5, 5), 
                           textcoords='offset points', fontsize=10)
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{self.array.array_type} Array - {plane.upper()} Plane', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_array_3d(self, 
                      show_labels: bool = True,
                      figsize: tuple = (10, 8)):
        """
        Plot 3D visualization of the antenna array.
        
        Args:
            show_labels: Whether to show antenna labels
            figsize: Figure size
            
        Returns:
            matplotlib figure and axis objects
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        positions = self.array.positions
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        
        ax.scatter(x, y, z, c='blue', s=100, marker='o', 
                  edgecolors='black', linewidth=2, depthshade=True)
        
        if show_labels:
            for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
                ax.text(xi, yi, zi, f'  {i}', fontsize=10)
        
        ax.set_xlabel('X (wavelengths)', fontsize=12)
        ax.set_ylabel('Y (wavelengths)', fontsize=12)
        ax.set_zlabel('Z (wavelengths)', fontsize=12)
        ax.set_title(f'{self.array.array_type} Array - 3D View', fontsize=14)
        
        # Set equal aspect ratio
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max()+x.min()) * 0.5
        mid_y = (y.max()+y.min()) * 0.5
        mid_z = (z.max()+z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_array_factor(self, 
                         elevation: float = 0.0,
                         wavelength: float = 1.0,
                         figsize: tuple = (10, 6)):
        """
        Plot the array factor pattern.
        
        Args:
            elevation: Elevation angle in degrees
            wavelength: Signal wavelength
            figsize: Figure size
            
        Returns:
            matplotlib figure and axis objects
        """
        azimuth_angles = np.linspace(-180, 180, 360)
        array_factor = self.analyzer.compute_array_factor(
            azimuth_angles, elevation, wavelength
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Linear plot
        ax1.plot(azimuth_angles, array_factor, 'b-', linewidth=2)
        ax1.set_xlabel('Azimuth Angle (degrees)', fontsize=12)
        ax1.set_ylabel('Normalized Array Factor', fontsize=12)
        ax1.set_title('Array Factor (Linear)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(-180, 180)
        
        # Polar plot
        ax2 = plt.subplot(122, projection='polar')
        theta = np.deg2rad(azimuth_angles)
        ax2.plot(theta, array_factor, 'b-', linewidth=2)
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        ax2.set_title('Array Factor (Polar)', fontsize=14, pad=20)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_spatial_correlation(self, 
                                angle_spread: float = 10.0,
                                figsize: tuple = (8, 6)):
        """
        Plot the spatial correlation matrix.
        
        Args:
            angle_spread: Angular spread in degrees
            figsize: Figure size
            
        Returns:
            matplotlib figure and axis objects
        """
        correlation = self.analyzer.compute_spatial_correlation(angle_spread)
        correlation_mag = np.abs(correlation)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(correlation_mag, cmap='hot', interpolation='nearest')
        ax.set_xlabel('Antenna Index', fontsize=12)
        ax.set_ylabel('Antenna Index', fontsize=12)
        ax.set_title(f'Spatial Correlation (Angle Spread: {angle_spread}°)', fontsize=14)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Magnitude', fontsize=12)
        
        # Add text annotations
        for i in range(correlation_mag.shape[0]):
            for j in range(correlation_mag.shape[1]):
                text = ax.text(j, i, f'{correlation_mag[i, j]:.2f}',
                             ha="center", va="center", color="white" if correlation_mag[i, j] > 0.5 else "black",
                             fontsize=8)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_capacity_vs_snr(self,
                            snr_range: tuple = (-10, 30),
                            angle_spread: float = 10.0,
                            figsize: tuple = (10, 6)):
        """
        Plot channel capacity vs SNR.
        
        Args:
            snr_range: Range of SNR values (min, max) in dB
            angle_spread: Angular spread in degrees
            figsize: Figure size
            
        Returns:
            matplotlib figure and axis objects
        """
        snr_values = np.linspace(snr_range[0], snr_range[1], 50)
        capacities = [self.analyzer.estimate_channel_capacity(snr, angle_spread) 
                     for snr in snr_values]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(snr_values, capacities, 'b-', linewidth=2, label=f'{self.array.array_type}')
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Capacity (bits/s/Hz)', fontsize=12)
        ax.set_title(f'Channel Capacity vs SNR\n({self.array.num_elements} elements, {angle_spread}° spread)', 
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        return fig, ax
    
    def create_summary_plot(self, figsize: tuple = (15, 10)):
        """
        Create a comprehensive summary plot with multiple subplots.
        
        Args:
            figsize: Figure size
            
        Returns:
            matplotlib figure object
        """
        fig = plt.figure(figsize=figsize)
        
        # 2D array plot
        ax1 = plt.subplot(2, 3, 1)
        positions = self.array.positions
        ax1.scatter(positions[:, 0], positions[:, 1], c='blue', s=100, 
                   marker='o', edgecolors='black', linewidth=2)
        ax1.set_xlabel('X (wavelengths)')
        ax1.set_ylabel('Y (wavelengths)')
        ax1.set_title(f'{self.array.array_type} Array (XY Plane)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 3D array plot
        ax2 = plt.subplot(2, 3, 2, projection='3d')
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        ax2.scatter(x, y, z, c='blue', s=100, marker='o', 
                   edgecolors='black', linewidth=2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('3D View')
        
        # Array factor
        ax3 = plt.subplot(2, 3, 3, projection='polar')
        azimuth_angles = np.linspace(-180, 180, 360)
        array_factor = self.analyzer.compute_array_factor(azimuth_angles)
        theta = np.deg2rad(azimuth_angles)
        ax3.plot(theta, array_factor, 'b-', linewidth=2)
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_title('Array Factor')
        
        # Spatial correlation
        ax4 = plt.subplot(2, 3, 4)
        correlation = self.analyzer.compute_spatial_correlation()
        im = ax4.imshow(np.abs(correlation), cmap='hot', interpolation='nearest')
        ax4.set_xlabel('Antenna Index')
        ax4.set_ylabel('Antenna Index')
        ax4.set_title('Spatial Correlation')
        plt.colorbar(im, ax=ax4)
        
        # Capacity vs SNR
        ax5 = plt.subplot(2, 3, 5)
        snr_values = np.linspace(-10, 30, 50)
        capacities = [self.analyzer.estimate_channel_capacity(snr) for snr in snr_values]
        ax5.plot(snr_values, capacities, 'b-', linewidth=2)
        ax5.set_xlabel('SNR (dB)')
        ax5.set_ylabel('Capacity (bits/s/Hz)')
        ax5.set_title('Channel Capacity')
        ax5.grid(True, alpha=0.3)
        
        # Analysis summary (text)
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        analysis = self.analyzer.analyze_array()
        summary_text = f"""
        Array Analysis Summary
        ----------------------
        Type: {analysis['array_type']}
        Elements: {analysis['num_elements']}
        Aperture: {analysis['aperture']:.3f} λ
        
        Spacing Statistics:
        Min: {analysis['min_spacing']:.3f} λ
        Max: {analysis['max_spacing']:.3f} λ
        Mean: {analysis['mean_spacing']:.3f} λ
        Std: {analysis['std_spacing']:.3f} λ
        
        Capacity (10dB SNR):
        {analysis['capacity_snr_10dB']:.2f} bits/s/Hz
        """
        ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        return fig
