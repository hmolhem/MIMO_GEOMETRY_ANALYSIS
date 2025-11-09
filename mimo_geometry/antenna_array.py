"""
Antenna Array Module

Defines antenna array configurations and positions for MIMO systems.
"""

import numpy as np
from typing import List, Tuple, Optional


class AntennaArray:
    """
    Represents a MIMO antenna array with configurable geometry.
    
    Attributes:
        positions: numpy array of antenna positions (N x 3) in meters
        array_type: Type of array configuration (ULA, URA, UCA, custom)
    """
    
    def __init__(self, positions: np.ndarray = None, array_type: str = "custom"):
        """
        Initialize antenna array.
        
        Args:
            positions: Array of antenna positions (N x 3), where N is number of antennas
            array_type: Type of array configuration
        """
        if positions is None:
            positions = np.array([[0, 0, 0]])
        
        self.positions = np.atleast_2d(positions)
        self.array_type = array_type
        
        if self.positions.shape[1] != 3:
            raise ValueError("Positions must be N x 3 array (x, y, z coordinates)")
    
    @classmethod
    def create_ula(cls, num_elements: int, spacing: float = 0.5):
        """
        Create a Uniform Linear Array (ULA).
        
        Args:
            num_elements: Number of antenna elements
            spacing: Distance between adjacent elements (in wavelengths)
            
        Returns:
            AntennaArray: ULA configuration
        """
        positions = np.zeros((num_elements, 3))
        positions[:, 0] = np.arange(num_elements) * spacing
        return cls(positions, "ULA")
    
    @classmethod
    def create_ura(cls, rows: int, cols: int, spacing: Tuple[float, float] = (0.5, 0.5)):
        """
        Create a Uniform Rectangular Array (URA).
        
        Args:
            rows: Number of rows
            cols: Number of columns
            spacing: Distance between elements (x_spacing, y_spacing) in wavelengths
            
        Returns:
            AntennaArray: URA configuration
        """
        positions = []
        for i in range(rows):
            for j in range(cols):
                positions.append([j * spacing[0], i * spacing[1], 0])
        
        return cls(np.array(positions), "URA")
    
    @classmethod
    def create_uca(cls, num_elements: int, radius: float = 1.0):
        """
        Create a Uniform Circular Array (UCA).
        
        Args:
            num_elements: Number of antenna elements
            radius: Radius of the circular array (in wavelengths)
            
        Returns:
            AntennaArray: UCA configuration
        """
        angles = np.linspace(0, 2 * np.pi, num_elements, endpoint=False)
        positions = np.zeros((num_elements, 3))
        positions[:, 0] = radius * np.cos(angles)
        positions[:, 1] = radius * np.sin(angles)
        
        return cls(positions, "UCA")
    
    @property
    def num_elements(self) -> int:
        """Get the number of antenna elements."""
        return len(self.positions)
    
    @property
    def center(self) -> np.ndarray:
        """Get the geometric center of the array."""
        return np.mean(self.positions, axis=0)
    
    def get_distances(self) -> np.ndarray:
        """
        Calculate pairwise distances between all antenna elements.
        
        Returns:
            Distance matrix (N x N)
        """
        N = self.num_elements
        distances = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                distances[i, j] = np.linalg.norm(
                    self.positions[i] - self.positions[j]
                )
        
        return distances
    
    def get_angles(self, reference_idx: int = 0) -> np.ndarray:
        """
        Calculate angles from a reference antenna to all other antennas.
        
        Args:
            reference_idx: Index of reference antenna
            
        Returns:
            Array of angles in radians
        """
        ref_pos = self.positions[reference_idx]
        relative_pos = self.positions - ref_pos
        
        angles = np.arctan2(relative_pos[:, 1], relative_pos[:, 0])
        return angles
    
    def translate(self, offset: np.ndarray):
        """
        Translate the array by an offset.
        
        Args:
            offset: Translation vector (3D)
        """
        self.positions += offset
    
    def rotate(self, angle: float, axis: str = 'z'):
        """
        Rotate the array around a specified axis.
        
        Args:
            angle: Rotation angle in radians
            axis: Rotation axis ('x', 'y', or 'z')
        """
        if axis == 'z':
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
        elif axis == 'y':
            rotation_matrix = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        elif axis == 'x':
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'")
        
        center = self.center
        centered_positions = self.positions - center
        rotated_positions = centered_positions @ rotation_matrix.T
        self.positions = rotated_positions + center
    
    def __repr__(self) -> str:
        return f"AntennaArray(type={self.array_type}, elements={self.num_elements})"
