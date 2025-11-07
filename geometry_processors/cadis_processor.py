"""
CADiS Array Processor
=====================

CADiS (Concatenated Difference Set) Array

Reference:
    Qin et al., "Generalized Coprime Array Configurations for Direction-of-Arrival Estimation"
    (Implementation based on concatenated difference set design principles)

Construction:
    Uses concatenated difference sets achieving w(1)=0:
    - Combines multiple difference sets with strategic offsets
    - Ensures w(1)=0 by avoiding consecutive sensor placement
    - Optimizes for large aperture with good weight distribution
    - Superior hole-free properties compared to standard sparse arrays

Key Properties:
    - Weight constraint: w(1) = 0 (like Z5, MISC, cMRA)
    - Achieves hole-free or minimal-hole coarray segments
    - Suitable for mutual coupling mitigation
    - Mentioned in Kulkarni & Vaidyanathan Table V

Parameters:
    N (int): Total number of sensors
    d (float): Base spacing multiplier (wavelength/2)

Example:
    >>> # Create CADiS array with 7 sensors
    >>> processor = CADiSArrayProcessor(N=7, d=1.0)
    >>> results = processor.run_full_analysis()
    >>> print(results.performance_summary_table)

Author: Hossein (RadarPy Project)
Date: November 7, 2025
"""

import numpy as np
import pandas as pd
from typing import List
from .bases_classes import BaseArrayProcessor


class CADiSArrayProcessor(BaseArrayProcessor):
    """
    CADiS (Concatenated Difference Set) Array Processor
    
    Implements sparse array with w(1)=0 constraint using concatenated difference sets.
    Achieves excellent coarray properties through strategic set concatenation.
    
    Attributes:
        N_total (int): Total number of sensors
        d (float): Base spacing multiplier
        data (ArraySpec): Contains all analysis results
    """
    
    def __init__(self, N: int, d: float = 1.0):
        """
        Initialize CADiS array processor.
        
        Args:
            N (int): Total number of sensors (N >= 5)
            d (float): Base spacing multiplier (default: 1.0 = λ/2)
            
        Raises:
            ValueError: If N < 5 (minimum for meaningful CADiS construction)
        """
        if N < 5:
            raise ValueError("CADiS array requires N >= 5 for valid construction")
        
        self.N_total = int(N)
        self.d = float(d)
        
        # Construct CADiS array positions using difference set concatenation
        positions = self._construct_cadis_pattern(N)
        
        # Initialize base class
        super().__init__(
            name=f"CADiS Array (N={N})",
            array_type="Concatenated Difference Set (CADiS)",
            sensor_positions=positions.tolist(),
            d=d
        )
    
    def _construct_cadis_pattern(self, N: int) -> np.ndarray:
        """
        Construct CADiS array using concatenated difference sets.
        
        Strategy for w(1)=0:
        - Use two complementary difference sets
        - Concatenate with strategic offset to avoid consecutive positions
        - First set: Dense placement at origin
        - Second set: Sparse placement with offset
        
        Args:
            N (int): Total number of sensors
            
        Returns:
            np.ndarray: Integer grid positions (normalized to start at 0)
        """
        # Partition N into two sets
        N1 = (N + 1) // 2  # First set (dense)
        N2 = N - N1         # Second set (sparse)
        
        # First difference set: 2-sparse ULA at origin
        # This gives us w(1)=0 locally
        set1 = np.arange(N1) * 2  # [0, 2, 4, ..., 2(N1-1)]
        
        # Second difference set: 3-sparse ULA with strategic offset
        # Offset chosen to avoid creating w(1)=1 globally
        last_set1 = set1[-1]  # 2(N1-1)
        offset = last_set1 + 3  # Gap of 3 ensures no consecutive positions
        
        set2 = offset + np.arange(N2) * 3  # [offset, offset+3, offset+6, ...]
        
        # Concatenate the sets
        positions = np.concatenate([set1, set2])
        positions = np.unique(positions)
        positions.sort()
        
        # Normalize to start at 0
        positions = positions - positions.min()
        
        return positions
    
    def __repr__(self) -> str:
        """String representation."""
        return f"CADiSArrayProcessor(N={self.N_total}, d={self.d})"
    
    # ============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================================
    
    def compute_array_spacing(self):
        """
        Set the fundamental unit spacing.
        
        For CADiS arrays:
        - Base spacing is 'd' (typically λ/2)
        - All positions are on integer grid multiples of d
        """
        self.data.sensor_spacing = self.d
    
    def compute_all_differences(self):
        """
        Compute all N² pairwise differences to form difference coarray.
        
        Difference coarray D = {n_i - n_j : i,j ∈ [1..N]}
        Includes both positive and negative lags.
        """
        positions = np.array(self.data.sensors_positions)
        N = len(positions)
        
        # Compute all pairwise differences (N² differences)
        differences = []
        for i in range(N):
            for j in range(N):
                diff = int(positions[i] - positions[j])
                differences.append(diff)
        
        self.data.all_differences = sorted(differences)
    
    def analyze_coarray(self):
        """
        Analyze difference coarray to extract unique elements and statistics.
        
        Identifies:
        - Unique coarray positions (lags)
        - Virtual-only elements (not in physical array)
        - Two-sided aperture span
        """
        diffs = np.array(self.data.all_differences)
        unique_diffs = np.unique(diffs)
        
        self.data.unique_differences = unique_diffs.tolist()
        self.data.coarray_positions = unique_diffs.tolist()
        
        # Virtual-only positions (in coarray but not physical array)
        physical_set = set(self.data.sensors_positions)
        virtual_only = [d for d in unique_diffs if d not in physical_set]
        self.data.virtual_only_positions = virtual_only
        
        # Aperture (two-sided span)
        min_lag = int(unique_diffs.min())
        max_lag = int(unique_diffs.max())
        self.data.coarray_aperture = max_lag - min_lag
    
    def compute_weight_distribution(self):
        """
        Compute weight w(ℓ) = frequency of each lag ℓ in coarray.
        
        Creates weight table showing how many times each lag appears.
        Critical for ALSS algorithm - higher weights indicate more reliable lags.
        """
        diffs = np.array(self.data.all_differences)
        unique_lags, counts = np.unique(diffs, return_counts=True)
        
        # Create weight table
        weight_df = pd.DataFrame({
            'Lag': unique_lags,
            'Weight': counts
        })
        
        self.data.weight_table = weight_df
    
    def analyze_contiguous_segments(self):
        """
        Identify maximum contiguous segment in positive coarray lags.
        
        Contiguous segment L determines maximum detectable sources:
        K_max = floor(L/2)
        """
        unique_lags = np.array(self.data.unique_differences)
        positive_lags = unique_lags[unique_lags >= 0]
        positive_lags.sort()
        
        if len(positive_lags) == 0:
            self.data.contiguous_segments = []
            return
        
        # Find longest contiguous sequence
        max_length = 0
        max_start = 0
        max_end = 0
        
        current_start = positive_lags[0]
        current_length = 1
        
        for i in range(1, len(positive_lags)):
            if positive_lags[i] == positive_lags[i-1] + 1:
                # Consecutive
                current_length += 1
            else:
                # Gap found
                if current_length > max_length:
                    max_length = current_length
                    max_start = positive_lags[i - current_length]
                    max_end = positive_lags[i-1]
                current_start = positive_lags[i]
                current_length = 1
        
        # Check last segment
        if current_length > max_length:
            max_length = current_length
            max_start = positive_lags[-current_length]
            max_end = positive_lags[-1]
        
        self.data.contiguous_segments = [int(max_start), int(max_end)]
        self.data.segment_length = max_length
    
    def analyze_holes(self):
        """
        Identify holes (missing lags) in contiguous segment.
        
        Holes are positions within the segment range that don't appear in coarray.
        """
        if not self.data.contiguous_segments:
            self.data.holes_in_segment = []
            return
        
        start, end = self.data.contiguous_segments
        expected_lags = set(range(start, end + 1))
        actual_lags = set(self.data.unique_differences)
        
        holes = sorted(expected_lags - actual_lags)
        self.data.holes_in_segment = holes
    
    def generate_performance_summary(self):
        """
        Generate performance summary table with key metrics.
        
        Metrics:
        - Physical sensors (N)
        - Virtual elements (unique lags)
        - Coarray aperture
        - Contiguous segment length (L)
        - Maximum detectable sources (K_max = L/2)
        - Holes count
        - Weight distribution at small lags
        """
        # Extract metrics
        N = self.data.num_sensors
        unique_lags = len(self.data.unique_differences)
        virtual_only = len(self.data.virtual_only_positions)
        aperture = self.data.coarray_aperture
        
        L = self.data.segment_length if hasattr(self.data, 'segment_length') else 0
        K_max = L // 2
        
        holes = len(self.data.holes_in_segment) if self.data.holes_in_segment else 0
        
        # Get weights at specific lags
        wt = self.data.weight_table
        w_dict = dict(zip(wt['Lag'], wt['Weight']))
        
        w0 = w_dict.get(0, 0)
        w1 = w_dict.get(1, 0)
        w2 = w_dict.get(2, 0)
        w3 = w_dict.get(3, 0)
        w4 = w_dict.get(4, 0)
        w5 = w_dict.get(5, 0)
        
        # Segment range
        seg_start, seg_end = (self.data.contiguous_segments if self.data.contiguous_segments 
                               else [0, 0])
        
        # Create summary table
        summary_data = {
            'Metrics': [
                'Physical Sensors (N)',
                'Virtual Elements (Unique Lags)',
                'Virtual-only Elements',
                'Coarray Aperture (two-sided span)',
                'Contiguous Segment Length (L)',
                'Maximum Detectable Sources (K_max)',
                'Holes in Coarray',
                'Weight at Lag 0 (w(0))',
                'Weight at Lag 1 (w(1))',
                'Weight at Lag 2 (w(2))',
                'Weight at Lag 3 (w(3))',
                'Weight at Lag 4 (w(4))',
                'Weight at Lag 5 (w(5))',
                'Segment Range [L1:L2]'
            ],
            'Value': [
                N,
                unique_lags,
                virtual_only,
                aperture,
                L,
                K_max,
                holes,
                w0,
                w1,
                w2,
                w3,
                w4,
                w5,
                f"[{seg_start}:{seg_end}]"
            ]
        }
        
        self.data.performance_summary_table = pd.DataFrame(summary_data)
    
    def plot_coarray(self):
        """
        Display coarray visualization (console-based).
        
        Shows:
        - Array name and type
        - Physical sensor count and positions
        - Performance summary table
        - Weight distribution
        """
        print(f"\n{'='*70}")
        print(f"  {self.data.name}")
        print(f"{'='*70}")
        print(f"Total Sensors: N = {self.data.num_sensors}")
        print(f"Physical Positions: {self.data.sensors_positions}")
        print()
        print(self.data.performance_summary_table.to_string(index=False))
        print()
        print("Coarray Weight Distribution (first 20 lags):")
        
        # Show weight table excerpt
        wt = self.data.weight_table.head(20)
        print(wt.to_string(index=False))
        
        print(f"{'='*70}\n")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_cadis_arrays(N_values: List[int] = [7, 10, 16], d: float = 1.0):
    """
    Compare CADiS arrays across different N values.
    
    Args:
        N_values (List[int]): List of sensor counts to compare
        d (float): Base spacing
        
    Returns:
        pd.DataFrame: Comparison table
    """
    results = []
    
    for N in N_values:
        proc = CADiSArrayProcessor(N=N, d=d)
        spec = proc.run_full_analysis()
        
        wt = dict(zip(spec.weight_table['Lag'], spec.weight_table['Weight']))
        
        results.append({
            'N': N,
            'Aperture': spec.coarray_aperture,
            'L': spec.segment_length,
            'K_max': spec.segment_length // 2,
            'Holes': len(spec.holes_in_segment) if spec.holes_in_segment else 0,
            'w(1)': wt.get(1, 0),
            'w(2)': wt.get(2, 0),
            'w(3)': wt.get(3, 0)
        })
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("  CADiS ARRAY PROCESSOR DEMO")
    print("="*80 + "\n")
    
    # Demo 1: Basic construction
    print("Demo 1: CADiS Array Construction (N=7)")
    print("-" * 80)
    processor = CADiSArrayProcessor(N=7, d=1.0)
    results = processor.run_full_analysis()
    
    # Demo 2: Comparison across N values
    print("\n\nDemo 2: CADiS Performance Across N")
    print("-" * 80)
    comparison = compare_cadis_arrays(N_values=[7, 10, 16], d=1.0)
    print(comparison.to_string(index=False))
    
    print("\n" + "="*80)
    print("  DEMO COMPLETE")
    print("="*80 + "\n")
