"""
MISC Array Processor
====================

MISC (Minimum Inter-element Spacing Constraint) Array

Reference:
    Zheng et al., "MISC Array with Minimum Inter-element Spacing Constraints"
    (Implementation based on common sparse array design principles)

Construction:
    Simplified pattern-based construction achieving w(1)=0:
    - Uses 2-sparse base pattern with strategic augmentation
    - Ensures w(1)=0 by avoiding consecutive sensor placement
    - Optimizes for large aperture while maintaining minimum spacing > d

Key Properties:
    - Weight constraint: w(1) = 0 (like Z5, CADiS, cMRA)
    - Achieves large aperture with sparse placement
    - Suitable for mutual coupling mitigation
    - Comparable performance to Z5 arrays

Parameters:
    N (int): Total number of sensors
    d (float): Base spacing multiplier (wavelength/2)

Example:
    >>> # Create MISC array with 7 sensors
    >>> processor = MISCArrayProcessor(N=7, d=1.0)
    >>> results = processor.run_full_analysis()
    >>> print(results.performance_summary_table)

Author: Hossein (RadarPy Project)
Date: November 7, 2025
"""

import numpy as np
import pandas as pd
from typing import List
from .bases_classes import BaseArrayProcessor


class MISCArrayProcessor(BaseArrayProcessor):
    """
    MISC (Minimum Inter-element Spacing Constraint) Array Processor
    
    Implements sparse array with w(1)=0 constraint following MISC design principles.
    Uses pattern-based construction to achieve large aperture with minimum spacing.
    
    Attributes:
        N_total (int): Total number of sensors
        d (float): Base spacing multiplier
        data (ArraySpec): Contains all analysis results
    """
    
    def __init__(self, N: int, d: float = 1.0):
        """
        Initialize MISC array processor.
        
        Args:
            N (int): Total number of sensors (N >= 5)
            d (float): Base spacing multiplier (default: 1.0 = λ/2)
            
        Raises:
            ValueError: If N < 5 (minimum for meaningful MISC construction)
        """
        if N < 5:
            raise ValueError("MISC array requires N >= 5 for valid construction")
        
        self.N_total = int(N)
        self.d = float(d)
        
        # Construct MISC array positions
        # Strategy: Pattern achieving w(1)=0 with good aperture
        positions = self._construct_misc_pattern(N)
        
        # Initialize base class
        super().__init__(
            name=f"MISC Array (N={N})",
            array_type="Minimum Inter-element Spacing Constraint (MISC)",
            sensor_positions=positions.tolist(),
            d=d
        )
    
    def _construct_misc_pattern(self, N: int) -> np.ndarray:
        """
        Construct MISC array sensor positions using pattern-based approach.
        
        Strategy for w(1)=0:
        - Base 2-sparse ULA for (N-2) sensors: 0, 2, 4, ..., 2(N-3)
        - Strategic augmentation sensors to maximize aperture
        - Ensure no consecutive positions (guarantees w(1)=0)
        
        Args:
            N (int): Total number of sensors
            
        Returns:
            np.ndarray: Integer grid positions (normalized to start at 0)
        """
        # Base 2-sparse ULA for majority of sensors
        N_sparse = N - 2
        sparse_segment = np.arange(N_sparse) * 2  # [0, 2, 4, ..., 2(N-3)]
        
        # Augmentation sensors for aperture extension
        # Pattern: Add sensors at large spacing from last sparse sensor
        last_sparse = sparse_segment[-1]  # 2(N-3)
        aug1 = last_sparse + 3  # Gap of 3
        aug2 = last_sparse + 5  # Gap of 5
        
        augmented_sensors = np.array([aug1, aug2])
        
        # Combine and ensure sorted
        positions = np.concatenate([sparse_segment, augmented_sensors])
        positions = np.unique(positions)
        positions.sort()
        
        # Normalize to start at 0
        positions = positions - positions.min()
        
        return positions
    
    def __repr__(self) -> str:
        """String representation."""
        return f"MISCArrayProcessor(N={self.N_total}, d={self.d})"
    
    # ============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================================
    
    def compute_array_spacing(self):
        """
        Set the fundamental unit spacing.
        
        For MISC arrays:
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
        print("Coarray Weight Distribution (first 10 lags):")
        
        # Show weight table excerpt
        wt = self.data.weight_table.head(20)
        print(wt.to_string(index=False))
        
        print(f"{'='*70}\n")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compare_misc_with_z5(N: int = 7, d: float = 1.0):
    """
    Compare MISC array with Z5 array for same N.
    
    Args:
        N (int): Number of sensors
        d (float): Base spacing
        
    Returns:
        pd.DataFrame: Comparison table
    """
    from .z5_processor import Z5ArrayProcessor
    
    # Run both processors
    misc_proc = MISCArrayProcessor(N=N, d=d)
    z5_proc = Z5ArrayProcessor(N=N, d=d)
    
    misc_results = misc_proc.run_full_analysis()
    z5_results = z5_proc.run_full_analysis()
    
    # Extract key metrics
    def extract_metrics(spec):
        wt = dict(zip(spec.weight_table['Lag'], spec.weight_table['Weight']))
        return {
            'Array': spec.name,
            'N': spec.num_sensors,
            'Aperture': spec.coarray_aperture,
            'L': spec.segment_length,
            'K_max': spec.segment_length // 2,
            'Holes': len(spec.holes_in_segment) if spec.holes_in_segment else 0,
            'w(1)': wt.get(1, 0),
            'w(2)': wt.get(2, 0),
            'w(3)': wt.get(3, 0)
        }
    
    comparison = pd.DataFrame([
        extract_metrics(misc_results),
        extract_metrics(z5_results)
    ])
    
    return comparison


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("  MISC ARRAY PROCESSOR DEMO")
    print("="*80 + "\n")
    
    # Demo 1: Basic construction
    print("Demo 1: MISC Array Construction (N=7)")
    print("-" * 80)
    processor = MISCArrayProcessor(N=7, d=1.0)
    results = processor.run_full_analysis()
    
    # Demo 2: Comparison with Z5
    print("\n\nDemo 2: MISC vs Z5 Comparison")
    print("-" * 80)
    try:
        comparison = compare_misc_with_z5(N=7, d=1.0)
        print(comparison.to_string(index=False))
    except ImportError:
        print("Z5 processor not available for comparison")
    
    print("\n" + "="*80)
    print("  DEMO COMPLETE")
    print("="*80 + "\n")
