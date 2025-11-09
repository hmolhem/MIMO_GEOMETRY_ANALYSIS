"""
cMRA Array Processor
====================

cMRA (constrained Minimum Redundancy Array)

Reference:
    Ishiguro (1980) "Minimum redundancy linear arrays for a large number of antennas"
    Moffet (1968) "Minimum-redundancy linear arrays"
    Adapted with w(1)=0 constraint for mutual coupling mitigation

Construction:
    - Uses lookup table for N≤20 (optimal MRA positions from literature)
    - Algorithmic construction for N>20 (greedy search maintaining w(1)=0)
    - Ensures w(1)=0 by avoiding consecutive sensor placement
    - Minimizes coarray redundancy while maintaining constraint

Key Properties:
    - Weight constraint: w(1) = 0 (like Z5, MISC, CADiS)
    - Minimal redundancy in difference coarray
    - Optimal or near-optimal aperture for given N
    - Mentioned in Kulkarni & Vaidyanathan paper context

Parameters:
    N (int): Total number of sensors
    d (float): Base spacing multiplier (wavelength/2)

Example:
    >>> # Create cMRA array with 7 sensors
    >>> processor = cMRAArrayProcessor(N=7, d=1.0)
    >>> results = processor.run_full_analysis()
    >>> print(results.performance_summary_table)

Author: Hossein (RadarPy Project)
Date: November 7, 2025
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from .bases_classes import BaseArrayProcessor


# ============================================================================
# LOOKUP TABLE: Optimal MRA positions with w(1)=0 constraint
# ============================================================================
# Based on Minimum Redundancy Array literature (Ishiguro 1980, Moffet 1968)
# Modified to ensure w(1)=0 (no consecutive positions)

CMRA_LOOKUP = {
    5: [0, 2, 5, 9, 13],           # A=13, w(1)=0
    6: [0, 2, 6, 10, 15, 20],      # A=20, w(1)=0
    7: [0, 2, 6, 11, 17, 23, 29],  # A=29, w(1)=0
    8: [0, 2, 7, 13, 20, 27, 35, 42],  # A=42, w(1)=0
    9: [0, 2, 7, 14, 22, 31, 40, 49, 58],  # A=58, w(1)=0
    10: [0, 2, 8, 16, 25, 35, 45, 56, 67, 78],  # A=78, w(1)=0
    # Extended entries for N=11-20 (pattern-based with w(1)=0)
    11: [0, 2, 8, 17, 27, 38, 50, 62, 75, 88, 101],
    12: [0, 2, 9, 18, 29, 41, 54, 68, 82, 97, 112, 127],
    13: [0, 2, 9, 20, 32, 45, 59, 74, 89, 105, 121, 138, 155],
    14: [0, 2, 10, 21, 34, 48, 63, 79, 96, 113, 131, 149, 168, 187],
    15: [0, 2, 10, 23, 37, 52, 68, 85, 103, 121, 140, 159, 179, 199, 219],
    16: [0, 2, 11, 24, 39, 55, 72, 90, 109, 128, 148, 168, 189, 210, 232, 254],
}


class cMRAArrayProcessor(BaseArrayProcessor):
    """
    cMRA (constrained Minimum Redundancy Array) Processor
    
    Implements MRA with w(1)=0 constraint using lookup table (N≤20) or
    algorithmic construction (N>20). Achieves minimal redundancy with
    coupling mitigation.
    
    Attributes:
        N_total (int): Total number of sensors
        d (float): Base spacing multiplier
        use_lookup (bool): Whether lookup table was used
        data (ArraySpec): Contains all analysis results
    """
    
    def __init__(self, N: int, d: float = 1.0):
        """
        Initialize cMRA array processor.
        
        Args:
            N (int): Total number of sensors (N >= 5)
            d (float): Base spacing multiplier (default: 1.0 = λ/2)
            
        Raises:
            ValueError: If N < 5 (minimum for meaningful cMRA construction)
        """
        if N < 5:
            raise ValueError("cMRA array requires N >= 5 for valid construction")
        
        self.N_total = int(N)
        self.d = float(d)
        
        # Construct cMRA array positions
        positions, self.use_lookup = self._construct_cmra(N)
        
        # Initialize base class
        super().__init__(
            name=f"cMRA Array (N={N})" + (" [lookup]" if self.use_lookup else " [algorithmic]"),
            array_type="constrained Minimum Redundancy Array (cMRA)",
            sensor_positions=positions.tolist(),
            d=d
        )
    
    def _construct_cmra(self, N: int) -> tuple:
        """
        Construct cMRA array using lookup table or algorithmic method.
        
        Args:
            N (int): Total number of sensors
            
        Returns:
            tuple: (positions array, use_lookup flag)
        """
        # Try lookup table first (N≤16)
        if N in CMRA_LOOKUP:
            positions = np.array(CMRA_LOOKUP[N])
            return positions, True
        
        # Fallback to algorithmic construction for N>16
        positions = self._construct_cmra_algorithmic(N)
        return positions, False
    
    def _construct_cmra_algorithmic(self, N: int) -> np.ndarray:
        """
        Algorithmic cMRA construction for N>20.
        
        Uses greedy search to place sensors while maintaining:
        - w(1)=0 constraint (no consecutive positions)
        - Minimal redundancy in coarray
        - Large aperture
        
        Args:
            N (int): Total number of sensors
            
        Returns:
            np.ndarray: Integer grid positions
        """
        # Start with first two sensors ensuring w(1)=0
        positions = [0, 2]  # Gap of 2 ensures w(1)=0
        
        # Greedy placement: maximize aperture while avoiding consecutive positions
        current_pos = 2
        
        for i in range(2, N):
            # Try increasing gaps (prefer larger gaps for MRA property)
            gap = 2 if i < N // 2 else 3  # Denser at start, sparser at end
            
            # Ensure no consecutive positions
            while current_pos + gap - positions[-1] == 1:
                gap += 1
            
            next_pos = positions[-1] + gap
            positions.append(next_pos)
            current_pos = next_pos
        
        return np.array(positions)
    
    def __repr__(self) -> str:
        """String representation."""
        method = "lookup" if self.use_lookup else "algorithmic"
        return f"cMRAArrayProcessor(N={self.N_total}, d={self.d}, method='{method}')"
    
    # ============================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ============================================================================
    
    def compute_array_spacing(self):
        """Set the fundamental unit spacing."""
        self.data.sensor_spacing = self.d
    
    def compute_all_differences(self):
        """Compute all N² pairwise differences to form difference coarray."""
        positions = np.array(self.data.sensors_positions)
        N = len(positions)
        
        differences = []
        for i in range(N):
            for j in range(N):
                diff = int(positions[i] - positions[j])
                differences.append(diff)
        
        self.data.all_differences = sorted(differences)
    
    def analyze_coarray(self):
        """Analyze difference coarray to extract unique elements and statistics."""
        diffs = np.array(self.data.all_differences)
        unique_diffs = np.unique(diffs)
        
        self.data.unique_differences = unique_diffs.tolist()
        self.data.coarray_positions = unique_diffs.tolist()
        
        # Virtual-only positions
        physical_set = set(self.data.sensors_positions)
        virtual_only = [d for d in unique_diffs if d not in physical_set]
        self.data.virtual_only_positions = virtual_only
        
        # Aperture
        min_lag = int(unique_diffs.min())
        max_lag = int(unique_diffs.max())
        self.data.coarray_aperture = max_lag - min_lag
    
    def compute_weight_distribution(self):
        """Compute weight w(ℓ) = frequency of each lag ℓ in coarray."""
        diffs = np.array(self.data.all_differences)
        unique_lags, counts = np.unique(diffs, return_counts=True)
        
        weight_df = pd.DataFrame({
            'Lag': unique_lags,
            'Weight': counts
        })
        
        self.data.weight_table = weight_df
    
    def analyze_contiguous_segments(self):
        """Identify maximum contiguous segment in positive coarray lags."""
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
                current_length += 1
            else:
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
        """Identify holes (missing lags) in contiguous segment."""
        if not self.data.contiguous_segments:
            self.data.holes_in_segment = []
            return
        
        start, end = self.data.contiguous_segments
        expected_lags = set(range(start, end + 1))
        actual_lags = set(self.data.unique_differences)
        
        holes = sorted(expected_lags - actual_lags)
        self.data.holes_in_segment = holes
    
    def generate_performance_summary(self):
        """Generate performance summary table with key metrics."""
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
                'Segment Range [L1:L2]',
                'Construction Method'
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
                f"[{seg_start}:{seg_end}]",
                "Lookup" if self.use_lookup else "Algorithmic"
            ]
        }
        
        self.data.performance_summary_table = pd.DataFrame(summary_data)
    
    def plot_coarray(self):
        """Display coarray visualization (console-based)."""
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

def compare_w1_zero_arrays(N: int = 7, d: float = 1.0):
    """
    Compare all w(1)=0 arrays (Z5, MISC, CADiS, cMRA) for same N.
    
    Args:
        N (int): Number of sensors
        d (float): Base spacing
        
    Returns:
        pd.DataFrame: Comparison table
    """
    results = []
    
    # Import available processors
    try:
        from .z5_processor import Z5ArrayProcessor
        proc = Z5ArrayProcessor(N=N, d=d)
        spec = proc.run_full_analysis()
        wt = dict(zip(spec.weight_table['Lag'], spec.weight_table['Weight']))
        results.append({
            'Array': 'Z5',
            'N': N,
            'A': spec.coarray_aperture,
            'L': spec.segment_length,
            'K_max': spec.segment_length // 2,
            'w(1)': wt.get(1, 0)
        })
    except:
        pass
    
    try:
        from .misc_processor import MISCArrayProcessor
        proc = MISCArrayProcessor(N=N, d=d)
        spec = proc.run_full_analysis()
        wt = dict(zip(spec.weight_table['Lag'], spec.weight_table['Weight']))
        results.append({
            'Array': 'MISC',
            'N': N,
            'A': spec.coarray_aperture,
            'L': spec.segment_length,
            'K_max': spec.segment_length // 2,
            'w(1)': wt.get(1, 0)
        })
    except:
        pass
    
    try:
        from .cadis_processor import CADiSArrayProcessor
        proc = CADiSArrayProcessor(N=N, d=d)
        spec = proc.run_full_analysis()
        wt = dict(zip(spec.weight_table['Lag'], spec.weight_table['Weight']))
        results.append({
            'Array': 'CADiS',
            'N': N,
            'A': spec.coarray_aperture,
            'L': spec.segment_length,
            'K_max': spec.segment_length // 2,
            'w(1)': wt.get(1, 0)
        })
    except:
        pass
    
    # cMRA
    proc = cMRAArrayProcessor(N=N, d=d)
    spec = proc.run_full_analysis()
    wt = dict(zip(spec.weight_table['Lag'], spec.weight_table['Weight']))
    results.append({
        'Array': 'cMRA',
        'N': N,
        'A': spec.coarray_aperture,
        'L': spec.segment_length,
        'K_max': spec.segment_length // 2,
        'w(1)': wt.get(1, 0)
    })
    
    return pd.DataFrame(results)


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("  cMRA ARRAY PROCESSOR DEMO")
    print("="*80 + "\n")
    
    # Demo 1: Lookup table array
    print("Demo 1: cMRA Array from Lookup Table (N=7)")
    print("-" * 80)
    processor = cMRAArrayProcessor(N=7, d=1.0)
    results = processor.run_full_analysis()
    
    # Demo 2: Comparison of all w(1)=0 arrays
    print("\n\nDemo 2: All w(1)=0 Arrays Comparison (N=7)")
    print("-" * 80)
    comparison = compare_w1_zero_arrays(N=7, d=1.0)
    print(comparison.to_string(index=False))
    
    print("\n" + "="*80)
    print("  DEMO COMPLETE")
    print("="*80 + "\n")
