"""
ANAII-2 Array Processor (Augmented Nested Array II-2)
======================================================

Implements the Augmented Nested Array II-2 (ANAII-2) geometry from:
Liu and Vaidyanathan, "Remarks on the Spatial Smoothing Step in Coarray MUSIC,"
IEEE Signal Processing Letters, 2015.

Construction Formula:
--------------------
ANAII-2 consists of three subarrays with strategic offset placement:

- **Subarray P1** (dense): [1, 2, 3, ..., N1]
  - N1 consecutive sensors with unit spacing
  - Provides dense sampling at the origin
  
- **Subarray P2** (bridge): (N1+1) × [1, 2, 3]
  - 3 sensors at positions [(N1+1), 2(N1+1), 3(N1+1)]
  - Offset chosen to bridge gap between P1 and P3
  - Creates additional coarray lags
  
- **Subarray P3** (sparse): (N1+4) × [1, 2, ..., N2]
  - N2 sensors with spacing (N1+4)
  - Offset (N1+4) = N1 + 1 + 3 ensures optimal lag coverage
  - Generates long-range virtual sensors

**Total sensors**: N = N1 + 3 + N2

Mathematical Insight:
--------------------
The offsets (N1+1) and (N1+4) are carefully chosen to:
1. Minimize holes in the difference coarray
2. Maximize contiguous segment length L
3. Achieve K_max = floor(L/2) detectable sources

The three-subarray design creates a rich set of pairwise differences, 
resulting in a longer contiguous coarray segment compared to standard 
two-subarray nested arrays with similar N.

Performance Characteristics:
---------------------------
- **Aperture**: O(N1·N2) - grows with both parameters
- **Segment Length**: Typically L > 2N for well-chosen N1, N2
- **Degrees of Freedom**: K_max ≈ N/2 to N (depends on N1/N2 ratio)
- **Advantage**: Better than standard nested arrays for DOA estimation

Reference:
---------
[10] C. L. Liu and P. P. Vaidyanathan, "Remarks on the spatial smoothing 
     step in coarray MUSIC," IEEE Signal Processing Letters, vol. 22, no. 9, 
     pp. 1438-1442, 2015.

Author: Hossein (RadarPy Project)
Date: November 7, 2025
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from .bases_classes import BaseArrayProcessor


class ANAII2ArrayProcessor(BaseArrayProcessor):
    """
    Augmented Nested Array II-2 (ANAII-2) Processor
    
    Three-subarray nested configuration with strategic offset placement
    for enhanced difference coarray properties.
    
    This processor implements the ANAII-2 geometry which achieves better
    coarray performance than standard two-subarray nested arrays through
    careful offset selection for the three subarrays.
    
    Parameters
    ----------
    N1 : int
        Number of sensors in first subarray (dense consecutive), N1 ≥ 1
    N2 : int
        Number of sensors in third subarray (sparse with offset), N2 ≥ 1
    d : float, optional
        Base sensor spacing multiplier in wavelengths (default: 1.0 = λ/2)
    
    Attributes
    ----------
    N1 : int
        First subarray size (dense)
    N2 : int
        Third subarray size (sparse)
    total_sensors : int
        Total number of physical sensors = N1 + 3 + N2
    data : ArraySpec
        Container with all analysis results (inherited from BaseArrayProcessor)
    
    Notes
    -----
    The second subarray P2 is fixed at 3 elements, positioned at offsets
    that bridge the gap between P1 and P3. This design maximizes the
    contiguous segment length in the difference coarray.
    
    Examples
    --------
    >>> # Create ANAII-2 with N1=4, N2=3 (total N=10 sensors)
    >>> processor = ANAII2ArrayProcessor(N1=4, N2=3, d=1.0)
    >>> results = processor.run_full_analysis()
    >>> 
    >>> # Display performance metrics
    >>> print(results.performance_summary_table.to_markdown(index=False))
    >>> 
    >>> # Access specific metrics
    >>> print(f"Aperture: {results.coarray_aperture}")
    >>> print(f"Segment Length: {results.segment_length}")
    >>> print(f"Max Sources: {results.segment_length // 2}")
    >>> 
    >>> # Compare different configurations
    >>> from geometry_processors.anaii2_processor import compare_anaii2_arrays
    >>> comparison = compare_anaii2_arrays(
    ...     N1_values=[3, 4, 5], 
    ...     N2_values=[2, 3, 4]
    ... )
    >>> print(comparison)
    
    See Also
    --------
    NestedArrayProcessor : Standard two-subarray nested array
    SNA3ArrayProcessor : Super nested array with 3 subarrays
    """
    
    def __init__(self, N1: int, N2: int, d: float = 1.0):
        """
        Initialize ANAII-2 array processor.
        
        Parameters
        ----------
        N1 : int
            Number of sensors in first subarray (must be ≥ 1)
        N2 : int
            Number of sensors in third subarray (must be ≥ 1)
        d : float, optional
            Base sensor spacing multiplier (default: 1.0)
        
        Raises
        ------
        ValueError
            If N1 < 1 or N2 < 1
        """
        if N1 < 1:
            raise ValueError(f"N1 must be ≥ 1, got N1={N1}")
        if N2 < 1:
            raise ValueError(f"N2 must be ≥ 1, got N2={N2}")
        
        self.N1 = N1
        self.N2 = N2
        self.total_sensors = N1 + 3 + N2  # Fixed 3 elements in P2
        
        # Construct sensor positions
        positions = self._construct_anaii2_pattern(N1, N2, d)
        
        # Initialize base processor
        super().__init__(
            name=f"ANAII-2 Array (N1={N1}, N2={N2})",
            array_type="Augmented Nested Array II-2 (ANAII-2)",
            sensor_positions=positions,
            d=d
        )
    
    def _construct_anaii2_pattern(self, N1: int, N2: int, d: float) -> np.ndarray:
        """
        Construct ANAII-2 sensor positions using three-subarray configuration.
        
        This method implements the ANAII-2 construction formula with strategic
        offset placement to maximize difference coarray coverage.
        
        Construction Algorithm:
        ----------------------
        1. **Subarray P1** (dense at origin):
           P1 = d × [1, 2, 3, ..., N1]
           
        2. **Subarray P2** (bridge with 3 sensors):
           P2 = d × (N1+1) × [1, 2, 3]
                = [d(N1+1), 2d(N1+1), 3d(N1+1)]
           
        3. **Subarray P3** (sparse at large offset):
           P3 = d × (N1+4) × [1, 2, ..., N2]
           
        4. Combine all subarrays: Positions = P1 ∪ P2 ∪ P3
        
        5. Convert to zero-based by subtracting minimum position
        
        Offset Rationale:
        ----------------
        - (N1+1): Creates overlap in difference coarray with P1
        - (N1+4) = N1 + 1 + 3: Ensures P3 starts after P2's span
        - These offsets minimize holes and maximize contiguous lags
        
        Parameters
        ----------
        N1 : int
            First subarray size (dense consecutive)
        N2 : int
            Third subarray size (sparse)
        d : float
            Base spacing multiplier (typically λ/2)
        
        Returns
        -------
        np.ndarray
            Integer sensor positions (zero-based, sorted)
            
        Examples
        --------
        >>> processor = ANAII2ArrayProcessor(N1=3, N2=2, d=1.0)
        >>> # Expected positions (zero-based):
        >>> # P1: [0, 1, 2]  (from [1,2,3] - 1)
        >>> # P2: [3, 7, 11] (from [4,8,12] - 1)
        >>> # P3: [6, 13]    (from [7,14] - 1)
        >>> # Combined: [0, 1, 2, 3, 6, 7, 11, 13]
        """
        # Subarray 1: Dense consecutive [1, 2, ..., N1]
        P1 = d * np.arange(1, N1 + 1)
        
        # Subarray 2: Three elements at offset (N1+1)
        # (N1+1) * [1, 2, 3] = [N1+1, 2*(N1+1), 3*(N1+1)]
        offset_P2 = N1 + 1
        P2 = d * offset_P2 * np.arange(1, 4)  # [1, 2, 3]
        
        # Subarray 3: Sparse array at offset (N1+4)
        # (N1+4) * [1, 2, ..., N2]
        offset_P3 = N1 + 4
        P3 = d * offset_P3 * np.arange(1, N2 + 1)
        
        # Combine all subarrays
        positions_1based = np.concatenate([P1, P2, P3])
        
        # Convert to zero-based (subtract minimum position)
        positions = positions_1based - np.min(positions_1based)
        
        # Sort and return as integers
        return np.sort(positions).astype(int)
    
    def compute_array_spacing(self) -> float:
        """
        Compute base array spacing for ANAII-2.
        
        Returns smallest non-zero difference between consecutive sensors in P1.
        For ANAII-2, this is the unit spacing 'd' in the dense subarray.
        
        Returns
        -------
        float
            Base spacing (d)
        """
        positions = np.array(self.data.sensors_positions)
        
        # P1 is the dense consecutive part (first N1 elements)
        P1 = positions[:self.N1]
        
        # Smallest spacing in P1 (should be d)
        if len(P1) > 1:
            diffs = np.diff(P1)
            return float(np.min(diffs[diffs > 0]))
        
        # Fallback: compute from all positions
        all_diffs = np.diff(positions)
        return float(np.min(all_diffs[all_diffs > 0]))
    
    def compute_all_differences(self):
        """
        Compute all N² pairwise differences to form the difference coarray.
        
        For ANAII-2 with N = N1 + 3 + N2 sensors, this computes all possible
        differences between sensor pairs: diff[i,j] = pos[i] - pos[j].
        
        The difference coarray D = {n_i - n_j : i,j ∈ [1..N]} contains:
        - N zero-lag differences (self-differences)
        - N(N-1)/2 positive-lag differences
        - N(N-1)/2 negative-lag differences
        
        Total: N² differences (with duplicates preserved for weight counting)
        
        Stores result in self.data.all_differences as a sorted list.
        
        Mathematical Background:
        -----------------------
        The three-subarray design creates multiple types of differences:
        1. Within-subarray differences (P1, P2, P3 internal)
        2. Cross-subarray differences (P1-P2, P1-P3, P2-P3)
        
        The strategic offsets ensure that cross-subarray differences fill
        gaps in the coarray, maximizing contiguous coverage.
        """
        positions = np.array(self.data.sensors_positions)
        N = len(positions)
        
        # Compute all pairwise differences
        differences = []
        for i in range(N):
            for j in range(N):
                diff = int(positions[i] - positions[j])
                differences.append(diff)
        
        self.data.all_differences = sorted(differences)
    
    def analyze_coarray(self):
        """
        Analyze difference coarray structure to extract unique lags and statistics.
        
        This method identifies:
        1. **Unique lags**: Distinct values in the difference coarray
        2. **Virtual-only positions**: Lags present in coarray but not in physical array
        3. **Coarray aperture**: Two-sided span (max_lag - min_lag)
        
        For ANAII-2, the three-subarray design typically produces:
        - Dense coverage around zero lag (from P1 internal differences)
        - Extended coverage from cross-subarray differences
        - Longer aperture than standard nested arrays
        
        Results are stored in self.data:
        - unique_differences: Sorted list of unique lags
        - coarray_positions: Alias for unique_differences
        - virtual_only_positions: Lags not in physical array
        - coarray_aperture: Total two-sided span
        
        Notes
        -----
        Virtual sensors enable DOA estimation beyond the physical array's
        resolution. ANAII-2's virtual array typically has 2-4× more elements
        than the physical array (depending on N1/N2 ratio).
        """
        positions = np.array(self.data.sensors_positions)
        all_differences = np.array(self.data.all_differences)
        
        # Find unique differences (lags)
        unique_diffs = np.unique(all_differences)
        
        # Store in self.data
        self.data.unique_differences = unique_diffs.tolist()
        self.data.coarray_positions = unique_diffs.tolist()
        
        # Identify virtual-only positions (lags not in physical array)
        physical_set = set(positions)
        virtual_only = [d for d in unique_diffs if d not in physical_set]
        self.data.virtual_only_positions = virtual_only
        
        # Compute aperture (two-sided span)
        min_lag = int(unique_diffs.min())
        max_lag = int(unique_diffs.max())
        self.data.coarray_aperture = max_lag - min_lag
    
    def compute_weight_distribution(self):
        """
        Compute weight distribution (frequency of each lag).
        Stores results in self.data.weight_table.
        """
        all_differences = np.array(self.data.all_differences)
        unique_lags, counts = np.unique(all_differences, return_counts=True)
        
        weight_table = pd.DataFrame({
            'Lag': unique_lags,
            'Weight': counts
        })
        
        # Sort by lag (already sorted from np.unique, but explicit for clarity)
        weight_table = weight_table.sort_values('Lag').reset_index(drop=True)
        
        self.data.weight_table = weight_table
    
    def analyze_contiguous_segments(self):
        """
        Find contiguous segments in the coarray (zero-lag centered).
        
        Identifies the longest contiguous sequence of integer lags around zero.
        This determines maximum detectable sources K_max = floor(L/2).
        Stores results in self.data.contiguous_segments and self.data.segment_length.
        """
        unique_differences = self.data.unique_differences
        
        # Find contiguous segment around zero
        unique_diffs_sorted = np.sort(unique_differences)
        
        # Check if zero is present (should always be true)
        if 0 not in unique_diffs_sorted:
            self.data.contiguous_segments = [0, 0]
            self.data.segment_length = 0
            return
        
        # Find longest contiguous segment containing zero
        # Strategy: expand from zero in both directions
        zero_idx = np.where(unique_diffs_sorted == 0)[0][0]
        
        # Expand left (negative lags)
        left_idx = zero_idx
        while left_idx > 0:
            if unique_diffs_sorted[left_idx] - unique_diffs_sorted[left_idx - 1] == 1:
                left_idx -= 1
            else:
                break
        
        # Expand right (positive lags)
        right_idx = zero_idx
        while right_idx < len(unique_diffs_sorted) - 1:
            if unique_diffs_sorted[right_idx + 1] - unique_diffs_sorted[right_idx] == 1:
                right_idx += 1
            else:
                break
        
        # Extract segment
        segment_start = int(unique_diffs_sorted[left_idx])
        segment_end = int(unique_diffs_sorted[right_idx])
        segment_length = segment_end - segment_start + 1
        
        # Store in self.data
        self.data.contiguous_segments = [segment_start, segment_end]
        self.data.segment_length = segment_length
    
    def analyze_holes(self):
        """
        Identify holes (missing lags) in the contiguous segment.
        Stores results in self.data.holes_in_segment.
        """
        if not self.data.contiguous_segments:
            self.data.holes_in_segment = []
            return
        
        start, end = self.data.contiguous_segments
        unique_differences = self.data.unique_differences
        
        # Expected lags in contiguous segment
        expected_lags = set(range(start, end + 1))
        
        # Find missing lags
        present_lags = set(unique_differences)
        holes = sorted(expected_lags - present_lags)
        
        self.data.holes_in_segment = holes
    
    def generate_performance_summary(self):
        """
        Generate comprehensive performance summary table.
        Stores result in self.data.performance_summary_table.
        """
        data = self.data
        
        # Extract key metrics
        N = data.num_sensors
        unique_lags = len(data.unique_differences)
        virtual_only = len(data.virtual_only_positions)
        aperture = data.coarray_aperture
        L = data.segment_length if hasattr(data, 'segment_length') else 0
        K_max = L // 2
        num_holes = len(data.holes_in_segment) if hasattr(data, 'holes_in_segment') and data.holes_in_segment else 0
        
        # Extract weights for small lags
        wt_dict = dict(zip(data.weight_table['Lag'], data.weight_table['Weight']))
        w0 = wt_dict.get(0, 0)
        w1 = wt_dict.get(1, 0)
        w2 = wt_dict.get(2, 0)
        w3 = wt_dict.get(3, 0)
        w4 = wt_dict.get(4, 0)
        w5 = wt_dict.get(5, 0)
        
        # Segment range
        seg_start, seg_end = (data.contiguous_segments if hasattr(data, 'contiguous_segments') and data.contiguous_segments 
                              else [0, 0])
        segment_range = f"[{seg_start}:{seg_end}]"
        
        # Create summary table
        summary = pd.DataFrame({
            'Metrics': [
                'Physical Sensors (N)',
                'Subarray Sizes (N1, N2)',
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
                f"({self.N1}, {self.N2})",
                unique_lags,
                virtual_only,
                aperture,
                L,
                K_max,
                num_holes,
                w0,
                w1,
                w2,
                w3,
                w4,
                w5,
                segment_range
            ]
        })
        
        self.data.performance_summary_table = summary
    
    def plot_coarray(self, save_path: Optional[str] = None):
        """
        Generate coarray visualization (console output).
        
        Parameters
        ----------
        save_path : str, optional
            Path to save plot (currently not implemented, console only)
        """
        print(f"\n{'='*70}")
        print(f"  {self.data.name}")
        print(f"{'='*70}")
        print(f"Total Sensors: N = {self.data.num_sensors}")
        print(f"Subarray Configuration: N1={self.N1}, N2={self.N2} (P2 fixed at 3 elements)")
        
        # Show physical positions by subarray
        positions = np.array(self.data.sensors_positions)
        P1 = positions[:self.N1]
        P2 = positions[self.N1:self.N1+3]
        P3 = positions[self.N1+3:]
        
        print(f"\nPhysical Positions:")
        print(f"  P1 (dense): {P1}")
        print(f"  P2 (offset): {P2}")
        print(f"  P3 (sparse): {P3}")
        
        # Performance summary
        print(f"\n{self.data.performance_summary_table.to_string(index=False)}")
        
        # Show weight distribution (first 20 lags)
        print(f"\nCoarray Weight Distribution (first 20 lags):")
        weight_subset = self.data.weight_table.head(20)
        print(weight_subset.to_string(index=False))
        
        print(f"{'='*70}\n")
    
    def __repr__(self) -> str:
        """String representation of ANAII-2 processor."""
        return (f"ANAII2ArrayProcessor(N1={self.N1}, N2={self.N2}, "
                f"total_N={self.total_sensors})")


def compare_anaii2_arrays(N1_values: List[int], N2_values: List[int], 
                          d: float = 1.0) -> pd.DataFrame:
    """
    Compare multiple ANAII-2 configurations side-by-side.
    
    This utility function analyzes multiple ANAII-2 arrays with different
    (N1, N2) parameters and compiles their performance metrics into a
    comparison table.
    
    Useful for:
    - Parameter selection (choosing optimal N1/N2 ratio)
    - Performance scaling studies
    - Trade-off analysis between array size and DOF
    
    Parameters
    ----------
    N1_values : List[int]
        List of N1 values (first subarray sizes) to test
    N2_values : List[int]
        List of N2 values (third subarray sizes) to test
        Must have same length as N1_values
    d : float, optional
        Base spacing multiplier for all arrays (default: 1.0)
    
    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        - N1, N2: Subarray sizes
        - Total_N: Total physical sensors (N1 + 3 + N2)
        - Aperture: Two-sided coarray span
        - Unique_Lags: Number of unique virtual sensors
        - L: Contiguous segment length
        - K_max: Maximum detectable sources
        - Holes: Number of missing lags in segment
        - w(1): Weight at lag 1 (higher is better)
    
    Raises
    ------
    ValueError
        If N1_values and N2_values have different lengths
    
    Examples
    --------
    >>> # Compare small to large configurations
    >>> comparison = compare_anaii2_arrays(
    ...     N1_values=[2, 4, 6, 8],
    ...     N2_values=[2, 3, 4, 5],
    ...     d=1.0
    ... )
    >>> print(comparison.to_markdown(index=False))
    >>> 
    >>> # Analyze scaling: fix N1, vary N2
    >>> scaling = compare_anaii2_arrays(
    ...     N1_values=[5, 5, 5, 5],
    ...     N2_values=[2, 4, 6, 8]
    ... )
    """
    if len(N1_values) != len(N2_values):
        raise ValueError("N1_values and N2_values must have same length")
    
    results = []
    
    for N1, N2 in zip(N1_values, N2_values):
        processor = ANAII2ArrayProcessor(N1=N1, N2=N2, d=d)
        data = processor.run_full_analysis()
        
        # Compute K_max from segment length
        L = data.segment_length
        K_max = L // 2
        
        # Count holes
        num_holes = len(data.holes_in_segment) if hasattr(data, 'holes_in_segment') and data.holes_in_segment else 0
        
        results.append({
            'N1': N1,
            'N2': N2,
            'Total_N': N1 + 3 + N2,
            'Aperture': data.coarray_aperture,
            'Unique_Lags': len(data.unique_differences),
            'L': L,
            'K_max': K_max,
            'Holes': num_holes,
            'w(1)': dict(zip(data.weight_table['Lag'], 
                            data.weight_table['Weight'])).get(1, 0)
        })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    """
    Demo: ANAII-2 Array Analysis
    """
    print("\n" + "="*70)
    print("  ANAII-2 (Augmented Nested Array II-2) Demonstration")
    print("="*70 + "\n")
    
    # Example 1: N1=4, N2=3 (Total N=10)
    print("Example 1: N1=4, N2=3")
    print("-" * 70)
    processor1 = ANAII2ArrayProcessor(N1=4, N2=3, d=1.0)
    results1 = processor1.run_full_analysis()
    
    # Example 2: N1=6, N2=4 (Total N=13)
    print("\n\nExample 2: N1=6, N2=4")
    print("-" * 70)
    processor2 = ANAII2ArrayProcessor(N1=6, N2=4, d=1.0)
    results2 = processor2.run_full_analysis()
    
    # Comparison
    print("\n\nComparison of Multiple ANAII-2 Configurations:")
    print("="*70)
    comparison = compare_anaii2_arrays(
        N1_values=[3, 4, 5, 6],
        N2_values=[2, 3, 4, 5],
        d=1.0
    )
    print(comparison.to_markdown(index=False))
