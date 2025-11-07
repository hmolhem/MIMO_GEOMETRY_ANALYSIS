"""
Two-level Coprime Array (TCA) Processor
========================================

This module implements the Two-level Coprime Array (TCA) geometry for MIMO radar
applications. TCA exploits the coprime property of two integers to achieve large
aperture with reduced mutual coupling compared to uniform linear arrays.

Mathematical Foundation
-----------------------

The TCA consists of two uniform linear subarrays with coprime integers M and N:

**Construction Formula:**
- Subarray 1 (P1): d × M × [0, 1, 2, ..., N-1]  (N elements, spacing M×d)
- Subarray 2 (P2): d × N × [0, 1, 2, ..., M-1]  (M elements, spacing N×d)
- Total sensors: N_total = M + N

where:
- M, N are coprime integers (gcd(M,N) = 1)
- d is the unit spacing (typically λ/2)
- Both subarrays share the origin (position 0)

**Example (M=3, N=5, d=1):**
```
P1 = [0, 3, 6, 9, 12]     # 5 elements, spacing 3
P2 = [0, 5, 10]           # 3 elements, spacing 5
Combined = [0, 3, 5, 6, 9, 10, 12]  # 7 unique positions (origin shared)
```

Mathematical Insight
--------------------

1. **Coprime Property:**
   When gcd(M,N)=1, the difference coarray has unique properties:
   - Consecutive lags from -(M×N-1) to (M×N-1)
   - Aperture extends to M×N×d
   - DOF approaches O(M×N), much larger than M+N

2. **Aperture Scaling:**
   Maximum aperture A = max(M×(N-1)×d, N×(M-1)×d)
   For typical cases where M≈N, A ≈ M×N×d

3. **Weight Distribution:**
   Unlike nested arrays, coprime arrays have:
   - Sparser weight distribution at small lags
   - More uniform weights across lags
   - Potential holes in coarray for certain (M,N) pairs

4. **Sensor Efficiency:**
   DOF per sensor ratio: K_max/(M+N) ≈ M×N/(2×(M+N))
   For M=N, this approaches N/4, which is less efficient than nested arrays
   but provides better coupling properties

Coarray Properties
------------------

The difference coarray of TCA has distinctive characteristics:

- **Contiguous Range:** 
  For coprime (M,N), consecutive lags exist in range [-(MN-1), MN-1]
  
- **Holes:**
  May have holes outside the main contiguous segment depending on (M,N) choice
  
- **Weight Pattern:**
  More uniform than nested arrays, reducing estimation bias

Performance Characteristics
---------------------------

**Strengths:**
- Large aperture: A ≈ M×N×d with only M+N sensors
- Reduced mutual coupling: Larger inter-element spacing than ULA
- Flexible design: Many coprime pairs available
- Better condition number than nested arrays in some cases

**Limitations:**
- Lower DOF efficiency than nested arrays (K_max ≈ MN/2 vs N² for nested)
- Potential holes in coarray
- More complex optimization (two parameters M, N with coprime constraint)

Typical Use Cases
-----------------

1. **Mutual Coupling Constrained Systems:**
   When antenna elements have large physical diameter relative to wavelength
   
2. **Wideband Applications:**
   Coprime spacing reduces grating lobes across frequency band
   
3. **Robust DOA Estimation:**
   More uniform weight distribution improves estimation robustness
   
4. **Hardware with Spacing Constraints:**
   Flexibility in choosing M, N allows matching to physical constraints

Design Guidelines
-----------------

1. **Choosing M and N:**
   - Ensure gcd(M,N) = 1 (coprime requirement)
   - Common pairs: (2,3), (3,4), (3,5), (4,5), (5,6), (5,7)
   - Larger |M-N| gives longer aperture but sparser coarray
   - M≈N gives more balanced subarray sizes

2. **Typical Configurations:**
   - M=2, N=3: Smallest TCA, 5 sensors, good for testing
   - M=3, N=5: Common choice, 8 sensors, good balance
   - M=4, N=5: 9 sensors, larger aperture
   - M=5, N=7: 12 sensors, high performance

3. **Spacing Selection:**
   - d = λ/2: Maximum angular resolution without aliasing
   - d > λ/2: Reduces coupling, but may introduce ambiguities
   - Minimum spacing: max(M×d, N×d) ≥ element_diameter

Performance Comparison
----------------------

**TCA vs Nested Arrays:**
- TCA: Lower DOF (K_max ≈ MN/2), larger spacing, better coupling
- Nested: Higher DOF (K_max ≈ N²/2), denser P1, higher coupling
- Use TCA when coupling is critical, nested when DOF is priority

**TCA vs ULA:**
- TCA: Much larger aperture (M×N vs M+N-1), more DOF
- ULA: Simpler hardware, no shared element (origin), uniform weights
- TCA clearly superior for sparse arrays

References
----------

1. P. Pal and P. P. Vaidyanathan, "Coprime sampling and the MUSIC algorithm," 
   IEEE DSP Workshop, 2011.
   
2. P. P. Vaidyanathan and P. Pal, "Sparse sensing with co-prime samplers and 
   arrays," IEEE Trans. Signal Process., 2011.
   
3. C.-L. Liu and P. P. Vaidyanathan, "Remarks on the spatial smoothing step 
   in coarray MUSIC," IEEE Signal Process. Letters, 2015.

Implementation Notes
--------------------

- Both subarrays include origin (position 0) → Total unique positions = M+N-1
- Zero-based indexing: P1 uses [0..N-1], P2 uses [0..M-1]
- Difference coarray computed as all pairwise differences
- Coprimality is NOT enforced automatically - user must ensure gcd(M,N)=1
- For non-coprime (M,N), coarray will have more holes and reduced performance

Author: Hossein (RadarPy Project)
Date: November 7, 2025
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from .bases_classes import BaseArrayProcessor, ArraySpec


class TCAArrayProcessor(BaseArrayProcessor):
    """
    Two-level Coprime Array (TCA) geometry processor.
    
    Implements coprime array construction with two uniform subarrays having
    coprime spacing factors M and N. Both subarrays share the origin, resulting
    in M+N-1 unique sensor positions.
    
    Parameters
    ----------
    M : int
        First coprime integer (spacing factor for P2, number of elements in P2).
        Must satisfy gcd(M, N) = 1 for optimal coarray properties.
        Typical range: 2 ≤ M ≤ 10.
        
    N : int
        Second coprime integer (spacing factor for P1, number of elements in P1).
        Must satisfy gcd(M, N) = 1 for optimal coarray properties.
        Typical range: 3 ≤ N ≤ 15.
        
    d : float, optional
        Unit spacing between sensors (typically λ/2).
        Default is 1.0 for normalized analysis.
        
    Attributes
    ----------
    M : int
        First coprime parameter (P2 element count, P1 spacing factor).
        
    N : int
        Second coprime parameter (P1 element count, P2 spacing factor).
        
    d : float
        Unit spacing multiplier.
        
    total_sensors : int
        Total number of unique physical sensors (M + N - 1, origin shared).
        
    is_coprime : bool
        True if gcd(M, N) = 1, False otherwise.
        
    data : ArraySpec
        Container for all analysis results with 47 attributes:
        - Physical array: sensors_positions, num_sensors, sensor_spacing, etc.
        - Coarray: unique_differences, coarray_positions, weight_table, etc.
        - Analysis: contiguous_segments, holes_in_segment, performance_summary, etc.
        
    Design Trade-offs
    -----------------
    
    **M vs N Selection:**
    - **Similar values (M≈N):** Balanced subarray sizes, moderate aperture
      Example: (M=3, N=5) gives aperture ≈15d
      
    - **Different values (|M-N| large):** Unbalanced subarrays, larger aperture
      Example: (M=2, N=7) gives aperture ≈12d but very unbalanced
      
    - **Small values:** Fewer sensors, easier hardware, lower DOF
      Example: (M=2, N=3) gives only 4 unique sensors
      
    - **Large values:** More sensors, higher DOF, but increased cost
      Example: (M=7, N=11) gives 17 sensors
    
    **Coprimality Requirement:**
    - gcd(M,N)=1 ensures maximum contiguous coarray segment
    - Non-coprime pairs create more holes and reduced DOF
    - Common coprime pairs: (2,3), (3,4), (3,5), (4,5), (5,6), (5,7), (7,11)
    
    Coarray Properties
    ------------------
    
    **Expected Aperture:** A ≈ M×N×d (theoretical maximum)
    
    **Expected DOF:** K_max ≈ (M×N - 1)/2 for coprime (M,N)
    
    **Weight Distribution:**
    - More uniform than nested arrays
    - Weight at lag k: Number of sensor pairs with difference k
    - Multiple lags may have same weight (unlike nested's decreasing pattern)
    
    **Holes:**
    - Coprime (M,N): Minimal holes in range [-(MN-1), MN-1]
    - Non-coprime: More holes, fragmented coarray
    
    Hardware Considerations
    -----------------------
    
    1. **Inter-element Spacing:**
       - Subarray 1 (P1): Spacing = M×d between consecutive elements
       - Subarray 2 (P2): Spacing = N×d between consecutive elements
       - Minimum spacing: d × min(M, N)
       - Both larger than λ/2 if M,N > 1 → Reduced mutual coupling
       
    2. **Shared Origin:**
       - Both subarrays include position 0
       - Physical implementation: One sensor at origin serves both subarrays
       - Reduces total sensor count by 1 (M+N-1 instead of M+N)
       
    3. **PCB Layout:**
       - Coprime spacing naturally avoids regular patterns
       - May be easier to route traces compared to nested arrays
       - Consider physical element size when choosing M×d and N×d
       
    4. **Beam Pattern:**
       - Coprime spacing reduces grating lobes in wideband operation
       - Better sidelobe performance than regular arrays in some cases
    
    Examples
    --------
    
    **Example 1: Small TCA (M=2, N=3)**
    
    >>> tca = TCAArrayProcessor(M=2, N=3, d=1.0)
    >>> results = tca.run_full_analysis()
    >>> print(f"Sensors: {tca.total_sensors}")
    Sensors: 4
    >>> print(f"Positions: {results.sensors_positions}")
    Positions: [0. 2. 3. 4.]
    >>> print(f"Aperture: {results.aperture}, L: {results.segment_length}")
    Aperture: 8, L: 5
    
    **Example 2: Medium TCA (M=3, N=5)**
    
    >>> tca = TCAArrayProcessor(M=3, N=5, d=1.0)
    >>> results = tca.run_full_analysis()
    >>> print(f"Sensors: {tca.total_sensors}")
    Sensors: 7
    >>> print(f"Positions: {results.sensors_positions}")
    Positions: [ 0.  3.  5.  6.  9. 10. 12.]
    >>> print(f"Aperture: {results.aperture}, K_max: {results.segment_length // 2}")
    Aperture: 24, K_max: 7
    
    **Example 3: Batch Comparison of Coprime Pairs**
    
    >>> coprime_pairs = [(2,3), (3,4), (3,5), (4,5), (5,7)]
    >>> for M, N in coprime_pairs:
    ...     tca = TCAArrayProcessor(M=M, N=N)
    ...     results = tca.run_full_analysis()
    ...     k_max = results.segment_length // 2
    ...     efficiency = k_max / tca.total_sensors
    ...     print(f"({M},{N}): N={tca.total_sensors}, K_max={k_max}, Eff={efficiency:.2f}")
    (2,3): N=4, K_max=2, Eff=0.50
    (3,4): N=6, K_max=5, Eff=0.83
    (3,5): N=7, K_max=7, Eff=1.00
    (4,5): N=8, K_max=9, Eff=1.12
    (5,7): N=11, K_max=17, Eff=1.55
    
    **Example 4: Compare TCA vs Nested**
    
    >>> # Similar sensor count
    >>> tca = TCAArrayProcessor(M=3, N=5, d=1.0)
    >>> tca_results = tca.run_full_analysis()
    >>> 
    >>> from geometry_processors.nested_processor import NestedArrayProcessor
    >>> nested = NestedArrayProcessor(N1=3, N2=4, d=1.0)
    >>> nested_results = nested.run_full_analysis()
    >>> 
    >>> print(f"TCA: N={tca.total_sensors}, Aperture={tca_results.aperture}")
    >>> print(f"Nested: N={7}, Aperture={nested_results.aperture}")
    TCA: N=7, Aperture=24
    Nested: N=7, Aperture=32
    # Nested has larger aperture but potentially higher coupling
    
    **Example 5: Non-Coprime Warning**
    
    >>> tca = TCAArrayProcessor(M=4, N=6, d=1.0)  # gcd(4,6) = 2, not coprime!
    >>> print(f"Is coprime: {tca.is_coprime}")
    Is coprime: False
    >>> results = tca.run_full_analysis()
    >>> print(f"Holes: {len(results.holes_in_segment)}")
    Holes: >0  # Non-coprime arrays have more holes
    
    See Also
    --------
    NestedArrayProcessor : Standard nested array (higher DOF efficiency)
    DNAArrayProcessor : Dilated nested array (similar spacing flexibility)
    DDNAArrayProcessor : Double dilated nested (maximum flexibility)
    
    References
    ----------
    .. [1] P. Pal and P. P. Vaidyanathan, "Coprime sampling and the MUSIC 
           algorithm," IEEE DSP Workshop, 2011.
    .. [2] P. P. Vaidyanathan and P. Pal, "Sparse sensing with co-prime samplers 
           and arrays," IEEE Trans. Signal Process., vol. 59, no. 2, 2011.
    """
    
    def __init__(self, M: int, N: int, d: float = 1.0):
        """
        Initialize Two-level Coprime Array processor.
        
        Parameters
        ----------
        M : int
            First coprime integer (P2 element count, P1 spacing factor).
        N : int
            Second coprime integer (P1 element count, P2 spacing factor).
        d : float, optional
            Unit spacing (default 1.0).
            
        Raises
        ------
        ValueError
            If M < 2 or N < 2 (minimum requirement for TCA).
        """
        if M < 2 or N < 2:
            raise ValueError(f"TCA requires M ≥ 2 and N ≥ 2, got M={M}, N={N}")
        
        self.M = M
        self.N = N
        self.d = d
        
        # Check coprimality
        self.is_coprime = (np.gcd(M, N) == 1)
        
        # Construct sensor positions
        sensor_positions = self._construct_tca_positions()
        self.total_sensors = len(sensor_positions)
        
        # Initialize base class with required 4 parameters
        super().__init__(
            name=f"TCA (M={M}, N={N})",
            array_type="Two-level Coprime Array",
            sensor_positions=sensor_positions,
            d=d
        )
    
    def _construct_tca_positions(self) -> np.ndarray:
        """
        Construct TCA sensor positions.
        
        Returns
        -------
        np.ndarray
            Sorted unique sensor positions.
            
        Algorithm
        ---------
        1. P1 = d × M × [0, 1, 2, ..., N-1]  (N elements, spacing M×d)
        2. P2 = d × N × [0, 1, 2, ..., M-1]  (M elements, spacing N×d)
        3. Combine and remove duplicates (origin appears in both)
        4. Sort ascending
        
        Note: Origin (0) is shared between subarrays, so total unique
        positions = M + N - 1 instead of M + N.
        """
        # Subarray 1: N elements with spacing M×d
        P1 = self.d * self.M * np.arange(self.N)
        
        # Subarray 2: M elements with spacing N×d
        P2 = self.d * self.N * np.arange(self.M)
        
        # Combine and get unique positions (origin shared)
        combined = np.concatenate([P1, P2])
        unique_positions = np.unique(combined)
        
        return unique_positions
    
    def compute_coarray_positions(self) -> np.ndarray:
        """
        Compute all difference coarray positions.
        
        Returns
        -------
        np.ndarray
            All coarray positions (including duplicates and zero).
            
        Implementation
        --------------
        Computes N² pairwise differences: d_ij = x_i - x_j for all i,j.
        Includes both positive and negative lags, and zero (self-differences).
        """
        positions = self.data.sensors_positions
        N = len(positions)
        
        # Compute all pairwise differences: d_ij = x_i - x_j
        coarray = []
        for i in range(N):
            for j in range(N):
                coarray.append(positions[i] - positions[j])
        
        return np.array(coarray)
    
    def compute_unique_coarray_elements(self) -> np.ndarray:
        """
        Identify unique coarray positions (virtual sensor locations).
        
        Returns
        -------
        np.ndarray
            Sorted unique coarray positions.
            
        Note
        ----
        For coprime (M,N), expect consecutive integers in range [-(MN-1), MN-1].
        For non-coprime, some integers may be missing (holes).
        """
        coarray = self.data.coarray_positions
        unique_coarray = np.unique(coarray)
        return np.sort(unique_coarray)
    
    def compute_virtual_only_elements(self) -> np.ndarray:
        """
        Find virtual-only coarray positions (not in physical array).
        
        Returns
        -------
        np.ndarray
            Virtual sensor positions that don't correspond to physical sensors.
            
        Explanation
        -----------
        Virtual-only elements arise from cross-differences between non-identical
        physical sensors. These extend the effective aperture beyond physical array.
        """
        physical = self.data.sensors_positions
        virtual = self.data.unique_differences
        
        # Virtual-only = all virtual positions not in physical array
        virtual_only = np.setdiff1d(virtual, physical)
        return virtual_only
    
    def compute_coarray_weight_distribution(self) -> pd.DataFrame:
        """
        Calculate weight (multiplicity) for each unique coarray lag.
        
        Returns
        -------
        pd.DataFrame
            Table with columns ['Lag', 'Weight'] sorted by Lag.
            
        Interpretation
        --------------
        Weight w(k) = number of sensor pairs (i,j) with difference x_i - x_j = k.
        Higher weights generally improve DOA estimation accuracy at that lag.
        
        For coprime arrays, weights tend to be more uniform than nested arrays.
        """
        coarray = self.data.coarray_positions
        unique_lags = self.data.unique_differences
        
        # Count occurrences of each lag
        weights = []
        for lag in unique_lags:
            weight = np.sum(coarray == lag)
            weights.append({'Lag': int(lag), 'Weight': int(weight)})
        
        df = pd.DataFrame(weights)
        df = df.sort_values('Lag').reset_index(drop=True)
        return df
    
    def compute_contiguous_virtual_segments(self) -> List[List[int]]:
        """
        Identify contiguous segments in the difference coarray.
        
        Returns
        -------
        List[List[int]]
            List of [start, end] pairs for each contiguous segment.
            Each segment contains consecutive integers with no gaps.
            
        Algorithm
        ---------
        1. Sort unique coarray positions
        2. Find gaps (difference > 1 between consecutive elements)
        3. Record [start, end] of each contiguous run
        
        Note
        ----
        For ideal coprime arrays, expect one large contiguous segment
        covering [-(MN-1), MN-1].
        """
        unique_lags = np.sort(self.data.unique_differences)
        
        if len(unique_lags) == 0:
            return []
        
        segments = []
        start = unique_lags[0]
        prev = start
        
        for lag in unique_lags[1:]:
            if lag - prev > 1:  # Gap detected
                segments.append([int(start), int(prev)])
                start = lag
            prev = lag
        
        # Add final segment
        segments.append([int(start), int(prev)])
        
        return segments
    
    def compute_holes_in_segment(self) -> List[int]:
        """
        Identify missing positions (holes) in the coarray.
        
        Returns
        -------
        List[int]
            Positions that should exist in a contiguous range but are missing.
            
        Algorithm
        ---------
        1. Find full range: [min_lag, max_lag]
        2. Identify all integers in range that are NOT in coarray
        3. These are the holes
        
        Interpretation
        --------------
        Holes indicate missing lags in the virtual array, which can:
        - Reduce effective DOF
        - Create ambiguities in DOA estimation
        - Require spatial smoothing or other interpolation techniques
        
        For coprime (M,N), expect minimal holes in main segment.
        """
        unique_lags = self.data.unique_differences
        
        if len(unique_lags) == 0:
            return []
        
        min_lag = int(np.min(unique_lags))
        max_lag = int(np.max(unique_lags))
        
        # All integers in range
        full_range = set(range(min_lag, max_lag + 1))
        
        # Actual lags
        actual_lags = set(unique_lags.astype(int))
        
        # Holes = expected - actual
        holes = sorted(full_range - actual_lags)
        
        return holes
    
    def generate_performance_summary(self) -> pd.DataFrame:
        """
        Generate comprehensive performance summary table.
        
        Returns
        -------
        pd.DataFrame
            Single-row table with key metrics for this TCA configuration.
            
        Metrics Included
        ----------------
        - Array name and parameters (M, N)
        - Physical sensors (N = M+N-1)
        - Coprimality status
        - Coarray aperture (max lag - min lag)
        - Unique virtual elements
        - Contiguous segment length L
        - Maximum detectable sources K_max = floor(L/2)
        - Number of holes
        - Weights at small lags: w(0), w(1), w(2)
        - DOF efficiency: K_max / N
        """
        segments = self.data.all_contiguous_segments
        
        # Find largest segment and store length
        if segments and len(segments) > 0:
            # segments is a list of [start, end] pairs
            segment_lengths = [int(seg[-1]) - int(seg[0]) + 1 for seg in segments]
            max_idx = np.argmax(segment_lengths)
            L = segment_lengths[max_idx]
            largest_seg = segments[max_idx]
            
            # Store segment_length as attribute for compatibility
            self.data.segment_length = L
        else:
            L = 0
            largest_seg = [0, 0]
            self.data.segment_length = 0
        
        K_max = L // 2
        
        # Extract small-lag weights
        wt = self.data.weight_table
        w0 = int(wt[wt['Lag'] == 0]['Weight'].iloc[0]) if 0 in wt['Lag'].values else 0
        w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
        w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
        
        # Ensure holes_in_segment is stored (if not already computed)
        if not hasattr(self.data, 'holes_in_segment'):
            self.data.holes_in_segment = self.compute_holes_in_segment()
        
        summary = {
            'Array': self.data.name,
            'M': self.M,
            'N': self.N,
            'Is_Coprime': self.is_coprime,
            'Total_Sensors': self.total_sensors,
            'aperture': self.data.aperture,
            'Unique_Lags': len(self.data.unique_differences),
            'Segment_Length_L': L,
            'K_max': K_max,
            'Holes': len(self.data.holes_in_segment),
            'w(0)': w0,
            'w(1)': w1,
            'w(2)': w2,
            'DOF_Efficiency': f"{K_max / self.total_sensors:.2f}" if self.total_sensors > 0 else "0.00",
            'Largest_Segment': f"[{int(largest_seg[0])}:{int(largest_seg[-1])}]"
        }
        
        # Store in self.data and return
        self.data.performance_summary_table = pd.DataFrame([summary])
        return self.data.performance_summary_table
    
    def plot_coarray(self) -> str:
        """
        Generate ASCII visualization of the coarray structure.
        
        Returns
        -------
        str
            Formatted string showing coarray segment and key properties.
            
        Visualization includes:
        - Coprimality status
        - Physical sensor positions
        - Contiguous segment range
        - Hole count and DOF metrics
        """
        output = []
        output.append("=" * 70)
        output.append(f"TCA ARRAY VISUALIZATION: M={self.M}, N={self.N}")
        output.append("=" * 70)
        output.append("")
        
        output.append(f"Coprimality: {'✓ Coprime (gcd=1)' if self.is_coprime else '✗ NOT Coprime'}")
        output.append("")
        
        output.append("Physical Sensor Positions:")
        output.append(f"  P1 (N={self.N}, spacing={self.M}d): {self.d * self.M * np.arange(self.N)}")
        output.append(f"  P2 (M={self.M}, spacing={self.N}d): {self.d * self.N * np.arange(self.M)}")
        output.append(f"  Combined (unique): {self.data.sensors_positions}")
        output.append(f"  Total: {self.total_sensors} sensors (origin shared)")
        output.append("")
        
        output.append("Difference Coarray:")
        output.append(f"  Aperture: {self.data.aperture}")
        output.append(f"  Unique lags: {len(self.data.unique_differences)}")
        
        segments = self.data.all_contiguous_segments
        if segments and len(segments) > 0:
            output.append(f"  Contiguous segments: {segments}")
            segment_lengths = [int(seg[-1]) - int(seg[0]) + 1 for seg in segments]
            L = max(segment_lengths)
            output.append(f"  Largest segment length L: {L}")
        else:
            output.append("  No contiguous segments found")
            L = 0
        
        output.append(f"  Holes: {len(self.data.holes_in_segment)}")
        output.append(f"  K_max (DOF): {L // 2}")
        output.append("")
        
        # Small-lag weights
        wt = self.data.weight_table
        output.append("Weight Distribution (first 10 positive lags):")
        positive_lags = wt[wt['Lag'] > 0].head(10)
        for _, row in positive_lags.iterrows():
            output.append(f"  w({int(row['Lag'])}) = {int(row['Weight'])}")
        
        output.append("=" * 70)
        
        return "\n".join(output)
    
    def __repr__(self) -> str:
        """String representation of TCA array."""
        return (f"TCAArrayProcessor(M={self.M}, N={self.N}, "
                f"total_sensors={self.total_sensors}, is_coprime={self.is_coprime})")


def compare_tca_arrays(configs: List[Tuple[int, int]], d: float = 1.0) -> pd.DataFrame:
    """
    Compare multiple TCA configurations side-by-side.
    
    Parameters
    ----------
    configs : List[Tuple[int, int]]
        List of (M, N) pairs to compare.
    d : float, optional
        Unit spacing (default 1.0).
        
    Returns
    -------
    pd.DataFrame
        Comparison table with all configurations.
        
    Example
    -------
    >>> coprime_pairs = [(2,3), (3,4), (3,5), (4,5), (5,7)]
    >>> comparison = compare_tca_arrays(coprime_pairs)
    >>> print(comparison.to_markdown(index=False))
    """
    results = []
    
    for M, N in configs:
        try:
            tca = TCAArrayProcessor(M=M, N=N, d=d)
            analysis = tca.run_full_analysis()
            results.append(analysis.performance_summary_table.iloc[0])
        except Exception as e:
            print(f"Error processing ({M},{N}): {e}")
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()
