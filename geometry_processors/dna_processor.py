"""
DNA Array Processor (Dilated Nested Array)
===========================================

Implements the Dilated Nested Array (DNA) geometry from:
Liu and Vaidyanathan, "Super Nested Arrays: Linear Sparse Arrays with Reduced 
Mutual Coupling—Part I: Fundamentals," IEEE Trans. Signal Processing, 2016.

Overview:
---------
DNA is a two-subarray sparse array configuration that extends standard nested
arrays by introducing a dilation factor D. This design parameter provides explicit
control over inter-sensor spacing, enabling reduced mutual coupling while maintaining
excellent difference coarray properties for DOA estimation.

Construction Formula:
--------------------
DNA consists of two subarrays with a dilation factor D applied to the second subarray:

- **Subarray P1** (dense consecutive): d × [1, 2, 3, ..., N1]
  - N1 sensors with unit spacing d (typically d = λ/2)
  - Provides dense sampling at the origin
  - Creates high-weight small lags in difference coarray
  - Positioned at: [d, 2d, 3d, ..., N1·d]
  
- **Subarray P2** (dilated sparse): d × (N1+1) × D × [1, 2, ..., N2]
  - N2 sensors with spacing (N1+1) × D × d
  - Dilation factor D controls inter-element spacing
  - Creates extended virtual aperture through cross-subarray differences
  - Positioned at: [(N1+1)·D·d, 2(N1+1)·D·d, ..., N2(N1+1)·D·d]

**Total sensors**: N = N1 + N2
**Physical aperture**: (N1 + N2(N1+1)D - 1) × d
**Coarray aperture**: Typically 2 × Physical aperture

Construction Example (N1=3, N2=2, D=2, d=1):
  P1 = [1, 2, 3]
  P2 = (3+1) × 2 × [1, 2] = 8 × [1, 2] = [8, 16]
  Combined: [1, 2, 3, 8, 16]
  Zero-based: [0, 1, 2, 7, 15]

Mathematical Insight:
--------------------
The dilation factor D provides a design parameter to control three key aspects:

1. **Mutual Coupling Reduction**: 
   - Larger D increases minimum inter-sensor spacing
   - Minimum spacing in P2: (N1+1) × D × d
   - D ≥ 2 typically ensures spacing ≥ λ for reduced coupling
   - Trade-off: May reduce contiguous segment length L

2. **Aperture Extension**:
   - Physical aperture grows as O(D · N1 · N2)
   - Virtual aperture extends further through difference coarray
   - Larger D creates longer-range virtual sensors
   - Benefit: Higher angular resolution potential

3. **Coarray Structure**:
   - D = 1: Dense coarray coverage, maximum L (standard nested)
   - D = 2: Moderate aperture extension, good balance
   - D ≥ 3: Large aperture but may introduce coarray holes
   - Design choice depends on application requirements

Performance Characteristics:
---------------------------
- **Aperture Growth**: A ≈ 2D(N1+1)N2 (two-sided coarray span)
- **Segment Length**: 
  - D = 1: L ≈ 2N (standard nested), maximum DOF
  - D = 2: L ≈ N to 1.5N, good balance
  - D ≥ 3: L may reduce due to holes
- **Degrees of Freedom**: K_max = ⌊L/2⌋ detectable sources
- **Weight Distribution**:
  - High w(1) from dense P1 subarray
  - Cross-subarray differences fill intermediate lags
  - Better low-lag weights than uniform linear arrays
- **Hardware Advantages**:
  - Reduced mutual coupling (D > 1)
  - Simpler calibration than random sparse arrays
  - Predictable coarray structure

Comparison with Related Arrays:
------------------------------
- **vs Standard Nested (D=1)**: DNA generalizes nested with D parameter
- **vs DDNA**: DDNA applies dilation to both subarrays (two D factors)
- **vs SNA3**: SNA3 uses three subarrays, DNA uses two
- **vs ULA**: DNA achieves O(N²) virtual sensors vs O(N) for ULA

Typical Use Cases:
-----------------
- **D = 1**: Maximum DOF applications (standard nested array)
- **D = 2**: Balanced DOF and mutual coupling reduction
- **D = 3**: When hardware constraints require large spacing
- **Variable D**: Parameter studies and optimization

Reference:
---------
[11] C. L. Liu and P. P. Vaidyanathan, "Super nested arrays: Linear sparse 
     arrays with reduced mutual coupling—Part I: Fundamentals," IEEE Trans. 
     Signal Processing, vol. 64, no. 15, pp. 3997-4012, Aug. 2016.

Implementation Notes:
--------------------
- All positions stored as zero-based integers after construction
- Dilation factor D must be positive integer (D ≥ 1)
- Base spacing d typically set to λ/2 (half-wavelength)
- Follows BaseArrayProcessor standardized 7-step pipeline

Author: Hossein (RadarPy Project)
Date: November 7, 2025
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from .bases_classes import BaseArrayProcessor


class DNAArrayProcessor(BaseArrayProcessor):
    """
    Dilated Nested Array (DNA) Processor
    
    Implements two-subarray nested configuration with tunable dilation factor
    for enhanced mutual coupling characteristics and flexible aperture control.
    
    The DNA extends the standard nested array concept by introducing a dilation
    parameter D that explicitly controls the spacing of the second (sparse) subarray.
    This design provides better hardware implementation characteristics (reduced
    mutual coupling) while maintaining excellent coarray properties for DOA estimation.
    
    Parameters
    ----------
    N1 : int
        Number of sensors in first subarray (dense consecutive), N1 ≥ 1
        - Larger N1 increases weight at small lags
        - Typical range: 3 to 10
    N2 : int
        Number of sensors in second subarray (dilated sparse), N2 ≥ 1
        - Larger N2 extends virtual aperture
        - Typical range: 2 to 8
        - Common ratio: N1 ≈ N2 or N1 slightly larger
    D : int, optional
        Dilation factor for second subarray spacing, D ≥ 1 (default: 1)
        - D=1: Standard nested array (maximum DOF)
        - D=2: Balanced design (good DOF, reduced coupling)
        - D=3+: Large spacing (minimal coupling, may have holes)
        - Design guideline: Choose D such that (N1+1)·D ≥ 2 wavelengths
    d : float, optional
        Base sensor spacing multiplier in wavelengths (default: 1.0 = λ/2)
        - Standard: d=1.0 corresponds to λ/2 physical spacing
        - Can be adjusted for specific frequency/wavelength
    
    Attributes
    ----------
    N1 : int
        First subarray size (dense consecutive)
    N2 : int
        Second subarray size (dilated sparse)
    D : int
        Dilation factor (spacing multiplier for P2)
    total_sensors : int
        Total number of physical sensors = N1 + N2
    data : ArraySpec
        Container with all analysis results (47 attributes):
        - sensors_positions: Physical sensor locations
        - all_differences: Full N² difference coarray
        - unique_differences: Unique virtual sensor positions
        - weight_table: Lag multiplicity distribution
        - contiguous_segments: Hole-free coarray segments
        - segment_length: Length L of longest segment
        - coarray_aperture: Two-sided virtual array span
        - holes_in_segment: Missing lags within segment
        - performance_summary_table: Comprehensive metrics
    
    Notes
    -----
    **Design Trade-offs**:
    
    1. Dilation Factor Selection:
       - D = 1: Maximum K_max, potential mutual coupling issues
       - D = 2: Good balance, typical choice for practical systems
       - D ≥ 3: Minimal coupling, but reduced K_max
    
    2. Subarray Size Ratio (N1/N2):
       - N1 > N2: Higher weight at small lags, better for correlated sources
       - N1 ≈ N2: Balanced aperture and DOF
       - N1 < N2: Extended aperture, useful for high-resolution needs
    
    3. Total Sensors (N = N1 + N2):
       - Physical aperture: O(D·N1·N2·d)
       - Virtual aperture: O(2·D·N1·N2·d)
       - DOF scaling: K_max ≈ N/2 for D=1, may reduce for D>1
    
    **Coarray Properties**:
    
    - Dense coarray near origin (from P1 self-differences)
    - Extended coarray tails (from P2 and cross-differences)
    - Zero holes for D=1 (standard nested)
    - Possible holes for D≥2 depending on N1, N2
    - Weight w(1) = N1 - 1 (decreases by 1 as D increases typically)
    
    **Hardware Considerations**:
    
    - Minimum spacing in P1: d (typically λ/2)
    - Minimum spacing in P2: (N1+1)·D·d
    - Inter-subarray gap: [(N1+1)D - N1]·d ≈ (N1·D + D - N1)·d
    - Choose D to ensure P2 spacing ≥ λ for negligible coupling
    
    Examples
    --------
    **Basic Usage - Standard Nested (D=1)**:
    
    >>> processor = DNAArrayProcessor(N1=4, N2=3, D=1)
    >>> results = processor.run_full_analysis()
    >>> print(f"Positions: {processor.data.sensors_positions}")
    Positions: [0, 1, 2, 3, 4, 9, 14]
    >>> print(f"Aperture: {results.coarray_aperture}, L: {results.segment_length}")
    Aperture: 28, L: 29
    
    **Dilated Design (D=2) for Reduced Coupling**:
    
    >>> processor = DNAArrayProcessor(N1=5, N2=4, D=2)
    >>> results = processor.run_full_analysis()
    >>> print(f"N: {processor.total_sensors}, K_max: {results.segment_length // 2}")
    N: 9, K_max: 4
    >>> print(f"Physical aperture: {max(processor.data.sensors_positions)}")
    Physical aperture: 47
    
    **Parameter Study - Vary Dilation**:
    
    >>> for D in [1, 2, 3]:
    ...     proc = DNAArrayProcessor(N1=4, N2=3, D=D)
    ...     res = proc.run_full_analysis()
    ...     print(f"D={D}: A={res.coarray_aperture}, L={res.segment_length}, "
    ...           f"Holes={len(res.holes_in_segment)}")
    D=1: A=28, L=29, Holes=0
    D=2: A=58, L=7, Holes=0
    D=3: A=88, L=7, Holes=0
    
    **Batch Comparison Using Utility Function**:
    
    >>> from geometry_processors.dna_processor import compare_dna_arrays
    >>> comparison = compare_dna_arrays(
    ...     N1_values=[3, 4, 5, 6],
    ...     N2_values=[3, 3, 4, 5],
    ...     D_values=[2, 2, 2, 2]
    ... )
    >>> print(comparison[['Total_N', 'Coarray_Aperture', 'L', 'K_max']])
       Total_N  Coarray_Aperture   L  K_max
             6                46   5      2
             7                58   7      3
             9                94   9      4
            11               138  11      5
    
    **Accessing Performance Metrics**:
    
    >>> processor = DNAArrayProcessor(N1=6, N2=5, D=2)
    >>> results = processor.run_full_analysis()
    >>> print(results.performance_summary_table.to_markdown(index=False))
    | Metric                       | Value                     |
    |:-----------------------------|:--------------------------|
    | Array Type                   | DNA (Dilated Nested Array)|
    | N1 (Dense Subarray)          | 6                         |
    | N2 (Dilated Sparse Subarray) | 5                         |
    | Dilation Factor D            | 2                         |
    | Total Sensors (N)            | 11                        |
    | ...                          | ...                       |
    
    **Visualization**:
    
    >>> processor.plot_coarray()  # Console-based visualization
    ======================================================================
    DNA ARRAY VISUALIZATION: N1=6, N2=5, D=2
    ======================================================================
    Physical Sensor Positions:
      P1 (Dense):  [0 1 2 3 4 5]
      P2 (Dilated Sparse): [13 27 41 55 69]
      Total: 11 sensors
    ...
    
    See Also
    --------
    NestedArrayProcessor : Standard nested array (DNA with D=1)
    DDNAArrayProcessor : Double dilated nested array (two dilation factors)
    SNA3ArrayProcessor : Super nested array variant (three subarrays)
    ANAII2ArrayProcessor : Augmented nested array II-2 (three subarrays with bridge)
    compare_dna_arrays : Utility function for batch comparison
    
    References
    ----------
    .. [1] C. L. Liu and P. P. Vaidyanathan, "Super nested arrays: Linear sparse 
           arrays with reduced mutual coupling—Part I: Fundamentals," IEEE Trans. 
           Signal Processing, vol. 64, no. 15, pp. 3997-4012, Aug. 2016.
    .. [2] P. P. Vaidyanathan and P. Pal, "Sparse sensing with co-prime samplers 
           and arrays," IEEE Trans. Signal Processing, vol. 59, no. 2, pp. 573-586, 
           Feb. 2011.
    """
    
    def __init__(self, N1: int, N2: int, D: int = 1, d: float = 1.0):
        """
        Initialize DNA Array Processor.
        
        Parameters
        ----------
        N1 : int
            First subarray size (dense), must be ≥ 1
        N2 : int
            Second subarray size (dilated sparse), must be ≥ 1
        D : int
            Dilation factor, must be ≥ 1
        d : float
            Base spacing multiplier (wavelengths)
        """
        if N1 < 1 or N2 < 1:
            raise ValueError(f"N1 and N2 must be ≥ 1, got N1={N1}, N2={N2}")
        if D < 1:
            raise ValueError(f"Dilation factor D must be ≥ 1, got D={D}")
        
        self.N1 = N1
        self.N2 = N2
        self.D = D
        self.total_sensors = N1 + N2
        
        # Construct sensor positions
        positions = self._construct_dna_pattern(N1, N2, D, d)
        
        # Initialize base processor
        super().__init__(
            name=f"DNA Array (N1={N1}, N2={N2}, D={D})",
            array_type="Dilated Nested Array (DNA)",
            sensor_positions=positions,
            d=d
        )
    
    def _construct_dna_pattern(self, N1: int, N2: int, D: int, d: float) -> np.ndarray:
        """
        Construct DNA sensor positions using two-subarray dilated configuration.
        
        This method implements the DNA construction formula with dilation factor
        D applied to the second subarray for mutual coupling control.
        
        Construction Algorithm:
        ----------------------
        1. **Subarray P1** (dense at origin):
           P1 = d × [1, 2, 3, ..., N1]
           
        2. **Subarray P2** (dilated sparse):
           P2 = d × (N1+1) × D × [1, 2, ..., N2]
                = [d·(N1+1)·D, d·(N1+1)·D·2, ..., d·(N1+1)·D·N2]
           
        3. Combine subarrays: Positions = P1 ∪ P2
        
        4. Convert to zero-based by subtracting minimum position
        
        Dilation Rationale:
        ------------------
        - **D = 1**: Standard nested array, minimum sensor spacing = (N1+1)·d
        - **D > 1**: Increased spacing = (N1+1)·D·d, reduces mutual coupling
        - Trade-off: Larger D extends aperture but may create more coarray holes
        
        Parameters
        ----------
        N1 : int
            First subarray size (dense consecutive)
        N2 : int
            Second subarray size (dilated sparse)
        D : int
            Dilation factor (≥ 1)
        d : float
            Base spacing multiplier
        
        Returns
        -------
        np.ndarray
            Zero-based integer sensor positions, sorted
        
        Examples
        --------
        >>> # Standard nested (D=1): N1=3, N2=2
        >>> positions = _construct_dna_pattern(3, 2, 1, 1.0)
        >>> print(positions)
        [0 1 2 4 8]  # P1=[1,2,3], P2=[4,8], offset=4
        
        >>> # Dilated (D=2): N1=3, N2=2
        >>> positions = _construct_dna_pattern(3, 2, 2, 1.0)
        >>> print(positions)
        [0 1 2 8 16]  # P1=[1,2,3], P2=[8,16], offset=8
        """
        # Subarray P1: dense consecutive sensors
        P1 = d * np.arange(1, N1 + 1)
        
        # Subarray P2: dilated sparse sensors
        offset_P2 = (N1 + 1) * D  # Dilation applied to offset
        P2 = d * offset_P2 * np.arange(1, N2 + 1)
        
        # Combine and normalize to zero-based
        positions = np.concatenate([P1, P2])
        positions = positions - np.min(positions)
        
        return np.sort(positions).astype(int)
    
    def compute_array_spacing(self) -> float:
        """
        Compute the base spacing of the DNA array.
        
        For DNA, this returns the unit spacing d used in the construction.
        The actual inter-sensor spacing varies:
        - Within P1: spacing = d
        - Within P2: spacing = (N1+1)·D·d
        - Between P1 and P2: minimum spacing = (N1+1)·D·d - N1·d
        
        Returns
        -------
        float
            Base spacing multiplier d
        """
        # DNA uses base spacing d; actual spacing varies by subarray
        positions = np.array(self.data.sensors_positions)
        return float(np.min(np.diff(np.sort(positions))))
    
    def compute_all_differences(self) -> None:
        """
        Compute all pairwise differences (difference coarray).
        
        The DNA difference coarray consists of:
        
        1. **Within P1**: Differences between dense sensors
           - Small positive lags: 1, 2, ..., N1-1
           - High weight at small lags
        
        2. **Within P2**: Differences between dilated sparse sensors
           - Large positive lags: (N1+1)·D, 2·(N1+1)·D, ...
           - Extends aperture significantly
        
        3. **Cross P1-P2**: Differences between subarrays
           - Fills intermediate lags
           - Critical for contiguous segment formation
        
        The dilation factor D affects the distribution of differences:
        - D = 1: Dense coarray coverage (standard nested)
        - D > 1: Extended aperture with potential holes
        
        Stores in self.data.all_differences as sorted list of all N² differences.
        """
        positions = np.array(self.data.sensors_positions)
        N = len(positions)
        
        # Compute all pairwise differences (including negatives and zero)
        differences = []
        for i in range(N):
            for j in range(N):
                differences.append(int(positions[i] - positions[j]))
        
        # Sort and store all differences
        self.data.all_differences = sorted(differences)
    
    def analyze_coarray(self) -> None:
        """
        Analyze the difference coarray structure.
        
        Identifies key coarray properties:
        - **Unique differences**: All distinct virtual sensor positions (lags)
        - **Coarray positions**: Both positive and negative lags
        - **Virtual-only positions**: Lags not in physical array
        - **Coarray aperture**: Total span of virtual array
        
        For DNA arrays, expect:
        - Dense virtual sensors near zero (from P1 differences)
        - Extended virtual aperture (from P2 differences)
        - Good contiguous segment for D ≤ 2
        - Possible holes for larger D values
        
        The dilation factor D controls the trade-off between aperture
        extension and coarray hole formation.
        
        Stores results in self.data:
        - unique_differences: List of unique lags
        - coarray_positions: All positive and negative lags
        - virtual_only_positions: Lags not in physical array
        - coarray_aperture: max_lag - min_lag
        """
        all_diffs = self.data.all_differences
        unique = sorted(set(all_diffs))
        
        # Identify virtual-only positions (not in physical array)
        physical_set = set(self.data.sensors_positions)
        virtual_only = [lag for lag in unique if lag not in physical_set and lag != 0]
        
        # Coarray aperture (two-sided span)
        aperture = max(unique) - min(unique)
        
        # Store results
        self.data.unique_differences = unique
        self.data.coarray_positions = unique
        self.data.virtual_only_positions = sorted(virtual_only)
        self.data.coarray_aperture = aperture
    
    def compute_weight_distribution(self) -> None:
        """
        Compute weight (multiplicity) of each lag in the difference coarray.
        
        Weight w(m) = number of sensor pairs (i,j) such that n_i - n_j = m
        
        For DNA arrays, the weight distribution has characteristic features:
        - **High weights at small lags**: From dense P1 subarray
        - **Lower weights at large lags**: From sparse P2 subarray
        - **Intermediate weights**: From cross-subarray differences
        
        Higher weights at small lags (w(1), w(2), etc.) generally improve
        DOA estimation accuracy. DNA with D=1 (standard nested) typically
        achieves the highest small-lag weights.
        
        Stores self.data.weight_table as pandas DataFrame with columns:
        - Lag: The difference value (virtual sensor position)
        - Weight: Frequency of this lag in the difference coarray
        """
        from collections import Counter
        
        # Count frequency of each difference
        diff_counts = Counter(self.data.all_differences)
        
        # Create weight table sorted by lag
        weight_data = [
            {"Lag": lag, "Weight": count}
            for lag, count in sorted(diff_counts.items())
        ]
        
        self.data.weight_table = pd.DataFrame(weight_data)
    
    def analyze_contiguous_segments(self) -> None:
        """
        Find the longest contiguous segment centered at zero lag.
        
        The contiguous segment is the longest sequence of consecutive integer
        lags without holes, starting from zero. This determines the maximum
        number of sources that can be estimated: K_max = floor(L/2).
        
        For DNA arrays:
        - D = 1: Typically achieves L ≈ 2N (standard nested)
        - D = 2: Often achieves L ≈ N1 + N2 (slight reduction)
        - D ≥ 3: May have reduced L due to coarray holes
        
        The dilation factor trades off aperture extension for segment length.
        
        Stores in self.data:
        - contiguous_segments: List of [start, end] for each segment
        - segment_length: Length L of longest segment (integer)
        """
        unique_diffs = self.data.unique_differences
        
        if 0 not in unique_diffs:
            self.data.contiguous_segments = []
            self.data.segment_length = 0
            return
        
        # Find longest contiguous segment starting from 0
        max_positive = 0
        for lag in unique_diffs:
            if lag >= 0 and lag == max_positive:
                max_positive += 1
            elif lag > max_positive:
                break
        
        max_negative = 0
        for lag in reversed(unique_diffs):
            if lag <= 0 and lag == -max_negative:
                max_negative += 1
            elif lag < -max_negative:
                break
        
        # Segment spans from -max_negative to +max_positive
        segment_start = -(max_negative - 1)
        segment_end = max_positive - 1
        segment_length = segment_end - segment_start + 1
        
        self.data.contiguous_segments = [[segment_start, segment_end]]
        self.data.segment_length = segment_length
    
    def analyze_holes(self) -> None:
        """
        Identify holes (missing lags) in the contiguous segment.
        
        A hole is an integer lag within the ideal range [-L/2, L/2] that
        does not appear in the difference coarray. Holes can reduce the
        effective degrees of freedom for DOA estimation.
        
        For DNA arrays:
        - D = 1: Typically zero or very few holes (standard nested)
        - D = 2: May have some holes depending on N1/N2 ratio
        - D ≥ 3: More likely to have holes
        
        The number of holes indicates coarray quality. Fewer holes generally
        mean better DOA estimation performance.
        
        Stores self.data.holes_in_segment as list of missing integer lags.
        """
        if not self.data.contiguous_segments:
            self.data.holes_in_segment = []
            return
        
        segment = self.data.contiguous_segments[0]
        start, end = segment[0], segment[1]
        
        # Expected lags in contiguous segment
        expected = set(range(start, end + 1))
        
        # Actual lags present
        actual = set(self.data.unique_differences)
        
        # Holes are expected but missing
        holes = sorted(expected - actual)
        
        self.data.holes_in_segment = holes
    
    def generate_performance_summary(self) -> None:
        """
        Generate comprehensive performance metrics summary.
        
        Creates a detailed table with key DNA array characteristics:
        - Physical array properties (N, N1, N2, D, aperture)
        - Virtual array properties (unique lags, segment length)
        - Performance metrics (K_max, holes)
        - Weight distribution (w at small lags)
        
        This summary enables comparison between different DNA configurations
        and other array types.
        
        Stores self.data.performance_summary_table as pandas DataFrame.
        """
        # Get weight at specific lags
        wt = self.data.weight_table
        w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
        w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
        w3 = int(wt[wt['Lag'] == 3]['Weight'].iloc[0]) if 3 in wt['Lag'].values else 0
        
        # Calculate metrics
        L = self.data.segment_length
        K_max = L // 2
        num_holes = len(self.data.holes_in_segment) if hasattr(self.data, 'holes_in_segment') else 0
        
        # Physical aperture (max - min sensor position)
        positions = np.array(self.data.sensors_positions)
        phys_aperture = int(np.max(positions) - np.min(positions))
        
        summary_data = [
            {"Metric": "Array Type", "Value": f"DNA (Dilated Nested Array)"},
            {"Metric": "N1 (Dense Subarray)", "Value": f"{self.N1}"},
            {"Metric": "N2 (Dilated Sparse Subarray)", "Value": f"{self.N2}"},
            {"Metric": "Dilation Factor D", "Value": f"{self.D}"},
            {"Metric": "Total Sensors (N)", "Value": f"{self.total_sensors}"},
            {"Metric": "Physical Aperture", "Value": f"{phys_aperture}"},
            {"Metric": "Coarray Aperture (A)", "Value": f"{self.data.coarray_aperture}"},
            {"Metric": "Unique Lags", "Value": f"{len(self.data.unique_differences)}"},
            {"Metric": "Virtual-Only Positions", "Value": f"{len(self.data.virtual_only_positions)}"},
            {"Metric": "Contiguous Segment Length (L)", "Value": f"{L}"},
            {"Metric": "Max Detectable Sources (K_max)", "Value": f"{K_max}"},
            {"Metric": "Holes in Segment", "Value": f"{num_holes}"},
            {"Metric": "Weight at Lag 1", "Value": f"{w1}"},
            {"Metric": "Weight at Lag 2", "Value": f"{w2}"},
            {"Metric": "Weight at Lag 3", "Value": f"{w3}"},
        ]
        
        self.data.performance_summary_table = pd.DataFrame(summary_data)
    
    def plot_coarray(self) -> None:
        """
        Visualize the DNA array and its difference coarray (console output).
        
        Displays:
        1. Physical sensor positions (P1 and P2 marked separately)
        2. Difference coarray positions (virtual sensors)
        3. Weight distribution at key lags
        
        This provides quick visual verification of the DNA construction
        and its coarray properties.
        """
        print("\n" + "="*70)
        print(f"DNA ARRAY VISUALIZATION: N1={self.N1}, N2={self.N2}, D={self.D}")
        print("="*70)
        
        # Physical array
        print("\nPhysical Sensor Positions:")
        positions = np.array(self.data.sensors_positions)
        print(f"  P1 (Dense):  {positions[:self.N1]}")
        print(f"  P2 (Dilated Sparse): {positions[self.N1:]}")
        print(f"  Total: {len(positions)} sensors")
        
        # Coarray
        print("\nDifference Coarray:")
        print(f"  Aperture: {self.data.coarray_aperture}")
        print(f"  Unique lags: {len(self.data.unique_differences)}")
        print(f"  Contiguous segment: {self.data.contiguous_segments}")
        print(f"  Segment length L: {self.data.segment_length}")
        print(f"  Holes: {len(self.data.holes_in_segment)}")
        
        # Weight distribution
        wt = self.data.weight_table
        print("\nWeight Distribution (first 10 positive lags):")
        positive_lags = wt[wt['Lag'] > 0].head(10)
        for _, row in positive_lags.iterrows():
            print(f"  w({int(row['Lag'])}) = {int(row['Weight'])}")
        
        print("="*70 + "\n")
    
    def __repr__(self) -> str:
        """String representation of DNA array."""
        return (f"DNAArrayProcessor(N1={self.N1}, N2={self.N2}, D={self.D}, "
                f"total_sensors={self.total_sensors})")


def compare_dna_arrays(N1_values: List[int], 
                       N2_values: List[int],
                       D_values: List[int],
                       d: float = 1.0) -> pd.DataFrame:
    """
    Compare multiple DNA configurations side-by-side.
    
    This utility function analyzes multiple DNA arrays with different
    (N1, N2, D) parameters and compiles their performance metrics into a
    comparison table.
    
    Useful for:
    - Parameter selection (choosing optimal N1/N2/D combination)
    - Dilation factor studies (comparing D=1,2,3,...)
    - Trade-off analysis between aperture, DOF, and mutual coupling
    
    Parameters
    ----------
    N1_values : List[int]
        List of N1 values (first subarray sizes) to test
    N2_values : List[int]
        List of N2 values (second subarray sizes) to test
        Must have same length as N1_values
    D_values : List[int]
        List of dilation factors to test
        Must have same length as N1_values
    d : float, optional
        Base spacing multiplier for all arrays (default: 1.0)
    
    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        - N1, N2, D: Configuration parameters
        - Total_N: Total physical sensors (N1 + N2)
        - Phys_Aperture: Physical array span
        - Coarray_Aperture: Two-sided coarray span
        - Unique_Lags: Number of unique virtual sensors
        - L: Contiguous segment length
        - K_max: Maximum detectable sources
        - Holes: Number of missing lags in segment
        - w(1): Weight at lag 1 (higher is better)
    
    Raises
    ------
    ValueError
        If input lists have different lengths
    
    Examples
    --------
    >>> # Compare dilation factors for fixed N1, N2
    >>> comparison = compare_dna_arrays(
    ...     N1_values=[4, 4, 4],
    ...     N2_values=[3, 3, 3],
    ...     D_values=[1, 2, 3],
    ...     d=1.0
    ... )
    >>> print(comparison.to_markdown(index=False))
    >>> 
    >>> # Compare array sizes for fixed dilation
    >>> comparison = compare_dna_arrays(
    ...     N1_values=[3, 4, 5, 6],
    ...     N2_values=[2, 3, 4, 5],
    ...     D_values=[2, 2, 2, 2]
    ... )
    >>> print(comparison[['Total_N', 'Coarray_Aperture', 'L', 'K_max']])
    """
    if not (len(N1_values) == len(N2_values) == len(D_values)):
        raise ValueError("N1_values, N2_values, and D_values must have same length")
    
    results = []
    
    for N1, N2, D in zip(N1_values, N2_values, D_values):
        processor = DNAArrayProcessor(N1=N1, N2=N2, D=D, d=d)
        data = processor.run_full_analysis()
        
        # Extract metrics
        L = data.segment_length
        K_max = L // 2
        num_holes = len(data.holes_in_segment) if hasattr(data, 'holes_in_segment') and data.holes_in_segment else 0
        
        # Get w(1)
        wt = data.weight_table
        w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
        
        # Physical aperture
        positions = np.array(data.sensors_positions)
        phys_aperture = int(np.max(positions) - np.min(positions))
        
        results.append({
            'N1': N1,
            'N2': N2,
            'D': D,
            'Total_N': N1 + N2,
            'Phys_Aperture': phys_aperture,
            'Coarray_Aperture': data.coarray_aperture,
            'Unique_Lags': len(data.unique_differences),
            'L': L,
            'K_max': K_max,
            'Holes': num_holes,
            'w(1)': w1
        })
    
    return pd.DataFrame(results)
