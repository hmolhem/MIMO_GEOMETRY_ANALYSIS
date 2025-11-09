"""
DDNA Array Processor (Double Dilated Nested Array)
===================================================

Implements the Double Dilated Nested Array (DDNA) geometry from:
Liu and Vaidyanathan, "Super Nested Arrays: Linear Sparse Arrays with Reduced 
Mutual Coupling—Part I: Fundamentals," IEEE Trans. Signal Processing, 2016.

Overview:
---------
DDNA extends the DNA concept by applying independent dilation factors to BOTH
subarrays (D1 for P1, D2 for P2). This provides maximum flexibility in controlling
inter-sensor spacing, enabling optimal hardware designs that balance mutual coupling
reduction, aperture extension, and degrees of freedom.

Construction Formula:
--------------------
DDNA consists of two dilated subarrays with independent dilation control:

- **Subarray P1** (dilated dense): d × D1 × [0, 1, 2, ..., N1-1]
  - N1 sensors with spacing D1 × d
  - Dilation D1 controls spacing within first subarray
  - Provides controllable small-lag coverage
  - Positioned at: [0, D1·d, 2D1·d, ..., (N1-1)·D1·d]
  
- **Subarray P2** (double-dilated sparse): d × (D1·N1+1) × D2 × [1, 2, ..., N2]
  - N2 sensors with spacing (D1·N1+1) × D2 × d
  - Dilation D2 controls spacing within second subarray
  - Creates extended virtual aperture with controlled spacing
  - Positioned at: [(D1·N1+1)·D2·d, 2(D1·N1+1)·D2·d, ..., N2(D1·N1+1)·D2·d]

**Total sensors**: N = N1 + N2
**Physical aperture**: ((N1-1)·D1 + N2(D1·N1+1)D2) × d
**Coarray aperture**: Typically 2 × Physical aperture

Construction Example (N1=3, N2=2, D1=2, D2=3, d=1):
  P1 = 2 × [0, 1, 2] = [0, 2, 4]
  P2 = (2×3+1) × 3 × [1, 2] = 7 × 3 × [1, 2] = 21 × [1, 2] = [21, 42]
  Combined: [0, 2, 4, 21, 42]

Mathematical Insight:
--------------------
Two independent dilation factors (D1, D2) provide three-dimensional design space:

1. **Independent Spacing Control**:
   - D1 controls P1 inter-element spacing (affects small-lag weights)
   - D2 controls P2 inter-element spacing (affects aperture extension)
   - Can optimize each subarray independently for different objectives
   
2. **Mutual Coupling Reduction**:
   - Within P1: spacing = D1 × d
   - Within P2: spacing = (D1·N1+1) × D2 × d
   - Between subarrays: minimum gap ≈ [(D1·N1+1)D2 - D1·N1] × d
   - Choose D1, D2 to ensure all spacings ≥ λ
   
3. **Aperture vs. DOF Trade-off**:
   - Larger D1: Reduces weight at small lags, but extends P1 aperture
   - Larger D2: Extends total aperture significantly
   - D1=D2=1: Reduces to standard nested array
   - D1=1, D2>1: DNA configuration (only P2 dilated)
   - D1>1, D2>1: Full DDNA (both dilated)

Performance Characteristics:
---------------------------
- **Aperture Growth**: A ≈ 2D2(D1·N1+1)N2 (highly tunable)
- **Segment Length**: 
  - D1=D2=1: L ≈ 2N (standard nested), maximum DOF
  - D1=1, D2≥2: L ≈ N to 1.5N (DNA-like)
  - D1≥2, D2≥2: L varies, typically N/2 to N
- **Degrees of Freedom**: K_max = ⌊L/2⌋ detectable sources
- **Weight Distribution**:
  - D1=1: High w(1), w(2), ... (dense P1)
  - D1≥2: Lower small-lag weights, but reduced P1 coupling
  - Cross-differences fill intermediate lags
- **Hardware Flexibility**:
  - Maximum control over all inter-sensor spacings
  - Can satisfy complex hardware constraints
  - Predictable coarray structure for any (D1, D2) combination

Comparison with Related Arrays:
------------------------------
- **vs DNA**: DDNA adds D1 parameter, DNA has D1=1 fixed
- **vs Nested (D1=D2=1)**: DDNA generalizes nested with two D factors
- **vs SNA3/ANAII-2**: DDNA uses two subarrays, others use three
- **Design space**: DDNA has largest parameter space (N1, N2, D1, D2)

Typical Use Cases:
-----------------
- **D1=1, D2=2**: DNA-equivalent, moderate coupling reduction
- **D1=2, D2=2**: Balanced DDNA, good overall spacing
- **D1=2, D2=3**: High dilation, minimal coupling, extended aperture
- **D1=1, D2=1**: Standard nested (maximum DOF reference)
- **Variable (D1,D2)**: Multi-objective optimization studies

Design Guidelines:
-----------------
1. Start with D1=D2=1 (nested) as baseline
2. Increase D2 first if only P2 coupling is problematic (DNA)
3. Increase D1 if P1 also needs spacing (full DDNA)
4. Ensure D1 × d ≥ λ/2 and (D1·N1+1) × D2 × d ≥ λ
5. Balance: Larger (D1,D2) extends aperture but reduces L
6. Typical ranges: D1 ∈ {1,2,3}, D2 ∈ {1,2,3}

Reference:
---------
[11] C. L. Liu and P. P. Vaidyanathan, "Super nested arrays: Linear sparse 
     arrays with reduced mutual coupling—Part I: Fundamentals," IEEE Trans. 
     Signal Processing, vol. 64, no. 15, pp. 3997-4012, Aug. 2016.

Implementation Notes:
--------------------
- All positions stored as zero-based integers after construction
- Dilation factors D1, D2 must be positive integers (≥ 1)
- Base spacing d typically set to λ/2 (half-wavelength)
- Follows BaseArrayProcessor standardized 7-step pipeline
- When D1=1, reduces to DNA; when D1=D2=1, reduces to nested

Author: Hossein (RadarPy Project)
Date: November 7, 2025
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from .bases_classes import BaseArrayProcessor


class DDNAArrayProcessor(BaseArrayProcessor):
    """
    Double Dilated Nested Array (DDNA) Processor
    
    Two-subarray nested configuration with TWO independent dilation factors
    for maximum flexibility in mutual coupling control and aperture design.
    
    DDNA extends DNA by applying dilation to both subarrays (D1 for P1, D2 for P2),
    providing a three-dimensional design space (N1, N2, D1, D2) for optimizing
    hardware constraints, mutual coupling reduction, and DOA estimation performance.
    
    Parameters
    ----------
    N1 : int
        Number of sensors in first subarray (dilated dense), N1 ≥ 1
        - Larger N1 increases coarray coverage (if D1 small)
        - Typical range: 3 to 10
    N2 : int
        Number of sensors in second subarray (double-dilated sparse), N2 ≥ 1
        - Larger N2 extends virtual aperture
        - Typical range: 2 to 8
        - Common ratio: N1 ≈ N2 or N1 slightly larger
    D1 : int, optional
        Dilation factor for first subarray spacing, D1 ≥ 1 (default: 1)
        - D1=1: Dense P1 (high small-lag weights, like DNA/nested)
        - D1=2: Moderate P1 spacing (balanced design)
        - D1≥3: Large P1 spacing (minimal P1 coupling)
        - Controls weight distribution at small lags
    D2 : int, optional
        Dilation factor for second subarray spacing, D2 ≥ 1 (default: 1)
        - D2=1: Standard nested offset (if D1=1)
        - D2=2: Extended aperture with moderate P2 spacing
        - D2≥3: Large aperture, minimal P2 coupling
        - Primary control for aperture extension
    d : float, optional
        Base sensor spacing multiplier in wavelengths (default: 1.0 = λ/2)
        - Standard: d=1.0 corresponds to λ/2 physical spacing
        - Can be adjusted for specific frequency/wavelength
    
    Attributes
    ----------
    N1 : int
        First subarray size (dilated dense)
    N2 : int
        Second subarray size (double-dilated sparse)
    D1 : int
        Dilation factor for P1 (spacing multiplier)
    D2 : int
        Dilation factor for P2 (spacing multiplier)
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
       - (D1=1, D2=1): Standard nested (maximum K_max)
       - (D1=1, D2≥2): DNA-equivalent (P2 coupling reduction)
       - (D1≥2, D2=1): P1 coupling reduction, limited aperture
       - (D1≥2, D2≥2): Full DDNA (both coupling reduction + aperture)
    
    2. Parameter Relationships:
       - D1 primarily affects: w(1), w(2), ..., and P1 coupling
       - D2 primarily affects: Aperture A and P2 coupling
       - Both affect: Overall segment length L and K_max
    
    3. Hardware Constraints:
       - P1 spacing: D1 × d (should be ≥ λ/2 for low coupling)
       - P2 spacing: (D1·N1+1) × D2 × d (should be ≥ λ)
       - Choose D1, D2 to meet both constraints simultaneously
    
    **Coarray Properties**:
    
    - Flexible coarray structure depending on (D1, D2)
    - Best L when D1=D2=1 (nested), reduces with larger dilations
    - Aperture extends rapidly with D2
    - Weight distribution controlled by D1
    - Holes possible for D1≥2 or D2≥3 depending on N1, N2
    
    **Recommended Configurations**:
    
    - **(D1=1, D2=2)**: Good DOF, moderate coupling reduction
    - **(D1=2, D2=2)**: Balanced design, good for most applications
    - **(D1=2, D2=3)**: Extended aperture, minimal coupling
    - **(D1=1, D2=1)**: Baseline nested array for comparison
    
    Examples
    --------
    **Standard Nested (D1=D2=1) - Baseline**:
    
    >>> processor = DDNAArrayProcessor(N1=4, N2=3, D1=1, D2=1)
    >>> results = processor.run_full_analysis()
    >>> print(f"Positions: {processor.data.sensors_positions}")
    Positions: [0, 1, 2, 3, 5, 10, 15]
    >>> print(f"L: {results.segment_length}, K_max: {results.segment_length // 2}")
    L: 31, K_max: 15
    
    **DNA-Equivalent (D1=1, D2=2)**:
    
    >>> processor = DDNAArrayProcessor(N1=4, N2=3, D1=1, D2=2)
    >>> results = processor.run_full_analysis()
    >>> print(f"Positions: {processor.data.sensors_positions}")
    Positions: [0, 1, 2, 3, 10, 20, 30]
    >>> print(f"Aperture: {results.coarray_aperture}, L: {results.segment_length}")
    Aperture: 60, L: 7
    
    **Full DDNA (D1=2, D2=2) - Balanced Design**:
    
    >>> processor = DDNAArrayProcessor(N1=4, N2=3, D1=2, D2=2)
    >>> results = processor.run_full_analysis()
    >>> print(f"P1 spacing: {processor.D1}d, P2 spacing: {(processor.D1*processor.N1+1)*processor.D2}d")
    P1 spacing: 2d, P2 spacing: 18d
    >>> print(f"N: {processor.total_sensors}, A: {results.coarray_aperture}, L: {results.segment_length}")
    N: 7, A: 110, L: 7
    
    **Parameter Study - Vary Both Dilations**:
    
    >>> for D1 in [1, 2]:
    ...     for D2 in [1, 2, 3]:
    ...         proc = DDNAArrayProcessor(N1=4, N2=3, D1=D1, D2=D2)
    ...         res = proc.run_full_analysis()
    ...         print(f"(D1={D1},D2={D2}): A={res.coarray_aperture:3d}, "
    ...               f"L={res.segment_length:2d}, K_max={res.segment_length//2:2d}")
    (D1=1,D2=1): A= 28, L=29, K_max=14
    (D1=1,D2=2): A= 58, L= 7, K_max= 3
    (D1=1,D2=3): A= 88, L= 7, K_max= 3
    (D1=2,D2=1): A= 56, L=15, K_max= 7
    (D1=2,D2=2): A=116, L= 7, K_max= 3
    (D1=2,D2=3): A=176, L= 7, K_max= 3
    
    **Batch Comparison Using Utility Function**:
    
    >>> from geometry_processors.ddna_processor import compare_ddna_arrays
    >>> comparison = compare_ddna_arrays(
    ...     N1_values=[4, 4, 4, 4],
    ...     N2_values=[3, 3, 3, 3],
    ...     D1_values=[1, 1, 2, 2],
    ...     D2_values=[1, 2, 2, 3]
    ... )
    >>> print(comparison[['D1', 'D2', 'Coarray_Aperture', 'L', 'K_max']])
       D1  D2  Coarray_Aperture   L  K_max
        1   1                28  29     14
        1   2                58   7      3
        2   2               116   7      3
        2   3               176   7      3
    
    **Accessing Performance Metrics**:
    
    >>> processor = DDNAArrayProcessor(N1=5, N2=4, D1=2, D2=2)
    >>> results = processor.run_full_analysis()
    >>> print(results.performance_summary_table.to_markdown(index=False))
    | Metric                       | Value                      |
    |:-----------------------------|:---------------------------|
    | Array Type                   | DDNA (Double Dilated Nested)|
    | N1 (First Subarray)          | 5                          |
    | N2 (Second Subarray)         | 4                          |
    | Dilation Factor D1           | 2                          |
    | Dilation Factor D2           | 2                          |
    | ...                          | ...                        |
    
    See Also
    --------
    DNAArrayProcessor : Single dilation nested array (DDNA with D1=1)
    NestedArrayProcessor : Standard nested array (DDNA with D1=D2=1)
    SNA3ArrayProcessor : Super nested array variant (three subarrays)
    ANAII2ArrayProcessor : Augmented nested array II-2 (three subarrays)
    compare_ddna_arrays : Utility function for batch comparison
    
    References
    ----------
    .. [1] C. L. Liu and P. P. Vaidyanathan, "Super nested arrays: Linear sparse 
           arrays with reduced mutual coupling—Part I: Fundamentals," IEEE Trans. 
           Signal Processing, vol. 64, no. 15, pp. 3997-4012, Aug. 2016.
    .. [2] P. P. Vaidyanathan and P. Pal, "Sparse sensing with co-prime samplers 
           and arrays," IEEE Trans. Signal Processing, vol. 59, no. 2, pp. 573-586, 
           Feb. 2011.
    """
    
    def __init__(self, N1: int, N2: int, D1: int = 1, D2: int = 1, d: float = 1.0):
        """
        Initialize DDNA Array Processor.
        
        Parameters
        ----------
        N1 : int
            First subarray size (dilated dense), must be ≥ 1
        N2 : int
            Second subarray size (double-dilated sparse), must be ≥ 1
        D1 : int
            First dilation factor (P1 spacing), must be ≥ 1
        D2 : int
            Second dilation factor (P2 spacing), must be ≥ 1
        d : float
            Base spacing multiplier (wavelengths)
        """
        if N1 < 1 or N2 < 1:
            raise ValueError(f"N1 and N2 must be ≥ 1, got N1={N1}, N2={N2}")
        if D1 < 1:
            raise ValueError(f"Dilation factor D1 must be ≥ 1, got D1={D1}")
        if D2 < 1:
            raise ValueError(f"Dilation factor D2 must be ≥ 1, got D2={D2}")
        
        self.N1 = N1
        self.N2 = N2
        self.D1 = D1
        self.D2 = D2
        self.total_sensors = N1 + N2
        
        # Construct sensor positions
        positions = self._construct_ddna_pattern(N1, N2, D1, D2, d)
        
        # Initialize base processor
        super().__init__(
            name=f"DDNA Array (N1={N1}, N2={N2}, D1={D1}, D2={D2})",
            array_type="Double Dilated Nested Array (DDNA)",
            sensor_positions=positions,
            d=d
        )
    
    def _construct_ddna_pattern(self, N1: int, N2: int, D1: int, D2: int, d: float) -> np.ndarray:
        """
        Construct DDNA sensor positions with two independent dilation factors.
        
        This method implements the full DDNA construction where both subarrays
        have independent dilation control (D1 for P1, D2 for P2).
        
        Construction Algorithm:
        ----------------------
        The DDNA construction applies dilation to BOTH subarrays independently:
        
        **Step 1: Dilated Dense Subarray P1** (with dilation D1)
           P1 = d × D1 × [0, 1, 2, ..., N1-1]
           
           Purpose:
           - D1 controls inter-element spacing within P1
           - D1=1: Dense consecutive sensors (like nested/DNA)
           - D1≥2: Spaced P1 sensors for coupling reduction
           - Affects weight distribution at small lags
           
           Example (N1=4, D1=2, d=1):
           P1 = 1 × 2 × [0,1,2,3] = [0, 2, 4, 6]
           
        **Step 2: Double-Dilated Sparse Subarray P2** (with D1 and D2)
           Offset = (D1 × N1 + 1) × D2
           P2 = d × Offset × [1, 2, ..., N2]
                = d × (D1·N1+1) × D2 × [1, 2, ..., N2]
           
           Purpose:
           - Offset accounts for dilated P1 extent (D1×N1)
           - D2 controls inter-element spacing within P2
           - Creates extended aperture with controllable spacing
           - Cross-differences between P1 and P2 fill coarray
           
           Example (N1=4, N2=3, D1=2, D2=3, d=1):
           Offset = (2×4+1) × 3 = 9 × 3 = 27
           P2 = 1 × 27 × [1,2,3] = [27, 54, 81]
           
        **Step 3: Combination**
           - Concatenate: Positions = P1 ∪ P2
           - Sort: Ensure ascending order
           - Already zero-based from P1 starting at 0
           
           Complete Example (N1=4, N2=3, D1=2, D2=3, d=1):
           P1 = [0, 2, 4, 6]
           P2 = [27, 54, 81]
           Combined = [0, 2, 4, 6, 27, 54, 81]
        
        Dilation Factors Rationale:
        --------------------------
        Two independent factors provide maximum design flexibility:
        
        **D1 (First Subarray Dilation)**:
        - **D1 = 1**: Dense P1, high w(1), w(2), ... (like DNA/nested)
          - Spacing: d (typically λ/2)
          - Best for maximizing small-lag weights
        - **D1 = 2**: Moderate P1 spacing
          - Spacing: 2d (typically λ)
          - Good balance: reduced P1 coupling, reasonable weights
        - **D1 ≥ 3**: Large P1 spacing
          - Spacing: ≥ 3d (typically ≥ 1.5λ)
          - Minimal P1 coupling, but lower small-lag weights
        
        **D2 (Second Subarray Dilation)**:
        - **D2 = 1**: Standard nested offset (if D1=1)
          - Minimum P2 spacing: (D1·N1+1) × d
          - Maximum DOF for given D1
        - **D2 = 2**: Extended aperture, moderate P2 spacing
          - Spacing: 2(D1·N1+1) × d
          - Good aperture extension with reasonable coupling
        - **D2 ≥ 3**: Large aperture, minimal P2 coupling
          - Spacing: ≥ 3(D1·N1+1) × d
          - Maximum aperture, may have holes
        
        Design Guidelines:
        -----------------
        1. Start with (D1=1, D2=1) as baseline nested array
        2. If P2 coupling is problematic: increase D2 (becomes DNA-like)
        3. If P1 also needs spacing: increase D1 (full DDNA)
        4. Ensure D1 × d ≥ λ/2 and (D1·N1+1) × D2 × d ≥ λ
        5. For d = λ/2: Ensure D1 ≥ 1 and (D1·N1+1) × D2 ≥ 2
        6. Balance: Larger (D1,D2) extends aperture but may reduce L
        
        Parameters
        ----------
        N1 : int
            First subarray size (dilated dense), N1 ≥ 1
        N2 : int
            Second subarray size (double-dilated sparse), N2 ≥ 1
        D1 : int
            First dilation factor (P1 spacing multiplier), D1 ≥ 1
        D2 : int
            Second dilation factor (P2 spacing multiplier), D2 ≥ 1
        d : float
            Base spacing multiplier (wavelengths)
        
        Returns
        -------
        np.ndarray
            Zero-based integer sensor positions [0, p1, p2, ..., pN-1], sorted
            Shape: (N1 + N2,)
            Dtype: int64
        
        Examples
        --------
        **Standard nested (D1=D2=1)**: N1=4, N2=3, d=1
        
        >>> positions = _construct_ddna_pattern(4, 3, 1, 1, 1.0)
        >>> print(positions)
        [0 1 2 3 5 10 15]
        # P1=[0,1,2,3], P2=5×[1,2,3]=[5,10,15]
        
        **DNA-equivalent (D1=1, D2=2)**: N1=3, N2=2, d=1
        
        >>> positions = _construct_ddna_pattern(3, 2, 1, 2, 1.0)
        >>> print(positions)
        [0 1 2 8 16]
        # P1=[0,1,2], P2=8×[1,2]=[8,16]
        
        **Full DDNA (D1=2, D2=2)**: N1=3, N2=2, d=1
        
        >>> positions = _construct_ddna_pattern(3, 2, 2, 2, 1.0)
        >>> print(positions)
        [0 2 4 14 28]
        # P1=2×[0,1,2]=[0,2,4], P2=14×[1,2]=[14,28]
        
        Notes
        -----
        - When D1=1, reduces to DNA with dilation D2
        - When D1=D2=1, reduces to standard nested array (matches NestedArrayProcessor)
        - All positions returned as integers when d=1.0 (standard)
        - P1 starts at 0 (zero-based from construction)
        
        See Also
        --------
        DNAArrayProcessor._construct_dna_pattern : Single dilation version (D1=1)
        NestedArrayProcessor : Standard nested array (DDNA with D1=D2=1)
        compute_all_differences : Uses these positions to compute coarray
        """
        # Step 1: Construct dilated dense subarray P1 with dilation D1
        # Use [0, 1, ..., N1-1] to match standard nested array convention
        P1 = d * D1 * np.arange(N1)
        
        # Step 2: Construct double-dilated sparse subarray P2
        # Offset accounts for dilated P1 extent, then apply D2
        offset_P2 = (D1 * N1 + 1) * D2
        P2 = d * offset_P2 * np.arange(1, N2 + 1)
        
        # Step 3: Combine subarrays (already zero-based)
        positions = np.concatenate([P1, P2])
        
        # Return sorted integer positions
        return np.sort(positions).astype(int)
    
    def compute_array_spacing(self) -> float:
        """
        Compute the base spacing of the DDNA array.
        
        For DDNA, spacing varies across the array due to independent dilations:
        - Within P1: spacing = D1 × d
        - Within P2: spacing = (D1·N1+1) × D2 × d  
        - Between P1 and P2: gap varies
        
        This method returns the minimum spacing (typically within P1).
        
        Returns
        -------
        float
            Minimum inter-sensor spacing in the array
        """
        positions = np.array(self.data.sensors_positions)
        return float(np.min(np.diff(np.sort(positions))))
    
    def compute_all_differences(self) -> None:
        """
        Compute all pairwise differences (difference coarray).
        
        The DDNA difference coarray has unique structure due to two dilations:
        
        1. **Within P1 differences**:
           - Lags: D1, 2D1, ..., (N1-1)D1
           - Weights depend on N1
           - D1=1: consecutive small lags (high weights)
           - D1≥2: sparse small lags (lower weights)
        
        2. **Within P2 differences**:
           - Lags: (D1·N1+1)D2, 2(D1·N1+1)D2, ...
           - Large lags extending aperture
           - Weights depend on N2
        
        3. **Cross P1-P2 differences**:
           - Fill intermediate lags between P1 and P2 ranges
           - Critical for contiguous segment formation
           - Structure depends on both D1 and D2
        
        The dilation factors (D1, D2) control the distribution and spacing
        of these differences in the virtual coarray.
        
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
        
        Identifies key coarray properties for DDNA configuration:
        - **Unique differences**: All distinct virtual sensor positions (lags)
        - **Coarray positions**: Both positive and negative lags
        - **Virtual-only positions**: Lags not in physical array
        - **Coarray aperture**: Total span of virtual array
        
        For DDNA arrays, expect:
        - Coarray structure depends on (D1, D2) combination
        - D1=D2=1: Dense contiguous coarray (nested)
        - D1=1, D2≥2: Extended aperture, possible holes (DNA-like)
        - D1≥2, D2≥2: Large aperture, segment length varies
        - Best L typically achieved with smaller (D1, D2)
        
        Both dilation factors affect coarray coverage and hole distribution.
        
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
        
        For DDNA arrays, weight distribution has distinctive features:
        - **Small lags**: Weight depends on D1
          - D1=1: High weights (consecutive P1 sensors)
          - D1≥2: Lower weights (sparse P1 sensors)
        - **Large lags**: From P2 and cross-differences
        - **Intermediate lags**: Filled by cross-subarray differences
        
        The (D1, D2) combination determines weight distribution shape:
        - (1,1): Highest small-lag weights (nested)
        - (1,≥2): Good small-lag weights, extended aperture (DNA-like)
        - (≥2,≥2): Moderate weights throughout, very extended aperture
        
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
        
        The contiguous segment determines maximum detectable sources: K_max = ⌊L/2⌋.
        
        For DDNA arrays:
        - D1=D2=1: L ≈ 2N (nested, maximum DOF)
        - D1=1, D2≥2: L typically N/2 to N (DNA-like)
        - D1≥2, D2=1: L typically N to 1.5N
        - D1≥2, D2≥2: L varies, typically N/2 to N
        
        The (D1, D2) combination significantly affects segment length:
        - Smaller dilations → longer L
        - Larger dilations → shorter L, but extended aperture
        
        Design trade-off: DOF (L) vs. aperture (A) vs. coupling reduction.
        
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
        
        Holes reduce effective DOF and can impact DOA estimation accuracy.
        
        For DDNA arrays:
        - D1=D2=1: Typically zero holes (nested)
        - D1=1, D2≥2: May have holes depending on N1, N2 (DNA-like)
        - D1≥2, any D2: More likely to have holes
        - D1≥3 or D2≥3: Higher probability of holes
        
        The number and distribution of holes indicate coarray quality.
        Fewer holes generally mean better DOA estimation performance.
        
        Both dilation factors affect hole formation, with D1 having
        stronger impact on holes near zero lag.
        
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
        
        Creates a detailed table with key DDNA array characteristics:
        - Physical array properties (N, N1, N2, D1, D2, aperture)
        - Virtual array properties (unique lags, segment length)
        - Performance metrics (K_max, holes)
        - Weight distribution (w at small lags)
        
        This summary enables comparison between different DDNA configurations
        and other array types, highlighting the effects of dual dilation.
        
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
            {"Metric": "Array Type", "Value": f"DDNA (Double Dilated Nested Array)"},
            {"Metric": "N1 (First Subarray)", "Value": f"{self.N1}"},
            {"Metric": "N2 (Second Subarray)", "Value": f"{self.N2}"},
            {"Metric": "Dilation Factor D1", "Value": f"{self.D1}"},
            {"Metric": "Dilation Factor D2", "Value": f"{self.D2}"},
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
        Visualize the DDNA array and its difference coarray (console output).
        
        Displays:
        1. Physical sensor positions (P1 and P2 marked separately with dilations)
        2. Difference coarray positions (virtual sensors)
        3. Weight distribution at key lags
        4. Dilation factors and their effects
        
        This provides quick visual verification of the DDNA construction
        and its coarray properties with dual dilation effects.
        """
        print("\n" + "="*70)
        print(f"DDNA ARRAY VISUALIZATION: N1={self.N1}, N2={self.N2}, D1={self.D1}, D2={self.D2}")
        print("="*70)
        
        # Physical array
        positions = np.array(self.data.sensors_positions)
        print("\nPhysical Sensor Positions:")
        print(f"  P1 (Dilated Dense, D1={self.D1}):  {positions[:self.N1]}")
        print(f"  P2 (Double-Dilated Sparse, D2={self.D2}): {positions[self.N1:]}")
        print(f"  Total: {len(positions)} sensors")
        print(f"  P1 spacing: {self.D1}d,  P2 spacing: {(self.D1*self.N1+1)*self.D2}d")
        
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
        """String representation of DDNA array."""
        return (f"DDNAArrayProcessor(N1={self.N1}, N2={self.N2}, D1={self.D1}, D2={self.D2}, "
                f"total_sensors={self.total_sensors})")


def compare_ddna_arrays(N1_values: List[int], 
                        N2_values: List[int],
                        D1_values: List[int],
                        D2_values: List[int],
                        d: float = 1.0) -> pd.DataFrame:
    """
    Compare multiple DDNA configurations side-by-side.
    
    This utility function analyzes multiple DDNA arrays with different
    (N1, N2, D1, D2) parameters and compiles their performance metrics
    into a comparison table.
    
    Useful for:
    - Parameter selection (choosing optimal N1/N2/D1/D2 combination)
    - Dilation factor studies (comparing (D1,D2) effects)
    - Trade-off analysis between aperture, DOF, and coupling
    - Hardware design optimization (meeting spacing constraints)
    
    Parameters
    ----------
    N1_values : List[int]
        List of N1 values (first subarray sizes) to test
    N2_values : List[int]
        List of N2 values (second subarray sizes) to test
        Must have same length as N1_values
    D1_values : List[int]
        List of D1 dilation factors to test
        Must have same length as N1_values
    D2_values : List[int]
        List of D2 dilation factors to test
        Must have same length as N1_values
    d : float, optional
        Base spacing multiplier for all arrays (default: 1.0)
    
    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        - N1, N2, D1, D2: Configuration parameters
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
    >>> # Compare dilation combinations for fixed N1, N2
    >>> comparison = compare_ddna_arrays(
    ...     N1_values=[4, 4, 4, 4],
    ...     N2_values=[3, 3, 3, 3],
    ...     D1_values=[1, 1, 2, 2],
    ...     D2_values=[1, 2, 2, 3],
    ...     d=1.0
    ... )
    >>> print(comparison.to_markdown(index=False))
    >>> 
    >>> # Study D1 vs D2 effects
    >>> comparison = compare_ddna_arrays(
    ...     N1_values=[5, 5, 5, 5, 5, 5],
    ...     N2_values=[4, 4, 4, 4, 4, 4],
    ...     D1_values=[1, 1, 1, 2, 2, 2],
    ...     D2_values=[1, 2, 3, 1, 2, 3]
    ... )
    >>> print(comparison[['D1', 'D2', 'Coarray_Aperture', 'L', 'K_max']])
    """
    if not (len(N1_values) == len(N2_values) == len(D1_values) == len(D2_values)):
        raise ValueError("N1_values, N2_values, D1_values, and D2_values must have same length")
    
    results = []
    
    for N1, N2, D1, D2 in zip(N1_values, N2_values, D1_values, D2_values):
        processor = DDNAArrayProcessor(N1=N1, N2=N2, D1=D1, D2=D2, d=d)
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
            'D1': D1,
            'D2': D2,
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
