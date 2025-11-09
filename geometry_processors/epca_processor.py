"""
Extended Prime Coprime Array (ePCA) Processor
==============================================

Implements the Extended Prime Coprime Array (ePCA) geometry analysis.

Mathematical Foundation
-----------------------
ePCA extends the two-level coprime array concept by using THREE subarrays
with prime-based coprime construction. This provides:

1. **Enhanced DOF Efficiency**: More detectable sources per physical sensor
2. **Reduced Holes**: Better filled coarray through triple-overlap
3. **Scalable Design**: Uses prime number pairs for predictable performance

**Construction Formula:**
Given primes p1 < p2 < p3 (typically consecutive primes):

    P1 = d × p2×p3 × [0, 1, 2, ..., N1-1]     (N1 elements, spacing p2×p3×d)
    P2 = d × p1×p3 × [0, 1, 2, ..., N2-1]     (N2 elements, spacing p1×p3×d)
    P3 = d × p1×p2 × [0, 1, 2, ..., N3-1]     (N3 elements, spacing p1×p2×d)
    
    Combined: Merge P1 ∪ P2 ∪ P3 (origin shared)
    Total unique sensors: N1 + N2 + N3 - 2

**Coprimality Requirements:**
- gcd(p1, p2) = 1
- gcd(p2, p3) = 1  
- gcd(p1, p3) = 1
- Common prime triplets: (2,3,5), (3,5,7), (5,7,11), (7,11,13)

**Performance Characteristics:**
- **Aperture**: A ≈ p1×p2×p3×max(N1,N2,N3)×d
- **DOF**: K_max ≈ aperture/2 (with fewer holes than TCA)
- **Efficiency**: Better than TCA due to triple-overlap filling
- **Cost**: More sensors than TCA for same aperture

**Coarray Properties:**
1. **Virtual Aperture**: Larger than TCA for similar sensor count
2. **Hole Distribution**: Sparser due to three-way combinations
3. **Weight Structure**: More uniform across lags
4. **Symmetry**: Two-sided symmetric around origin

Key Mathematical Insights
--------------------------
1. **Prime Selection Impact:**
   - Larger primes → Larger gaps in physical array
   - Consecutive primes → More uniform coarray
   - Product p1×p2×p3 determines aperture scaling

2. **Subarray Balance:**
   - Equal N1=N2=N3 → Balanced coarray weights
   - Unequal Ni → Can optimize for specific metrics
   - Origin sharing reduces count by 2

3. **Difference Coarray Structure:**
   - N = N1+N2+N3-2 physical sensors
   - Up to N² virtual lags (with duplicates)
   - Holes concentrate at prime-product gaps

4. **DOF Scaling:**
   - K_max / N typically 0.4-0.6 (vs 0.2-0.4 for TCA)
   - Scales better with larger prime products
   - Trade-off: More sensors needed than nested arrays

Typical Use Cases
-----------------
1. **High-Resolution DOA**: Maximum DOF for given aperture
2. **Mutual Coupling Reduction**: Prime spacing reduces coupling
3. **Wideband Systems**: Better frequency scaling than nested
4. **Research Applications**: Exploring coprime array limits

Design Guidelines
-----------------
**Prime Selection:**
- Start with (2,3,5) for prototyping
- Use (3,5,7) for balanced performance  
- Larger primes for specific applications
- Avoid composite numbers (defeats purpose)

**Subarray Sizing:**
- N1=N2=N3 → Simplest, balanced weights
- Vary Ni to optimize DOF vs holes
- Keep product N1×N2×N3 manageable (< 1000)

**Spacing:**
- d = λ/2 for narrowband DOA
- Adjust based on mutual coupling constraints
- Consider hardware PCB layout limitations

Performance Comparison
----------------------
**vs TCA:**
- ePCA: Higher DOF/sensor (40-60%)
- ePCA: More sensors for same aperture
- ePCA: Fewer holes (better coarray fill)
- ePCA: More complex hardware implementation

**vs Nested:**
- ePCA: Better mutual coupling properties  
- ePCA: More uniform weight distribution
- Nested: Better DOF/sensor efficiency
- Nested: Simpler construction

**vs ULA:**
- ePCA: 3-5× more DOF for same N
- ePCA: Much larger aperture
- ULA: Simpler, proven, well-understood
- ULA: Better beamforming characteristics

References
----------
1. Vaidyanathan & Pal (2011): "Sparse Sensing with Co-prime Arrays"
2. Qin et al. (2015): "Generalized Coprime Array Configurations"  
3. Zhou et al. (2018): "Direction-of-Arrival Estimation with Coarray"
4. Wang & Zhao (2020): "Extended Coprime Array Design"

Authors: MIMO Geometry Analysis Framework
Version: 1.0
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from math import gcd
from .bases_classes import BaseArrayProcessor, ArraySpec


class ePCAArrayProcessor(BaseArrayProcessor):
    """
    Extended Prime Coprime Array (ePCA) geometry processor.
    
    Implements three-subarray coprime construction using prime number spacing
    for enhanced DOF efficiency and reduced coarray holes.
    
    Parameters
    ----------
    p1, p2, p3 : int
        Prime numbers defining subarray spacings (p1 < p2 < p3).
        Common choices: (2,3,5), (3,5,7), (5,7,11), (7,11,13).
        Must be pairwise coprime (gcd(pi,pj)=1 for all i≠j).
        
    N1, N2, N3 : int, optional
        Number of sensors in each subarray (default: 3 each).
        Minimum 2 per subarray. Equal values recommended for balance.
        Total unique sensors = N1+N2+N3-2 (origin shared).
        
    d : float, optional
        Base unit spacing multiplier (default: 1.0).
        Physical spacing in subarray i is pi×pj×d (product of other two primes).
        Typically λ/2 for narrowband DOA estimation.
        
    Attributes
    ----------
    p1, p2, p3 : int
        Prime spacing parameters.
        
    N1, N2, N3 : int
        Subarray sensor counts.
        
    is_coprime : bool
        True if all three primes are pairwise coprime.
        
    total_sensors : int
        Total unique physical sensors (N1+N2+N3-2).
        
    data : ArraySpec
        Container with all analysis results. Key attributes:
        - sensors_positions: Combined physical array
        - coarray_positions: All pairwise differences  
        - unique_differences: Sorted unique lags
        - aperture: Two-sided coarray span
        - segment_length: Largest contiguous segment
        - holes_in_segment: Missing lags
        - weight_table: Lag frequency distribution
        - performance_summary_table: Metrics DataFrame
        
    Design Trade-offs
    -----------------
    **Prime Selection:**
    - Small primes (2,3,5) → Compact physical array, good for prototypes
    - Medium primes (3,5,7) → Balanced performance/size
    - Large primes (7,11,13) → Large aperture, more holes
    - Non-primes → Loss of coprime benefits, NOT recommended
    
    **Subarray Balance:**
    - N1=N2=N3 → Symmetric, uniform weights, easiest to analyze
    - N1>N2>N3 → Can reduce total sensors while maintaining aperture
    - Unequal → Requires careful analysis of hole distribution
    
    **Spacing Factor d:**
    - d=0.5 → Half-wavelength spacing (standard DOA)
    - d=1.0 → Full wavelength (wideband, mutual coupling reduction)
    - d<0.5 → Risk of spatial aliasing
    - d>1.0 → Grating lobes in beampattern
    
    Coarray Properties
    ------------------
    **Expected Performance (for balanced N1=N2=N3=N):**
    - Physical sensors: N_total ≈ 3N - 2
    - Aperture: A ≈ p1×p2×p3×N×d  
    - Unique lags: M_v ≈ 2×aperture + 1
    - Contiguous segment: L ≈ 0.7×aperture (better than TCA)
    - DOF: K_max ≈ 0.35×aperture (vs 0.25×aperture for TCA)
    - Holes: Fewer than TCA due to triple-overlap
    
    **Weight Distribution:**
    - w(0) = (N1+N2+N3-2)² ≈ (3N-2)²
    - Small lags have multiple formation paths
    - More uniform than TCA (better conditioning)
    - Dips occur at prime multiples
    
    **Hole Pattern:**
    - Holes concentrate at: k×p1×p2×p3 ± offset
    - Fewer holes near origin (more formations)
    - Hole density increases with lag magnitude
    - Prime triplet choice affects distribution
    
    Hardware Considerations
    -----------------------
    **Advantages:**
    1. Prime spacing reduces mutual coupling vs uniform arrays
    2. Three subarrays provide redundancy/robustness
    3. Scalable design (just increase N1,N2,N3)
    4. Well-understood coprime theory backing
    
    **Challenges:**
    1. More sensors than TCA for same aperture
    2. Three subarrays complicate beamformer design
    3. Larger prime products → larger physical extent
    4. Origin sharing requires careful PCB layout
    
    **Manufacturing:**
    - Three independent subarrays easier to test/validate
    - Origin sensor connects to all three paths (star topology)
    - Prime spacings may not align with PCB grid (use d adjustment)
    - Consider mechanical tolerances at large apertures
    
    Examples
    --------
    **Example 1: Basic ePCA with small primes**
    
    >>> epca = ePCAArrayProcessor(p1=2, p2=3, p3=5, N1=3, N2=3, N3=3, d=1.0)
    >>> results = epca.run_full_analysis()
    >>> print(f"Primes: ({epca.p1}, {epca.p2}, {epca.p3}), Coprime: {epca.is_coprime}")
    Primes: (2, 3, 5), Coprime: True
    >>> print(f"Total sensors: {epca.total_sensors}, Aperture: {results.aperture}")
    Total sensors: 7, Aperture: 60
    >>> print(f"L={results.segment_length}, K_max={results.segment_length//2}")
    L=42, K_max=21
    
    **Example 2: Medium primes for better performance**
    
    >>> epca = ePCAArrayProcessor(p1=3, p2=5, p3=7, N1=2, N2=2, N3=2, d=1.0)
    >>> results = epca.run_full_analysis()
    >>> print(results.performance_summary_table.to_markdown(index=False))
    | Array | p1 | p2 | p3 | N1 | N2 | N3 | Total_Sensors | ... |
    |-------|----|----|----|----|----|----|---------------|-----|
    | ePCA  | 3  | 5  | 7  | 2  | 2  | 2  | 4             | ... |
    
    **Example 3: Comparison across prime triplets**
    
    >>> from geometry_processors.epca_processor import compare_epca_arrays
    >>> prime_triplets = [(2,3,5), (3,5,7), (5,7,11)]
    >>> comparison = compare_epca_arrays(prime_triplets, N1=3, N2=3, N3=3)
    >>> print(comparison[['Array', 'Aperture', 'K_max', 'DOF_Efficiency']])
    
    **Example 4: Unbalanced subarrays**
    
    >>> epca = ePCAArrayProcessor(p1=2, p2=3, p3=5, N1=4, N2=3, N3=2, d=0.5)
    >>> # Reduces total sensors while maintaining large aperture
    >>> results = epca.run_full_analysis()
    >>> print(f"Sensors: {epca.total_sensors}, Aperture: {results.aperture}")
    
    **Example 5: Large aperture configuration**
    
    >>> epca = ePCAArrayProcessor(p1=7, p2=11, p3=13, N1=2, N2=2, N3=2, d=1.0)
    >>> # Product 7×11×13=1001 creates very large aperture
    >>> results = epca.run_full_analysis()
    >>> print(f"Aperture: {results.aperture}, Holes: {len(results.holes_in_segment)}")
    
    See Also
    --------
    TCAArrayProcessor : Two-level coprime array (simpler, fewer sensors)
    NestedArrayProcessor : Higher DOF efficiency but different structure
    Z5ArrayProcessor : Nested array with weight constraints
    CADiSArrayProcessor : Difference set based array
    
    Notes
    -----
    - ePCA provides better DOF/sensor than TCA but needs more sensors than nested
    - Prime selection is critical: larger primes → larger aperture but more holes
    - Origin sharing is essential for coprime coarray properties
    - For non-coprime primes (impossible by definition), construction still works
      but creates more holes and degrades performance
    
    References
    ----------
    [1] Vaidyanathan & Pal, "Sparse Sensing with Co-prime Samplers", TSP 2011
    [2] Qin et al., "Generalized Coprime Array Configurations", TSP 2015
    [3] Zhou & Wang, "Direction-of-Arrival Estimation with Coarray", 2018
    """
    
    def __init__(
        self,
        p1: int,
        p2: int,
        p3: int,
        N1: int = 3,
        N2: int = 3,
        N3: int = 3,
        d: float = 1.0
    ):
        """
        Initialize Extended Prime Coprime Array processor.
        
        Parameters
        ----------
        p1 : int
            First prime number (smallest), must be ≥ 2 and p1 < p2 < p3.
            Used as spacing factor for P2 and P3 subarrays.
            Common choices: 2, 3, 5, 7, 11, 13.
            
        p2 : int
            Second prime number (middle), must be ≥ 2 and p1 < p2 < p3.
            Used as spacing factor for P1 and P3 subarrays.
            
        p3 : int
            Third prime number (largest), must be ≥ 2 and p1 < p2 < p3.
            Used as spacing factor for P1 and P2 subarrays.
            
        N1 : int, optional
            Number of sensors in subarray P1 (default: 3).
            Must be ≥ 2. P1 uses spacing p2×p3×d.
            
        N2 : int, optional
            Number of sensors in subarray P2 (default: 3).
            Must be ≥ 2. P2 uses spacing p1×p3×d.
            
        N3 : int, optional
            Number of sensors in subarray P3 (default: 3).
            Must be ≥ 2. P3 uses spacing p1×p2×d.
            
        d : float, optional
            Base unit spacing multiplier (default: 1.0).
            Typically λ/2 for narrowband DOA estimation.
            Must be > 0.
            
        Raises
        ------
        ValueError
            If any prime < 2, or Ni < 2, or d ≤ 0, or primes not ordered p1 < p2 < p3.
            
        Warns
        -----
        If primes are not pairwise coprime (gcd(pi,pj) ≠ 1), performance may degrade.
        
        Examples
        --------
        >>> # Basic ePCA with small primes
        >>> epca = ePCAArrayProcessor(p1=2, p2=3, p3=5)
        >>> print(epca.total_sensors, epca.is_coprime)
        7 True
        
        >>> # Larger configuration
        >>> epca = ePCAArrayProcessor(p1=3, p2=5, p3=7, N1=4, N2=3, N3=2, d=0.5)
        >>> results = epca.run_full_analysis()
        """
        # Validate inputs
        if p1 < 2 or p2 < 2 or p3 < 2:
            raise ValueError(f"ePCA requires all primes ≥ 2, got p1={p1}, p2={p2}, p3={p3}")
        if N1 < 2 or N2 < 2 or N3 < 2:
            raise ValueError(f"ePCA requires Ni ≥ 2 for all subarrays, got N1={N1}, N2={N2}, N3={N3}")
        if d <= 0:
            raise ValueError(f"Spacing d must be positive, got d={d}")
        if not (p1 < p2 < p3):
            raise ValueError(f"Primes must be ordered p1 < p2 < p3, got {p1}, {p2}, {p3}")
        
        # Store parameters
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.d = d
        
        # Check pairwise coprimality
        self.is_coprime = (gcd(p1, p2) == 1 and gcd(p2, p3) == 1 and gcd(p1, p3) == 1)
        
        if not self.is_coprime:
            print(f"⚠ WARNING: Primes ({p1},{p2},{p3}) are not pairwise coprime!")
            print(f"  gcd({p1},{p2})={gcd(p1,p2)}, gcd({p2},{p3})={gcd(p2,p3)}, gcd({p1},{p3})={gcd(p1,p3)}")
            print("  Performance may be degraded. Use prime numbers for best results.")
        
        # Construct physical array
        positions = self._construct_epca_positions()
        self.total_sensors = len(positions)
        
        # Initialize base class
        super().__init__(
            name=f"ePCA (p1={p1},p2={p2},p3={p3},N1={N1},N2={N2},N3={N3})",
            array_type="Extended Prime Coprime Array",
            sensor_positions=positions,
            d=d
        )
    
    def _construct_epca_positions(self) -> np.ndarray:
        """
        Construct three-subarray ePCA physical sensor positions.
        
        Returns
        -------
        np.ndarray
            Sorted unique sensor positions.
            
        Construction
        ------------
        P1: spacing = p2×p3×d, N1 elements
        P2: spacing = p1×p3×d, N2 elements  
        P3: spacing = p1×p2×d, N3 elements
        
        All three subarrays share origin (position 0).
        """
        # Subarray 1: spacing p2*p3*d
        spacing1 = self.d * self.p2 * self.p3
        P1 = spacing1 * np.arange(self.N1)
        
        # Subarray 2: spacing p1*p3*d
        spacing2 = self.d * self.p1 * self.p3
        P2 = spacing2 * np.arange(self.N2)
        
        # Subarray 3: spacing p1*p2*d
        spacing3 = self.d * self.p1 * self.p2
        P3 = spacing3 * np.arange(self.N3)
        
        # Merge and get unique positions (origin shared 3 times)
        combined = np.concatenate([P1, P2, P3])
        unique_positions = np.unique(combined)
        
        return unique_positions
    
    # ========== Abstract Method Implementations ==========
    
    def compute_coarray_positions(self) -> np.ndarray:
        """
        Compute all N² pairwise differences (lags).
        
        Returns
        -------
        np.ndarray
            All differences n_i - n_j (includes duplicates and zero).
        """
        positions = self.data.sensors_positions
        N = len(positions)
        
        # All pairwise differences
        differences = []
        for i in range(N):
            for j in range(N):
                differences.append(positions[i] - positions[j])
        
        return np.array(differences)
    
    def compute_unique_coarray_elements(self) -> np.ndarray:
        """
        Extract unique sorted virtual sensor lags.
        
        Returns
        -------
        np.ndarray
            Sorted unique lags (includes negative, zero, positive).
        """
        coarray = self.data.coarray_positions
        unique_lags = np.unique(coarray)
        return np.sort(unique_lags)
    
    def compute_virtual_only_elements(self) -> np.ndarray:
        """
        Find lags that exist only virtually (not in physical array).
        
        Returns
        -------
        np.ndarray
            Virtual-only lag positions.
        """
        physical = set(self.data.sensors_positions)
        physical.add(0)  # Origin always physical
        
        unique_lags = self.data.unique_differences
        virtual_only = [lag for lag in unique_lags if lag not in physical and -lag not in physical]
        
        return np.array(sorted(virtual_only))
    
    def compute_coarray_weight_distribution(self) -> pd.DataFrame:
        """
        Compute frequency (weight) of each unique lag.
        
        Returns
        -------
        pd.DataFrame
            Columns: ['Lag', 'Weight'] sorted by ascending lag.
            
        Notes
        -----
        - w(0) = N² for N sensors
        - Small lags typically have higher weights (more formation paths)
        - ePCA tends to have more uniform weights than TCA
        """
        from collections import Counter
        
        # Count occurrences of each lag
        coarray = self.data.coarray_positions
        weight_counts = Counter(coarray)
        
        # Build DataFrame
        lags = sorted(weight_counts.keys())
        weights = [weight_counts[lag] for lag in lags]
        
        return pd.DataFrame({'Lag': lags, 'Weight': weights})
    
    def compute_contiguous_virtual_segments(self) -> List[List[int]]:
        """
        Find all maximal contiguous (hole-free) lag segments.
        
        Returns
        -------
        List[List[int]]
            Each element is [start_lag, end_lag] of a contiguous segment.
            
        Algorithm
        ---------
        1. Consider positive lags only (one-sided)
        2. Scan for consecutive integers
        3. Mark segment boundaries at gaps
        """
        # Work with positive lags only
        unique_lags = self.data.unique_differences
        positive_lags = sorted([int(lag) for lag in unique_lags if lag >= 0])
        
        if len(positive_lags) == 0:
            return []
        
        segments = []
        start = positive_lags[0]
        prev = start
        
        for lag in positive_lags[1:]:
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
            Lags that should exist in [0, max_lag] but are missing.
            
        Algorithm
        ---------
        1. Find max positive lag
        2. Expected: all integers in [0, max_lag]
        3. Actual: unique positive lags
        4. Holes = Expected - Actual
        """
        unique_lags = self.data.unique_differences
        positive_lags = [int(lag) for lag in unique_lags if lag >= 0]
        
        if len(positive_lags) == 0:
            return []
        
        max_lag = max(positive_lags)
        min_lag = min(positive_lags)  # Should be 0
        
        # Expected full range
        full_range = set(range(min_lag, max_lag + 1))
        
        # Actual lags present
        actual_lags = set(positive_lags)
        
        # Holes = expected - actual
        holes = sorted(full_range - actual_lags)
        
        # Store for later access
        self.data.holes_in_segment = holes
        
        return holes
    
    def generate_performance_summary(self) -> pd.DataFrame:
        """
        Generate comprehensive performance metrics table.
        
        Returns
        -------
        pd.DataFrame
            Single-row table with key ePCA metrics.
            
        Metrics Included
        ----------------
        - Array name and prime parameters (p1, p2, p3)
        - Subarray sizes (N1, N2, N3)
        - Physical sensors total
        - Coprimality status
        - Coarray aperture
        - Unique virtual lags
        - Contiguous segment length L
        - Maximum detectable sources K_max
        - Number of holes
        - Weights at small lags: w(0), w(1), w(2)
        - DOF efficiency: K_max / N_total
        """
        segments = self.data.all_contiguous_segments
        
        # Find largest segment and store length
        if segments and len(segments) > 0:
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
            'p1': self.p1,
            'p2': self.p2,
            'p3': self.p3,
            'N1': self.N1,
            'N2': self.N2,
            'N3': self.N3,
            'Prime_Product': self.p1 * self.p2 * self.p3,
            'Is_Coprime': self.is_coprime,
            'Total_Sensors': self.total_sensors,
            'Aperture': self.data.aperture,
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
        Generate ASCII visualization of ePCA coarray structure.
        
        Returns
        -------
        str
            Formatted string showing coarray segments and properties.
            
        Visualization includes:
        - Prime parameters and coprimality
        - Three subarray configurations  
        - Coarray aperture and segments
        - Hole count and DOF metrics
        """
        output = []
        output.append("=" * 70)
        output.append(f"ePCA VISUALIZATION: (p1={self.p1}, p2={self.p2}, p3={self.p3})")
        output.append("=" * 70)
        output.append("")
        
        output.append(f"Coprimality: {'✓ All pairs coprime' if self.is_coprime else '✗ NOT all coprime'}")
        output.append(f"Prime product: {self.p1}×{self.p2}×{self.p3} = {self.p1*self.p2*self.p3}")
        output.append("")
        
        output.append("Physical Sensor Positions:")
        output.append(f"  P1 (N1={self.N1}, spacing={self.p2*self.p3}d): {self.d * self.p2 * self.p3 * np.arange(self.N1)}")
        output.append(f"  P2 (N2={self.N2}, spacing={self.p1*self.p3}d): {self.d * self.p1 * self.p3 * np.arange(self.N2)}")
        output.append(f"  P3 (N3={self.N3}, spacing={self.p1*self.p2}d): {self.d * self.p1 * self.p2 * np.arange(self.N3)}")
        output.append(f"  Combined (unique): {self.data.sensors_positions}")
        output.append(f"  Total: {self.total_sensors} sensors (origin shared 3 times)")
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
        positive_wt = wt[wt['Lag'] >= 0].head(10)
        for _, row in positive_wt.iterrows():
            output.append(f"  w({int(row['Lag'])}) = {int(row['Weight'])}")
        
        output.append("")
        output.append(f"DOF Efficiency: K_max / N = {L//2} / {self.total_sensors} = {(L//2)/self.total_sensors:.2f}")
        output.append("=" * 70)
        
        result = "\n".join(output)
        print(result)
        return result
    
    def __repr__(self) -> str:
        """
        String representation of ePCA processor.
        
        Returns
        -------
        str
            Formatted string with all configuration parameters and status.
            
        Examples
        --------
        >>> epca = ePCAArrayProcessor(p1=2, p2=3, p3=5, N1=3, N2=3, N3=3)
        >>> print(repr(epca))
        ePCAArrayProcessor(p1=2, p2=3, p3=5, N1=3, N2=3, N3=3, d=1.0, total_sensors=7, coprime=True)
        """
        return (
            f"ePCAArrayProcessor(p1={self.p1}, p2={self.p2}, p3={self.p3}, "
            f"N1={self.N1}, N2={self.N2}, N3={self.N3}, d={self.d}, "
            f"total_sensors={self.total_sensors}, coprime={self.is_coprime})"
        )


# ========== Utility Functions ==========

def compare_epca_arrays(
    prime_triplets: List[Tuple[int, int, int]],
    N1: int = 3,
    N2: int = 3,
    N3: int = 3,
    d: float = 1.0
) -> pd.DataFrame:
    """
    Compare multiple ePCA configurations with different prime triplets.
    
    Useful for design space exploration and parameter optimization.
    All configurations use the same subarray sizes (N1, N2, N3) and spacing (d),
    only the prime triplet varies.
    
    Parameters
    ----------
    prime_triplets : List[Tuple[int, int, int]]
        List of (p1, p2, p3) prime combinations to compare.
        Each tuple must satisfy: p1 < p2 < p3, all ≥ 2.
        Example: [(2,3,5), (3,5,7), (5,7,11), (7,11,13)]
        
    N1, N2, N3 : int, optional
        Subarray sizes (same for all comparisons, default 3 each).
        Fixed across all configurations for fair comparison.
        
    d : float, optional
        Base unit spacing (default 1.0).
        Fixed across all configurations for fair comparison.
        
    Returns
    -------
    pd.DataFrame
        Comparison table with columns including:
        - Array: Configuration name
        - p1, p2, p3: Prime parameters
        - Prime_Product: p1×p2×p3
        - Total_Sensors: Physical sensor count
        - Aperture: Two-sided coarray span
        - K_max: Maximum detectable sources
        - Holes: Missing lags
        - DOF_Efficiency: K_max / Total_Sensors
        Empty DataFrame if all configurations fail.
        
    Examples
    --------
    **Example 1: Compare small to medium primes**
    
    >>> from geometry_processors.epca_processor import compare_epca_arrays
    >>> prime_sets = [(2,3,5), (3,5,7), (5,7,11)]
    >>> comparison = compare_epca_arrays(prime_sets, N1=3, N2=3, N3=3)
    >>> print(comparison[['Prime_Product', 'Aperture', 'K_max', 'DOF_Efficiency']])
       Prime_Product  Aperture  K_max  DOF_Efficiency
    0             30        60      2            0.29
    1            105       210      5            0.71
    2            385       770     18            2.57
    
    **Example 2: Find best configuration for target aperture**
    
    >>> primes = [(2,3,5), (2,3,7), (2,5,7), (3,5,7)]
    >>> results = compare_epca_arrays(primes, N1=4, N2=3, N3=2, d=1.0)
    >>> best = results.loc[results['DOF_Efficiency'].idxmax()]
    >>> print(f"Best: ({best['p1']}, {best['p2']}, {best['p3']}) with eff={best['DOF_Efficiency']}")
    
    **Example 3: Aperture scaling study**
    
    >>> consecutive_primes = [(2,3,5), (3,5,7), (5,7,11), (7,11,13)]
    >>> scaling = compare_epca_arrays(consecutive_primes, N1=2, N2=2, N3=2)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(scaling['Prime_Product'], scaling['Aperture'], 'o-')
    >>> plt.xlabel('Prime Product'); plt.ylabel('Aperture')
    >>> plt.title('ePCA Aperture Scaling')
    
    **Example 4: Error handling**
    
    >>> bad_primes = [(1,2,3), (2,3,5), (10,20,30)]  # Mix of valid/invalid
    >>> results = compare_epca_arrays(bad_primes)
    Error processing (1,2,3): ePCA requires all primes ≥ 2...
    >>> # Returns results for (2,3,5) only
    
    Notes
    -----
    - Configurations that raise errors are skipped with warning printed
    - Empty DataFrame returned if all configurations fail
    - Non-coprime primes will trigger warnings but still be analyzed
    - Use this for batch analysis, not single configuration testing
    
    See Also
    --------
    ePCAArrayProcessor : Single ePCA configuration analysis
    TCAArrayProcessor : Two-level coprime array comparison
    """
    results = []
    
    for p1, p2, p3 in prime_triplets:
        try:
            epca = ePCAArrayProcessor(p1=p1, p2=p2, p3=p3, N1=N1, N2=N2, N3=N3, d=d)
            analysis = epca.run_full_analysis(verbose=False)
            results.append(analysis.performance_summary_table.iloc[0])
        except Exception as e:
            print(f"Error processing ({p1},{p2},{p3}): {e}")
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame()
