# geometry_processors/bases_classes.py
"""
Base Classes for MIMO Array Geometry Analysis Framework.

This module provides the foundational classes for analyzing MIMO radar array geometries
through difference coarray computation. It defines:

1. ArraySpec: Data container with 47 pre-defined attributes for storing analysis results
2. BaseArrayProcessor: Abstract base class defining the standardized 7-step analysis pipeline

Author: [Your Name]
Date: November 6, 2025
Version: 1.0.0
"""
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class ArraySpec:
    """
    Structured data container for array geometry analysis results.
    
    This dataclass stores all inputs, intermediate computations, and final metrics
    from the 7-step difference coarray analysis pipeline. All spatial quantities
    are stored in integer lag units unless otherwise noted.
    
    Attributes are organized by analysis phase:
    
    **Identity Attributes:**
        name (str): Human-readable array name (e.g., "Array Z5 (N=7)")
        array_type (str): Array category (e.g., "Weight-Constrained Sparse Array")
    
    **Design/Input Attributes:**
        sensors_positions (List[int]): Physical sensor positions in integer grid units
        num_sensors (int): Total number of physical sensors (N)
        sensor_spacing (float): Base spacing multiplier (d) in physical units
    
    **Core Derived Attributes (Coarray Computation):**
        all_differences_with_duplicates (np.ndarray): N² pairwise differences (n_i - n_j)
        unique_differences (np.ndarray): Sorted unique two-sided integer lags
        num_unique_positions (int): Count of unique virtual sensor positions (Mv)
        coarray_positions (np.ndarray): Alias for unique_differences
        physical_positions (np.ndarray): Sensor positions as numpy array
        virtual_only_positions (np.ndarray): Virtual positions not in physical array
        coarray_holes (np.ndarray): Missing positions in virtual array
    
    **Segment Analysis Attributes (One-sided Analysis):**
        largest_contiguous_segment (np.ndarray): Longest hole-free segment [0, L-1]
        all_contiguous_segments (List[np.ndarray]): All contiguous segments
        segment_lengths (List[int]): Length of each segment
        segment_ranges (List[List[int]]): Start/end indices [[L1_start, L1_end], ...]
        max_detectable_sources (int): K_max = floor(L/2) where L is longest segment
    
    **Holes Analysis Attributes:**
        num_holes (int): Count of missing positions
        missing_virtual_positions (np.ndarray): All one-sided holes up to max(+)
        missing_virtual_positions_below_A (np.ndarray): Holes below aperture A
    
    **Weight Distribution Attributes:**
        weight_table (pd.DataFrame): DataFrame with columns ['Lag', 'Weight']
        weight_dict (Dict[int, int]): Dictionary mapping {lag: frequency}
    
    **Miscellaneous Attributes:**
        segmentation (List[np.ndarray]): Custom segmentation results
        num_segmentation (int): Number of segments
        aperture (int): Two-sided span (max_lag - min_lag) in integer units
    
    **Presentation Attributes:**
        performance_summary_table (pd.DataFrame): Final metrics table for comparison
    
    Usage:
        >>> from geometry_processors.z5_processor_ import Z5ArrayProcessor
        >>> processor = Z5ArrayProcessor(N=7, d=1.0)
        >>> spec = processor.run_full_analysis()
        >>> print(f"Virtual sensors: {spec.num_unique_positions}")
        >>> print(f"Max sources: {spec.max_detectable_sources}")
    """
    # Identity
    name: str
    array_type: str

    # Design / inputs
    sensors_positions: List[int]  # integer grid positions (NOT multiplied by d)
    num_sensors: int
    sensor_spacing: float  # d (physical spacing unit)

    # Core derived artifacts (all in integer lag units unless noted)
    all_differences_with_duplicates: Optional[np.ndarray] = None  # two-sided integer lags (with dupes)
    unique_differences: Optional[np.ndarray] = None               # sorted unique two-sided integer lags
    num_unique_positions: Optional[int] = None

    coarray_positions: Optional[np.ndarray] = None                # alias for unique_differences (two-sided)
    physical_positions: Optional[np.ndarray] = None               # sensor grid positions as ndarray (ints)
    virtual_only_positions: Optional[np.ndarray] = None           # setdiff1d(coarray, physical)
    coarray_holes: Optional[np.ndarray] = None

    # Segments and holes (one-sided analysis)
    largest_contiguous_segment: Optional[np.ndarray] = None       # non-negative lags contiguous window
    all_contiguous_segments: Optional[List[np.ndarray]] = None
    segment_lengths: Optional[List[int]] = None
    segment_ranges: Optional[List[List[int]]] = None              # [[L1, L2], ...]
    max_detectable_sources: Optional[int] = None

    num_holes: Optional[int] = None
    missing_virtual_positions: Optional[np.ndarray] = None            # all one-sided holes up to max(+)
    missing_virtual_positions_below_A: Optional[np.ndarray] = None    # canonical (< A) holes if applicable

    # Weighting
    weight_table: Optional[pd.DataFrame] = None  # columns: Lag, Weight
    weight_dict: Optional[Dict[int, int]] = None # {lag: count}

    # Misc / meta
    segmentation: Optional[List[np.ndarray]] = None
    num_segmentation: Optional[int] = None
    aperture: Optional[int] = None  # two-sided span (max lag - min lag), integer units

    # Presentation
    performance_summary_table: Optional[pd.DataFrame] = None


class BaseArrayProcessor(abc.ABC):
    """
    Abstract base class for MIMO array geometry analysis.
    
    This class defines the standardized 7-step analysis pipeline for computing
    and analyzing difference coarrays. All array processors (ULA, Nested, Z1-Z6)
    inherit from this class and implement the required abstract methods.
    
    **Analysis Pipeline (7 Steps):**
        1. compute_array_spacing() - Define physical sensor layout
        2. compute_all_differences() - Calculate N² pairwise differences
        3. analyze_coarray() - Identify unique positions, holes, segments
        4. compute_weight_distribution() - Count frequency of each lag
        5. analyze_contiguous_segments() - Find longest hole-free segments
        6. analyze_holes() - Identify missing positions in virtual array
        7. generate_performance_summary() - Create metrics comparison table
    
    **Abstract Methods (must be implemented by subclasses):**
        - compute_array_spacing(): Define sensor positions
        - compute_all_differences(): Compute difference coarray (n_i - n_j)
        - analyze_coarray(): Analyze unique positions and holes
        - compute_weight_distribution(): Count lag frequencies
        - analyze_contiguous_segments(): Find contiguous segments
        - analyze_holes(): Identify missing positions
        - generate_performance_summary(): Create performance metrics table
        - plot_coarray(): Visualize virtual array (optional)
    
    **Attributes:**
        data (ArraySpec): Container for all analysis results with 47 attributes
    
    **Usage Example:**
        >>> from geometry_processors.z5_processor_ import Z5ArrayProcessor
        >>> processor = Z5ArrayProcessor(N=7, d=1.0)
        >>> results = processor.run_full_analysis()
        >>> print(results.performance_summary_table.to_markdown(index=False))
    
    **Implementation Pattern:**
        class CustomArrayProcessor(BaseArrayProcessor):
            def __init__(self, custom_params):
                positions = # compute sensor positions
                super().__init__(
                    name="CustomArray",
                    array_type="Custom",
                    sensor_positions=positions,
                    d=1.0
                )
            
            def compute_all_differences(self):
                # Core algorithm: compute n_i - n_j for all pairs
                pass
            
            # ... implement remaining 7 abstract methods
    
    See Also:
        - ArraySpec: Data container with 47 pre-defined attributes
        - ULArrayProcessor: Uniform Linear Array implementation
        - Z5ArrayProcessor: Advanced weight-constrained sparse array
    """

    def __init__(self, name: str, array_type: str, sensor_positions: List[int], d: float):
        """
        Initialize array processor with basic configuration.
        
        Args:
            name (str): Human-readable array name (e.g., "Array Z5 (N=7)")
            array_type (str): Array category (e.g., "Weight-Constrained Sparse")
            sensor_positions (List[int]): Physical sensor positions (integer grid)
            d (float): Base spacing multiplier (physical units, typically wavelength/2)
        
        Initializes:
            self.data (ArraySpec): Empty data container with identity populated
        """
        self.data = ArraySpec(
            name=name,
            array_type=array_type,
            sensors_positions=list(map(int, sensor_positions)),
            num_sensors=int(len(sensor_positions)),
            sensor_spacing=float(d),
        )

    # ---------- Pipeline ----------

    def run_full_analysis(self, verbose: bool = True) -> ArraySpec:
        """
        Execute complete 7-step analysis pipeline.
        
        This is the main entry point for array analysis. It orchestrates all
        analysis steps in the correct sequence and returns populated results.
        
        **Pipeline Execution Order:**
            1. compute_array_spacing() - Define physical layout
            2. compute_all_differences() - Compute N² differences
            3. analyze_coarray() - Extract unique positions
            4. compute_weight_distribution() - Count lag frequencies
            5. analyze_contiguous_segments() - Find segments
            6. analyze_holes() - Identify missing lags
            7. generate_performance_summary() - Create metrics table
            8. plot_coarray() - Optional visualization
        
        Args:
            verbose (bool): If True, print progress messages. Default: True
        
        Returns:
            ArraySpec: Complete analysis results with all 47 attributes populated
        
        Raises:
            NotImplementedError: If any required abstract method not implemented
            ValueError: If invalid parameters detected during analysis
        
        Usage:
            >>> processor = Z5ArrayProcessor(N=7, d=1.0)
            >>> results = processor.run_full_analysis(verbose=False)
            >>> print(f"K_max = {results.max_detectable_sources}")
            K_max = 21
        
        Note:
            This method populates self.data (ArraySpec) sequentially. Each step
            depends on previous steps, so execution order is critical.
        """
        if verbose:
            # Safety guard: fall back to processor's own attributes if data class missing fields
            name = getattr(self.data, "name", None) or getattr(self, "name", "Array")
            atype = getattr(self.data, "array_type", None) or getattr(self, "array_type", "Array")
            print(f"--- Starting analysis for {name} ({atype}) ---")
        self.compute_array_spacing()
        self.compute_all_differences()
        self.analyze_coarray()
        self.compute_weight_distribution()
        self.analyze_contiguous_segments()
        self.analyze_holes()
        self.generate_performance_summary()
        # Optional plot from subclass
        self.plot_coarray()
        if verbose:
            print("--- Analysis Complete ---\n")
        return self.data

    # ---------- Steps (defaults may be overridden) ----------

    def compute_array_spacing(self):
        """
        Step 1: Define physical sensor layout.
        
        Validates and stores sensor count. Physical spacing (d) is already set
        in __init__. Subclasses may override to compute custom layouts.
        
        Populates:
            self.data.num_sensors (int): Total physical sensors (N)
        
        Note:
            Default implementation uses sensors_positions from __init__.
            Override for custom array generation logic.
        """
        # Already stored, but keep method to align interface
        self.data.num_sensors = int(len(self.data.sensors_positions))
        # self.data.sensor_spacing is set in __init__
        return

    def compute_all_differences(self):
        """
        Step 2: Compute N² pairwise differences (difference coarray).
        
        Calculates all pairwise differences (n_i - n_j) for i,j ∈ [0, N-1]
        and normalizes to integer lag units by dividing by spacing d.
        
        **Algorithm:**
            1. Form all N² pairs (i, j)
            2. Compute grid[j] - grid[i] for each pair
            3. Normalize: lag = round((grid[j] - grid[i]) / d)
            4. Store with duplicates (two-sided: includes ±lags)
        
        Populates:
            self.data.all_differences_with_duplicates (np.ndarray):
                N² integer lags including duplicates
        
        Mathematical Background:
            Virtual sensor at lag m exists if ∃(i,j): n_j - n_i = m
            Weight w(m) = |{(i,j): n_j - n_i = m}|
        
        Note:
            This is the core mathematical operation. All subsequent analysis
            depends on this difference set. Duplicates are preserved to enable
            weight distribution computation in Step 4.
        """
        grid = np.asarray(self.data.sensors_positions, dtype=float)
        # physical locations = grid * d  (we only need grid differences divided by d => integer lags)
        # Use full difference set with duplicates (two-sided).
        diffs = []
        for i in range(grid.size):
            for j in range(grid.size):
                # physical difference divided by d equals (grid[j]-grid[i])
                diffs.append(grid[j] - grid[i])
        diffs = np.asarray(diffs, dtype=float)

        # Normalize to integer lag grid (divide by d and round)
        d = self.data.sensor_spacing if self.data.sensor_spacing != 0 else 1.0
        lags = np.rint(diffs / d).astype(int)

        self.data.all_differences_with_duplicates = lags

    def analyze_coarray(self):
        """
        Step 3: Analyze unique coarray positions and identify virtual elements.
        
        Extracts unique lags from difference set and computes:
        - Two-sided unique positions (sorted)
        - Virtual-only positions (not in physical array)
        - Coarray aperture (span from min to max lag)
        
        **Algorithm:**
            1. Extract unique values from all_differences_with_duplicates
            2. Sort to get ordered coarray positions
            3. Identify virtual-only: coarray ∖ physical
            4. Compute aperture: max_lag - min_lag
        
        Populates:
            self.data.unique_differences (np.ndarray): Sorted unique lags
            self.data.num_unique_positions (int): Count Mv (virtual sensors)
            self.data.coarray_positions (np.ndarray): Alias for unique_differences
            self.data.physical_positions (np.ndarray): Original sensor positions
            self.data.virtual_only_positions (np.ndarray): Lags not in physical
            self.data.aperture (int): Two-sided span (max - min)
        
        Example:
            Physical: [0, 5, 8]
            Differences: {-8, -5, -3, 0, 3, 5, 8}
            Virtual-only: {-8, -5, -3, 3, 5, 8}
            Aperture: 8 - (-8) = 16
        """
        lags = np.asarray(self.data.all_differences_with_duplicates, dtype=int)
        uniq = np.unique(lags)

        self.data.unique_differences = uniq
        self.data.num_unique_positions = int(uniq.size)
        self.data.coarray_positions = uniq

        self.data.physical_positions = np.asarray(self.data.sensors_positions, dtype=int)
        self.data.virtual_only_positions = np.setdiff1d(uniq, self.data.physical_positions)

        # Aperture (two-sided span)
        if uniq.size:
            self.data.aperture = int(uniq.max() - uniq.min())
        else:
            self.data.aperture = 0

    def compute_weight_distribution(self):
        """
        Step 4: Compute weight distribution (lag frequency counts).
        
        Counts how many times each lag appears in the difference set.
        Weight w(m) = number of sensor pairs (i,j) satisfying n_j - n_i = m.
        
        **Properties:**
            - w(0) = N (diagonal pairs, always equals sensor count)
            - w(m) = w(-m) (two-sided symmetry)
            - Higher weights at small lags → better DOA estimation accuracy
        
        Populates:
            self.data.weight_table (pd.DataFrame): 
                Columns ['Lag', 'Weight'], sorted by lag
            self.data.weight_dict (Dict[int, int]): 
                Quick lookup {lag: count}
        
        Usage:
            >>> wt = results.weight_dict
            >>> print(f"w(1) = {wt.get(1, 0)}")
            w(1) = 8
        
        Note:
            Weight distribution affects estimation accuracy. Specialized arrays
            (Z4, Z5, Z6) are designed with constraints like w(1)=w(2)=0.
        """
        lags = np.asarray(self.data.all_differences_with_duplicates, dtype=int)
        lvals, counts = np.unique(lags, return_counts=True)
        df = pd.DataFrame({"Lag": lvals, "Weight": counts})
        df.sort_values("Lag", inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
        self.data.weight_table = df
        self.data.weight_dict = dict(zip(lvals.tolist(), counts.tolist()))

    def analyze_contiguous_segments(self):
        """
        Step 5: Find contiguous segments in virtual array.
        
        Identifies all hole-free (contiguous) segments in the non-negative
        coarray positions and determines the longest segment. The longest
        contiguous segment defines K_max (maximum detectable sources).
        
        **Algorithm:**
            1. Extract non-negative coarray positions [0, ∞)
            2. Sort positions
            3. Split at gaps > 1 (e.g., [0,1,2] gap [4,5,6])
            4. Identify longest segment L
            5. Compute K_max = floor(L/2)
            6. Find holes: missing positions in [0, max]
        
        Populates:
            self.data.all_contiguous_segments (List[np.ndarray]): 
                All segments (e.g., [[0,1,2,3], [5,6,7]])
            self.data.segment_lengths (List[int]): 
                Length of each segment
            self.data.segment_ranges (List[List[int]]): 
                [[start, end], ...] for each segment
            self.data.largest_contiguous_segment (np.ndarray): 
                Longest segment (tie-break: earliest)
            self.data.max_detectable_sources (int): 
                K_max = L // 2 where L is longest segment length
            self.data.missing_virtual_positions (np.ndarray): 
                Holes in [0, max_lag]
            self.data.num_holes (int): 
                Count of holes
        
        Mathematical Background:
            MUSIC requires contiguous virtual ULA of length L to estimate
            K = floor(L/2) sources. Holes limit DOF.
        
        Example:
            Virtual positions: [0, 1, 2, 3, 5, 6, 7]
            Segments: [[0,1,2,3], [5,6,7]]
            Longest: [0,1,2,3] (L=4)
            K_max: 4 // 2 = 2
            Holes: [4]
        """
        uniq = np.asarray(self.data.coarray_positions, dtype=int)
        nonneg = np.sort(uniq[uniq >= 0])

        segments: List[np.ndarray] = []
        if nonneg.size == 0:
            seg = np.array([], dtype=int)
            segments = [seg]
        else:
            # split on gaps greater than 1
            breaks = np.where(np.diff(nonneg) > 1)[0]
            start = 0
            for b in breaks:
                segments.append(nonneg[start:b+1])
                start = b + 1
            segments.append(nonneg[start:])

        self.data.all_contiguous_segments = segments
        self.data.segment_lengths = [int(len(s)) for s in segments]
        self.data.segment_ranges = [[int(s[0]), int(s[-1])] for s in segments if s.size]
        # largest by length (tie-break: earliest)
        if segments:
            seg = max(segments, key=lambda s: (len(s), -s[0] if s.size else 0))
        else:
            seg = np.array([], dtype=int)

        self.data.largest_contiguous_segment = seg
        self.data.max_detectable_sources = int(len(seg) // 2)
        # generic holes (one-sided up to max positive)
        if nonneg.size:
            ideal = np.arange(0, int(nonneg.max()) + 1, dtype=int)
            holes = np.setdiff1d(ideal, nonneg, assume_unique=False)
        else:
            holes = np.array([], dtype=int)
        self.data.missing_virtual_positions = holes
        self.data.num_holes = int(holes.size)

    def analyze_holes(self):
        """
        Step 6: Analyze missing positions (holes) in virtual array.
        
        Identifies positions missing from the virtual array within the
        defined range. Default implementation uses holes computed in
        analyze_contiguous_segments(). Subclasses may override to compute
        array-specific canonical holes (e.g., holes below aperture A).
        
        Populates:
            self.data.missing_virtual_positions_below_A (np.ndarray): 
                Custom holes below aperture (subclass-specific)
        
        Note:
            Default implementation is a pass-through. Specialized arrays
            (Z4, Z5) override to compute canonical hole sets.
        """
        self.data.missing_virtual_positions_below_A = None
        return

    def generate_performance_summary(self):
        """
        Step 7: Generate performance metrics summary table.
        
        Creates comprehensive table of key performance indicators for array
        comparison. Includes geometry metrics, DOF, and weight distribution.
        
        **Metrics Included:**
            - Physical Sensors (N): Total physical sensors
            - Virtual Elements (Mv): Unique coarray positions
            - Virtual-only Elements: Positions not in physical array
            - Coarray Aperture: Two-sided span (max_lag - min_lag)
            - Max Positive Lag: Maximum one-sided lag
            - Contiguous Segment Length (L): Longest hole-free segment
            - Maximum Detectable Sources (K_max): floor(L/2)
            - Holes: Count of missing positions
            - w(0), w(1), w(2), w(3): Weight distribution at small lags
        
        Populates:
            self.data.performance_summary_table (pd.DataFrame):
                Two columns: ['Metrics', 'Value']
        
        Usage:
            >>> results = processor.run_full_analysis()
            >>> print(results.performance_summary_table.to_markdown(index=False))
            | Metrics                     | Value |
            |-----------------------------|-------|
            | Physical Sensors (N)        | 7     |
            | Virtual Elements (Mv)       | 43    |
            | Maximum Detectable Sources  | 21    |
        
        Note:
            Subclasses may override to add array-specific metrics.
            Default implementation covers standard coarray properties.
        """
        d = self.data
        rows = [
            ("Physical Sensors (N)", d.num_sensors),
            ("Virtual Elements (Unique Lags)", d.num_unique_positions or 0),
            ("Virtual-only Elements", int(len(d.virtual_only_positions)) if d.virtual_only_positions is not None else 0),
            ("Coarray Aperture (two-sided span, lags)", d.aperture if d.aperture is not None else 0),
            ("Max Positive Lag (one-sided)", int(np.max(d.coarray_positions)) if d.coarray_positions is not None and d.coarray_positions.size else 0),
            ("Contiguous Segment Length (L, one-sided)", int(len(d.largest_contiguous_segment)) if d.largest_contiguous_segment is not None else 0),
            ("Maximum Detectable Sources (K_max)", d.max_detectable_sources if d.max_detectable_sources is not None else 0),
            ("Holes (one-sided)", int(d.num_holes) if d.num_holes is not None else 0),
        ]
        # Add a few weights if available
        if d.weight_dict:
            rows.extend([
                ("Weight at Lag 0 (w(0))", d.weight_dict.get(0, 0)),
                ("Weight at Lag 1 (w(1))", d.weight_dict.get(1, 0)),
                ("Weight at Lag 2 (w(2))", d.weight_dict.get(2, 0)),
                ("Weight at Lag 3 (w(3))", d.weight_dict.get(3, 0)),
            ])
        self.data.performance_summary_table = pd.DataFrame(rows, columns=["Metrics", "Value"])

    # ---------- Optional visualization ----------

    def plot_coarray(self):
        """
        Optional visualization of virtual array layout.
        
        Subclasses can override to provide custom visualization (matplotlib plots,
        ASCII art, etc.). Default implementation does nothing.
        
        Usage:
            Override in subclass for custom visualization:
            
            def plot_coarray(self):
                import matplotlib.pyplot as plt
                plt.stem(self.data.coarray_positions, ...)
                plt.savefig(f'results/plots/{self.data.name}.png')
        
        Note:
            This method is called last in run_full_analysis() so all data
            is available for plotting. Not required for analysis.
        """
        return

    # ---------- Abstract marker for subclasses (if they require custom behavior) ----------

    @abc.abstractmethod
    def __repr__(self) -> str:
        """
        Return string representation of the array processor.
        
        Author: Hossein Molhem
        
        Args:
            None
            
        Returns:
            str: String representation including class name and array name
            
        Raises:
            None
        """
        return f"<{self.__class__.__name__}: {self.data.name}>"
