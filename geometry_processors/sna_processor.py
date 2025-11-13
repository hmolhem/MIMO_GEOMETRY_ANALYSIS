# geometry_processors/sna_processor.py
"""
Super Nested Array (SNA) Processor - 3rd Order

Implements SNA3 from Liu & Vaidyanathan (2016):
"Super nested arrays: Linear sparse arrays with reduced mutual coupling—Part I"

Key Properties:
- Same DOFs as standard nested array
- Reduced mutual coupling (smaller w(2), w(3))
- O(N²) aperture
- Construction: Expands nested array with specific gaps

Reference:
    C.-L. Liu and P. P. Vaidyanathan, "Super nested arrays: Linear sparse
    arrays with reduced mutual coupling—Part I: Fundamentals," IEEE Trans.
    Signal Process., vol. 64, no. 15, pp. 3997–4012, Aug. 2016.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List
from .bases_classes import BaseArrayProcessor


class SNA3ArrayProcessor(BaseArrayProcessor):
    """
    Super Nested Array (3rd order) Processor.
    
    Construction (from paper [9]):
        SNA3 = P1 ∪ P2 ∪ P3
        where:
        P1 = {1, 2, ..., N1}                    (dense ULA)
        P2 = (N1+1) * {1, 2, ..., N2}           (sparse ULA)
        P3 = (N1+1)(N2+1) * {1, 2, ..., N3}     (very sparse ULA)
    
    Total sensors: N = N1 + N2 + N3
    Aperture: A ≈ (N1+1)(N2+1)(N3+1) - 1
    DOFs: Similar to nested array with same N
    
    Advantages over nested array:
    - Reduced w(2) and w(3) (better mutual coupling mitigation)
    - Same or more DOFs
    - Slightly larger aperture
    
    Parameters
    ----------
    N1 : int
        Number of sensors in dense subarray P1
    N2 : int
        Number of sensors in medium-sparse subarray P2
    N3 : int
        Number of sensors in very-sparse subarray P3
    d : float, optional
        Physical spacing multiplier (default: 1.0)
        Final positions = integer_positions * d
    
    Examples
    --------
    >>> # Paper example: N=16 for comparison
    >>> proc = SNA3ArrayProcessor(N1=7, N2=6, N3=3, d=0.5)
    >>> results = proc.run_full_analysis()
    >>> print(results.performance_summary_table.to_markdown(index=False))
    
    >>> # Access specific metrics
    >>> print(f"Aperture: {results.coarray_aperture}")
    >>> print(f"Max DOAs: {results.max_detectable_sources}")
    
    Notes
    -----
    - For N=16 sensors, paper suggests N1=7, N2=6, N3=3
    - Results in reduced w(2) compared to nested array
    - Check paper Table IV for expected values at N=16
    """
    
    def __init__(self, N1: int, N2: int, N3: int, d: float = 1.0):
        """
        Initialize SNA3 array processor.
        
        Parameters
        ----------
        N1 : int
            Dense subarray size (typically largest)
        N2 : int
            Medium subarray size
        N3 : int
            Sparse subarray size (typically smallest)
        d : float
            Physical spacing multiplier
        """
        self.N1 = int(N1)
        self.N2 = int(N2)
        self.N3 = int(N3)
        self.d = float(d)
        self.N_total = self.N1 + self.N2 + self.N3
        
        # Build sensor positions (integer grid indices)
        positions_grid = self._build_sna3_positions()
        
        super().__init__(
            name=f"SNA3 (N1={N1}, N2={N2}, N3={N3})",
            array_type="Super Nested Array (3rd Order)",
            sensor_positions=positions_grid.tolist(),
            d=self.d,
        )
        
        # Initialize result containers
        self.data.coarray_positions = None
        self.data.largest_contiguous_segment = None
        self.data.missing_virtual_positions = None
        self.data.weight_table = pd.DataFrame(columns=["Lag", "Weight"])
    
    def __repr__(self) -> str:
        """
        Return string representation of the SNA3 array processor.
        
        Author: Hossein Molhem
        
        Args:
            None
            
        Returns:
            str: String representation with N1, N2, N3, and spacing parameters
            
        Raises:
            None
        """
        return f"SNA3ArrayProcessor(N1={self.N1}, N2={self.N2}, N3={self.N3}, d={self.d})"
    
    def _build_sna3_positions(self) -> np.ndarray:
        """
        Build SNA3 sensor positions according to paper construction.
        
        Returns
        -------
        np.ndarray
            Integer grid positions (before physical spacing multiplication)
        """
        # P1: Dense ULA [1, 2, ..., N1]
        P1 = np.arange(1, self.N1 + 1, dtype=int)
        
        # P2: Sparse ULA (N1+1)*[1, 2, ..., N2]
        P2 = (self.N1 + 1) * np.arange(1, self.N2 + 1, dtype=int)
        
        # P3: Very sparse ULA (N1+1)(N2+1)*[1, 2, ..., N3]
        P3 = (self.N1 + 1) * (self.N2 + 1) * np.arange(1, self.N3 + 1, dtype=int)
        
        # Combine all subarrays
        positions = np.concatenate([P1, P2, P3])
        
        # Add reference sensor at origin (position 0)
        positions = np.concatenate([[0], positions])
        
        # Sort to ensure ascending order
        positions = np.sort(positions)
        
        return positions
    
    # ========== Abstract Method Implementations ==========
    
    def analyze_geometry(self):
        """Analyze physical array geometry properties."""
        # All geometry analysis done in __init__ via base class
        pass
    
    def analyze_coarray(self):
        """Compute difference coarray and its properties."""
        positions = np.asarray(self.data.sensors_positions, dtype=int)
        
        # Compute all pairwise differences (difference coarray)
        diff = positions[:, None] - positions[None, :]
        lags = np.unique(diff.ravel())
        
        self.data.coarray_positions = lags
        
        # Weight distribution (count occurrences of each lag)
        unique_lags, counts = np.unique(diff.ravel(), return_counts=True)
        self.data.weight_table = pd.DataFrame({
            "Lag": unique_lags.astype(int),
            "Weight": counts.astype(int)
        })
        
        # Find largest contiguous segment in positive lags
        positive_lags = lags[lags >= 0]
        segment = self._find_largest_contiguous_segment(positive_lags)
        self.data.largest_contiguous_segment = segment
        
        # Identify holes (missing lags in range [0, max_lag])
        if len(positive_lags) > 0:
            max_lag = int(positive_lags.max())
            expected_lags = np.arange(0, max_lag + 1)
            holes = np.setdiff1d(expected_lags, positive_lags)
            self.data.missing_virtual_positions = holes
        else:
            self.data.missing_virtual_positions = np.array([])
    
    def _find_largest_contiguous_segment(self, lags: np.ndarray) -> np.ndarray:
        """
        Find the largest contiguous segment in sorted lags.
        
        Parameters
        ----------
        lags : np.ndarray
            Sorted array of lag values
        
        Returns
        -------
        np.ndarray
            Largest contiguous segment [start, start+1, ..., end]
        """
        if len(lags) == 0:
            return np.array([])
        
        # Find consecutive sequences
        best_start = 0
        best_length = 1
        current_start = 0
        current_length = 1
        
        for i in range(1, len(lags)):
            if lags[i] == lags[i-1] + 1:
                # Consecutive
                current_length += 1
            else:
                # Gap found
                if current_length > best_length:
                    best_length = current_length
                    best_start = current_start
                current_start = i
                current_length = 1
        
        # Check final segment
        if current_length > best_length:
            best_length = current_length
            best_start = current_start
        
        return lags[best_start:best_start + best_length]
    
    def compute_weight_distribution(self):
        """Compute weight distribution (already done in analyze_coarray)."""
        pass
    
    def generate_performance_summary(self):
        """Generate performance metrics summary table."""
        lags = self.data.coarray_positions
        if lags is None or len(lags) == 0:
            lags = np.array([])
        
        segment = self.data.largest_contiguous_segment
        if segment is None or len(segment) == 0:
            segment = np.array([])
        
        # Extract metrics
        N = self.N_total
        num_unique_lags = len(np.unique(lags)) if len(lags) > 0 else 0
        num_virtual_only = num_unique_lags - N
        aperture = int(lags.max() - lags.min()) if len(lags) > 0 else 0
        
        L = len(segment)  # Contiguous segment length
        K_max = L // 2    # Maximum detectable sources
        
        L1 = int(segment[0]) if len(segment) > 0 else None
        L2 = int(segment[-1]) if len(segment) > 0 else None
        
        # Weight values at small lags
        wt_df = self.data.weight_table
        wt = {int(r["Lag"]): int(r["Weight"]) for _, r in wt_df.iterrows()} if not wt_df.empty else {}
        
        # Holes
        holes = self.data.missing_virtual_positions
        num_holes = len(holes) if holes is not None else 0
        
        # Build summary table
        rows = [
            ["Physical Sensors (N)", N],
            ["Virtual Elements (Unique Lags)", num_unique_lags],
            ["Virtual-only Elements", max(0, num_virtual_only)],
            ["Coarray Aperture (two-sided span)", aperture],
            ["Contiguous Segment Length (L)", L],
            ["Maximum Detectable Sources (K_max)", K_max],
            ["Holes in Coarray", num_holes],
            ["Weight at Lag 0 (w(0))", wt.get(0, 0)],
            ["Weight at Lag 1 (w(1))", wt.get(1, 0)],
            ["Weight at Lag 2 (w(2))", wt.get(2, 0)],
            ["Weight at Lag 3 (w(3))", wt.get(3, 0)],
            ["Weight at Lag 4 (w(4))", wt.get(4, 0)],
            ["Weight at Lag 5 (w(5))", wt.get(5, 0)],
            ["Segment Range [L1:L2]", f"[{L1}:{L2}]" if L else "NA"],
        ]
        
        self.data.performance_summary_table = pd.DataFrame(
            rows, columns=["Metrics", "Value"]
        )
    
    def plot_coarray(self):
        """Display coarray visualization (console-based)."""
        print(f"\n{'='*70}")
        print(f"  {self.data.name}")
        print(f"{'='*70}")
        print(f"Total Sensors: N = {self.N_total} (N1={self.N1}, N2={self.N2}, N3={self.N3})")
        print(f"Physical Positions: {self.data.sensors_positions}")
        
        if self.data.performance_summary_table is not None:
            print(f"\n{self.data.performance_summary_table.to_string(index=False)}")
        
        # Show weight distribution for small lags
        if self.data.weight_table is not None and not self.data.weight_table.empty:
            print(f"\nCoarray Weight Distribution (first 10 lags):")
            wt_df = self.data.weight_table
            small_lags = wt_df[wt_df['Lag'].abs() <= 10].sort_values('Lag')
            print(small_lags.to_string(index=False))
        
        print(f"{'='*70}\n")


# ========== Utility Function ==========

def optimize_sna3_parameters(N_total: int) -> tuple:
    """
    Find optimal N1, N2, N3 for given total sensors.
    
    Heuristic from paper: Distribute sensors to maximize DOFs
    while keeping subarrays balanced.
    
    Parameters
    ----------
    N_total : int
        Total number of sensors
    
    Returns
    -------
    tuple
        (N1, N2, N3) optimal parameters
    
    Examples
    --------
    >>> N1, N2, N3 = optimize_sna3_parameters(16)
    >>> print(f"N=16: N1={N1}, N2={N2}, N3={N3}")
    """
    # Paper suggests roughly: N1 ≈ N/2, N2 ≈ N/3, N3 ≈ N/6
    # Adjust to sum to N_total
    N1 = max(1, N_total // 2)
    N2 = max(1, N_total // 3)
    N3 = max(1, N_total - N1 - N2)
    
    # Ensure positive
    if N3 <= 0:
        N1 = N_total // 2
        N2 = N_total - N1
        N3 = 0
    
    return (N1, N2, N3)


# ========== Example Usage ==========

if __name__ == "__main__":
    print("Super Nested Array (SNA3) - Example\n")
    
    # Example 1: N=16 (paper comparison)
    print("Example 1: N=16 sensors (paper Table IV)")
    proc = SNA3ArrayProcessor(N1=7, N2=6, N3=3, d=0.5)
    results = proc.run_full_analysis()
    proc.plot_coarray()
    
    # Example 2: Auto-optimize for N=20
    print("\nExample 2: N=20 sensors (auto-optimized)")
    N1, N2, N3 = optimize_sna3_parameters(20)
    proc2 = SNA3ArrayProcessor(N1=N1, N2=N2, N3=N3, d=0.5)
    results2 = proc2.run_full_analysis()
    print(f"\nOptimal parameters: N1={N1}, N2={N2}, N3={N3}")
    print(results2.performance_summary_table.to_markdown(index=False))
