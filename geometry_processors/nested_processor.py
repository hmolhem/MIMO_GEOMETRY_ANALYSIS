# geometry_processors/nested_processor.py

import numpy as np
import pandas as pd
# Assuming ArraySpec and BaseArrayProcessor are in bases_classes.py
from .bases_classes import BaseArrayProcessor 

class NestedArrayProcessor(BaseArrayProcessor):
    """
    Concrete implementation for a standard Nested Array (NA).
    Achieves O(N^2) degrees of freedom with a contiguous coarray.
    """
    def __init__(self, N1: int, N2: int, d: int = 1):
        
        # 1. Calculate the sensor positions
        self.N1 = N1
        self.N2 = N2
        self.M = d
        
        # Subarray 1: Dense ULA [0, d, ..., (N1-1)d]
        P1 = np.arange(N1) * d
        
        # Subarray 2: Sparse ULA [ (N1+1)d, 2(N1+1)d, ..., N2(N1+1)d ]
        spacing = (N1 + 1) * d
        P2 = np.arange(1, N2 + 1) * spacing
        
        # Combine and ensure unique positions
        positions = np.unique(np.concatenate((P1, P2)))
        
        # Total number of sensors (N = N1 + N2, assuming no overlap for d=1)
        N = N1 + N2
        
        super().__init__(
            name=f"Nested Array N={N}", 
            array_type="Nested Array", 
            sensor_positions=positions.tolist(),
            d=d
        )
        
        # Recalculate N based on unique positions if necessary, but typically N=N1+N2
        self.data.num_sensors = N 
        self.data.aperture = positions[-1] - positions[0]

    def __repr__(self):
        """String representation of the Nested Array processor."""
        return f"NestedArrayProcessor(N1={self.N1}, N2={self.N2}, d={self.M})"

    # ------------------------------------------------------------------
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ------------------------------------------------------------------

    def compute_array_spacing(self):
        """Sets the fundamental unit spacing."""
        self.data.sensor_spacing = self.M

    def compute_all_differences(self):
        """
        3.1, 3.2, 3.3. Computes all N^2 difference elements.
        (FIXED IMPLEMENTATION)
        """
        positions = self.data.sensors_positions
        N = self.data.num_sensors
        
        # --- CORRECT IMPLEMENTATION ---
        # Create meshgrids directly from the positions array
        n_i, n_j = np.meshgrid(positions, positions)
        
        # Compute differences and flatten
        diff_flat = (n_i - n_j).flatten()
        ni_flat = n_i.flatten()
        nj_flat = n_j.flatten()
        # --- END OF CORRECT IMPLEMENTATION ---
        
        # 3.2. all_differences_with_duplicates
        self.data.all_differences_with_duplicates = diff_flat

        # 3.3. total_diff_computations
        self.data.total_diff_computations = N * N
        
        # 3.1. all_differences_table
        # Note: S_i and S_j index computation is omitted for brevity as in the previous example
        self.data.all_differences_table = pd.DataFrame({
            'n_i': ni_flat,
            'n_j': nj_flat,
            'Difference': diff_flat
        })

    def analyze_coarray(self):
        """
        3.4-3.11. Processes differences. NA coarray is hole-free up to a large lag.
        """
        all_diffs = self.data.all_differences_with_duplicates

        # 3.4. unique_differences (Coarray positions)
        unique_diffs = np.unique(all_diffs)
        self.data.unique_differences = unique_diffs

        # 3.5. num_unique_positions
        self.data.num_unique_positions = len(unique_diffs)

        # 3.6, 3.8. physical_positions and coarray_positions
        self.data.physical_positions = self.data.sensors_positions
        self.data.coarray_positions = unique_diffs

        # 3.7. virtual_only_positions
        self.data.virtual_only_positions = np.setdiff1d(
            self.data.coarray_positions, 
            self.data.physical_positions
        )

        # 3.9-3.11. Holes and Segmentation
        max_lag = (self.N1 + 1) * self.N2 - 1
        
        # Check for non-negative holes
        ideal_positive_range = np.arange(0, max_lag + 1)
        positive_coarray = unique_diffs[unique_diffs >= 0]
        
        # The ideal NA coarray is contiguous from -max_lag to max_lag
        holes = np.setdiff1d(ideal_positive_range, positive_coarray)
        
        self.data.coarray_holes = np.concatenate((-holes, holes))
        self.data.segmentation = [unique_diffs] if len(unique_diffs) > 0 else []
        self.data.num_segmentation = 1
        
    def plot_coarray(self):
        """
        3.12. Placeholder for visualization.
        """
        print("\n[Plotting Placeholder]: Nested Array Visualization")
        max_lag = self.data.coarray_positions[-1]
        print(f"Coarray Segment: [{-max_lag}, {max_lag}]")
        print(f"Length: {len(self.data.coarray_positions)} elements")
        print("-" * 50)
        
    def compute_weight_distribution(self):
        """
        Count the occurrences of each integer lag (w(k)).
        For ULA, the closed form is w(k) = N_total - |k| for k in [-(N-1)..(N-1)].
        """

        lags = np.asarray(self.data.all_differences_with_duplicates, dtype=int)
        uniq, counts = np.unique(lags, return_counts=True)

        # Store a weight vector aligned to unique_differences
        weight_dict = dict(zip(uniq.tolist(), counts.tolist()))
        weights = [weight_dict.get(int(k), 0) for k in self.data.unique_differences]
        self.data.coarray_weight_distribution = np.asarray(weights, dtype=int)

        # Also keep a small table (named columns for robustness)
        self.data.weight_table = pd.DataFrame({
            "Lag": uniq.astype(int),
            "Weight": counts.astype(int)
        })


    def analyze_contiguous_segments(self):
        lags = np.asarray(self.data.unique_differences, dtype=int)
        lpos = np.sort(lags[lags >= 0])

        if lpos.size:
            dif = np.diff(lpos)
            br = np.where(dif != 1)[0]
            starts = np.insert(br + 1, 0, 0)
            ends = np.append(br, lpos.size - 1)
            segments = [lpos[s:e+1] for s, e in zip(starts, ends)]
        else:
            segments = []

        self.data.all_contiguous_segments = segments
        largest = max(segments, key=len) if segments else np.array([], dtype=int)
        self.data.largest_contiguous_segment = largest
        self.data.shortest_contiguous_segment = min(segments, key=len) if segments else np.array([], dtype=int)
        self.data.segment_lengths = [int(len(s)) for s in segments]

        L = int(len(largest))
        self.data.max_detectable_sources = L // 2
        self.data.segment_ranges = [(int(largest[0]), int(largest[-1]))] if L > 0 else []


    def analyze_holes(self):
        lags = np.asarray(self.data.unique_differences, dtype=int)
        lpos = np.sort(lags[lags >= 0])

        if lpos.size == 0:
            self.data.missing_virtual_positions = np.array([], dtype=int)
            self.data.num_holes = 0
            return

        ideal = np.arange(lpos.min(), lpos.max() + 1, dtype=int)
        holes = np.setdiff1d(ideal, lpos, assume_unique=False)

        self.data.missing_virtual_positions = holes
        self.data.num_holes = int(holes.size)


    def generate_performance_summary(self):
        """7. Generates the final performance summary table."""
        data = self.data
        weights_dict = dict(zip(data.unique_differences, data.coarray_weight_distribution))
        
        aperture = data.coarray_positions[-1] - data.coarray_positions[0] if data.num_unique_positions > 0 else 0

        metrics = [
            'Physical Sensors (N)',
            'Virtual Elements (Unique Lags)',
            'Virtual-only Elements',
            'Coarray Aperture',
            'Contiguous Segment Length (L)',
            'Maximum Detectable Sources (K_max)',
            'Holes',
            'Weight at Lag 1 (w(1))',
            'Weight at Lag 2 (w(2))',
            'Weight at Lag 3 (w(3))'
        ]
        values = [
            data.num_sensors,
            data.num_unique_positions,
            len(data.virtual_only_positions),
            aperture,
            len(data.largest_contiguous_segment),
            data.max_detectable_sources,
            data.num_holes,
            weights_dict.get(1, 0),
            weights_dict.get(2, 0),
            weights_dict.get(3, 0)
        ]

        self.data.performance_summary_table = pd.DataFrame({'Metrics': metrics, 'Value': values})