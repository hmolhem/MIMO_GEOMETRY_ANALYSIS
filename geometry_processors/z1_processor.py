# geometry_processors/z1_processor.py

import numpy as np
import pandas as pd
from .bases_classes import BaseArrayProcessor 

class Z1ArrayProcessor(BaseArrayProcessor):
    """
    Implements the Weight-Constrained Sparse Array Z1.
    Z1 Array: 2-Sparse ULA of (N-1) sensors + 1 Augmented Sensor.
    Key Property: w(1) = 0.
    """
    def __init__(self, N: int, d: float = 1.0):
        if N < 3:
            raise ValueError("Z1 Array requires N >= 3 sensors for meaningful sparse definition.")
            
        self.M = d
        self.N_total = N
        
        # 1. Generate the (N-1)-sensor 2-sparse ULA segment
        # Positions: [0, 2d, 4d, ..., 2(N-2)d]
        N_sparse = N - 1
        sparse_segment = np.arange(N_sparse) * 2 * d
        
        # 2. Augment with the N-th sensor at 2N-1
        augmented_sensor = (2 * N - 1) * d
        
        # Combine and ensure ordering (positions should naturally be ordered)
        positions = np.concatenate((sparse_segment, [augmented_sensor]))
        
        super().__init__(
            name=f"Array Z1 (N={N})", 
            array_type="Weight-Constrained Sparse Array (Z1)", 
            sensor_positions=positions.tolist()
        )
        
        self.data.num_sensors = N
        self.data.aperture = positions[-1] - positions[0]


    # ------------------------------------------------------------------
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ------------------------------------------------------------------

    def compute_array_spacing(self):
        """Sets the fundamental unit spacing."""
        self.data.sensor_spacing = self.M

    def compute_all_differences(self):
        """3.1, 3.2, 3.3. Computes all N^2 difference elements."""
        positions = self.data.sensors_positions
        N = self.data.num_sensors
        
        # Create meshgrids directly from the positions array
        n_i, n_j = np.meshgrid(positions, positions)
        
        # Compute differences and flatten
        diff_flat = (n_i - n_j).flatten()
        ni_flat = n_i.flatten()
        nj_flat = n_j.flatten()
        
        self.data.all_differences_with_duplicates = diff_flat
        self.data.total_diff_computations = N * N
        
        self.data.all_differences_table = pd.DataFrame({
            'n_i': ni_flat,
            'n_j': nj_flat,
            'Difference': diff_flat
        })

    def analyze_coarray(self):
        """3.4-3.11. Processes differences to find unique elements and segments."""
        all_diffs = self.data.all_differences_with_duplicates
        unique_diffs = np.unique(all_diffs)
        
        self.data.unique_differences = unique_diffs
        self.data.num_unique_positions = len(unique_diffs)
        self.data.physical_positions = self.data.sensors_positions
        self.data.coarray_positions = unique_diffs
        
        # Virtual-only positions
        self.data.virtual_only_positions = np.setdiff1d(
            self.data.coarray_positions, 
            self.data.physical_positions
        )

        # Holes: Z1 coarray is D_z1+ = {0, 2, 3, 4, ..., 2N-4, 2N-3, 2N-1}
        # Holes are at 1 and 2N-2 (or A-1, since A=2N-1) [cite: 315]
        min_pos = self.data.coarray_positions[0]
        max_pos = self.data.coarray_positions[-1]
        
        ideal_range = np.arange(min_pos, max_pos + 1)
        
        # 3.9. Holes calculation
        holes = np.setdiff1d(ideal_range, self.data.coarray_positions)
        self.data.coarray_holes = holes
        
        # 3.10, 3.11. Segmentation: One segment from L1=-(2N-3) to L2=(2N-3) plus edges
        # The ULA segment is [2, 2N-3] for positive lags 
        self.data.segmentation = [unique_diffs] # For simplicity, treat as one block
        self.data.num_segmentation = 1

    def plot_coarray(self):
        """3.12. Placeholder for visualization."""
        print("\n[Plotting Placeholder]: Array Z1 Visualization (w(1)=0)")
        print(f"Physical Array Positions: {self.data.sensors_positions}")
        print(f"Coarray Positions (Positive): {self.data.coarray_positions[self.data.coarray_positions >= 0]}")
        print(f"Holes (Positive Lags): {self.data.coarray_holes[self.data.coarray_holes >= 0]}")
        print("Note: w(1) is guaranteed to be 0 by design.")
        print("-" * 50)
        
    def compute_weight_distribution(self):
        """4.1, 4.2. Calculates the weight (count) of each unique lag."""
        all_diffs = self.data.all_differences_with_duplicates
        lags, counts = np.unique(all_diffs, return_counts=True)
        
        weight_dict = dict(zip(lags, counts))
        
        weights = [weight_dict.get(lag, 0) for lag in self.data.unique_differences]
        self.data.coarray_weight_distribution = np.array(weights)
        
        self.data.weight_table = pd.DataFrame({
            'Lag (Difference)': lags, 
            'Count (Weight)': counts
        })

    def analyze_contiguous_segments(self):
        """5.1-5.7. Identifies the contiguous segment and K_max."""
        # For Z1, the ULA segment starts at L1=2 and ends at L2=2N-3. Length L=2N-4 
        L1 = 2
        L2 = 2 * self.N_total - 3
        L = L2 - L1 + 1
        
        # Check if L is valid (N must be >= 3)
        if L < 1:
             L = 0 
             segment = np.array([])
        else:
             segment = np.arange(L1, L2 + 1)

        # The paper uses the length of the one-sided segment [cite: 228, 229]
        self.data.all_contiguous_segments = [segment]
        self.data.largest_contiguous_segment = segment 
        self.data.shortest_contiguous_segment = segment
        self.data.segment_lengths = [L]
        self.data.max_detectable_sources = np.floor(L / 2.0).astype(int) # Dm = floor(L/2) [cite: 229, 273]
        self.data.segment_ranges = [(L1, L2)]

    def analyze_holes(self):
        """6.1, 6.2. Analyzes the missing positions."""
        self.data.missing_virtual_positions = self.data.coarray_holes
        self.data.num_holes = len(self.data.coarray_holes)

    def generate_performance_summary(self):
        """7. Generates the final performance summary table."""
        data = self.data
        weights_dict = dict(zip(data.unique_differences, data.coarray_weight_distribution))
        
        aperture = data.coarray_positions[-1] - data.coarray_positions[0] if data.num_unique_positions > 0 else 0
        
        # The paper specifically compares w(1), w(2), w(3)
        w1 = weights_dict.get(1, 0)
        w2 = weights_dict.get(2, 0)
        w3 = weights_dict.get(3, 0)

        metrics = [
            'Physical Sensors (N)',
            'Virtual Elements (Unique Lags)',
            'Virtual-only Elements',
            'Coarray Aperture (Max-Min)',
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
            self.data.segment_lengths[0] if self.data.segment_lengths else 0, # Use L from segment analysis
            data.max_detectable_sources,
            data.num_holes,
            w1,
            w2,
            w3
        ]

        self.data.performance_summary_table = pd.DataFrame({'Metrics': metrics, 'Value': values})