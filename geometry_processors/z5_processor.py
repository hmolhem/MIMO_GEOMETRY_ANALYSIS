# geometry_processors/z5_processor.py

import numpy as np
import pandas as pd
from .bases_classes import BaseArrayProcessor 

class Z5ArrayProcessor(BaseArrayProcessor):
    """
    Implements the Weight-Constrained Sparse Array Z5.
    Z5 Array: 4-Sparse ULA of (N-3) sensors + 3 Augmented Sensors.
    Key Properties: w(1)=0, w(2)=0. Designed to minimize w(3), w(4).
    Aperture: 4N-5.
    """
    def __init__(self, N: int, d: float = 1.0):
        # The paper assumes N >= 5 for general expressions to be valid
        if N < 5:
            raise ValueError("Z5 Array requires N >= 5 sensors for a valid construction.")
            
        self.M = d
        self.N_total = N
        
        # 1. Generate the (N-3)-sensor 4-sparse ULA segment
        # Positions: [0, 4d, 8d, ..., 4(N-4)d]
        N_sparse = N - 3
        sparse_segment = np.arange(N_sparse) * 4 * d
        
        # 2. Augmented sensors based on the paper's description (Sec. IV-D)
        # S0: -5d 
        # S(N-2): (4N-13)d
        # S(N-1): (4N-10)d
        augmented_sensors = np.array([-5 * d, (4 * N - 13) * d, (4 * N - 10) * d])
        
        # Combine all sensors
        positions = np.unique(np.concatenate((augmented_sensors, sparse_segment)))
        
        # Normalize the array so the smallest position is 0 (shift by +5d)
        min_pos = np.min(positions)
        positions = positions - min_pos
        
        super().__init__(
            name=f"Array Z5 (N={N})", 
            array_type="Weight-Constrained Sparse Array (Z5)", 
            sensor_positions=positions.tolist()
        )
        
        self.data.num_sensors = N
        self.data.aperture = positions[-1] - positions[0]


    # ------------------------------------------------------------------
    # ABSTRACT METHOD IMPLEMENTATIONS (Focus on Z5-specific segment analysis)
    # ------------------------------------------------------------------

    def compute_array_spacing(self):
        self.data.sensor_spacing = self.M

    def compute_all_differences(self):
        positions = self.data.sensors_positions
        N = self.data.num_sensors
        
        n_i, n_j = np.meshgrid(positions, positions)
        diff_flat = (n_i - n_j).flatten()
        ni_flat = n_i.flatten()
        nj_flat = n_j.flatten()
        
        self.data.all_differences_with_duplicates = diff_flat
        self.data.total_diff_computations = N * N
        self.data.all_differences_table = pd.DataFrame({
            'n_i': ni_flat, 'n_j': nj_flat, 'Difference': diff_flat
        })

    def analyze_coarray(self):
        """Processes differences. ULA segment starts at L1=3."""
        all_diffs = self.data.all_differences_with_duplicates
        unique_diffs = np.unique(all_diffs)
        
        self.data.unique_differences = unique_diffs
        self.data.num_unique_positions = len(unique_diffs)
        self.data.physical_positions = self.data.sensors_positions
        self.data.coarray_positions = unique_diffs
        
        self.data.virtual_only_positions = np.setdiff1d(
            self.data.coarray_positions, 
            self.data.physical_positions
        )

        min_pos = self.data.coarray_positions[0]
        max_pos = self.data.coarray_positions[-1]
        ideal_range = np.arange(min_pos, max_pos + 1)
        
        holes = np.setdiff1d(ideal_range, self.data.coarray_positions)
        self.data.coarray_holes = holes
        
        self.data.segmentation = [unique_diffs] 
        self.data.num_segmentation = 1

    def plot_coarray(self):
        """3.12. Placeholder for visualization."""
        print("\n[Plotting Placeholder]: Array Z5 Visualization (w(1)=w(2)=0)")
        print(f"Physical Array Positions: {self.data.sensors_positions}")
        print(f"Coarray Positions (Positive): {self.data.coarray_positions[self.data.coarray_positions >= 0]}")
        print(f"Holes (Positive Lags): {self.data.coarray_holes[self.data.coarray_holes >= 0]}")
        print("Note: The ULA segment starts from L1=3.")
        print("-" * 50)
        
    def compute_weight_distribution(self):
        all_diffs = self.data.all_differences_with_duplicates
        lags, counts = np.unique(all_diffs, return_counts=True)
        weight_dict = dict(zip(lags, counts))
        weights = [weight_dict.get(lag, 0) for lag in self.data.unique_differences]
        self.data.coarray_weight_distribution = np.array(weights)
        self.data.weight_table = pd.DataFrame({
            'Lag (Difference)': lags, 'Count (Weight)': counts
        })

    def analyze_contiguous_segments(self):
        """5.1-5.7. Identifies the contiguous segment. ULA segment starts at L1=3."""
        # ULA segment starts at L1=3 and ends at L2=4N-13. Length L=4N-15.
        L1 = 3
        L2 = 4 * self.N_total - 13
        L = L2 - L1 + 1
        
        if L < 1:
             L = 0 
             segment = np.array([])
        else:
             segment = np.arange(L1, L2 + 1)

        self.data.all_contiguous_segments = [segment]
        self.data.largest_contiguous_segment = segment 
        self.data.shortest_contiguous_segment = segment
        self.data.segment_lengths = [L]
        
        # Dm = floor(L/2)
        self.data.max_detectable_sources = np.floor(L / 2.0).astype(int) 
        self.data.segment_ranges = [(L1, L2)]

    def analyze_holes(self):
        self.data.missing_virtual_positions = self.data.coarray_holes
        self.data.num_holes = len(self.data.coarray_holes)

    def generate_performance_summary(self):
        data = self.data
        weights_dict = dict(zip(data.unique_differences, data.coarray_weight_distribution))
        
        aperture = data.coarray_positions[-1] - data.coarray_positions[0] if data.num_unique_positions > 0 else 0
        
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
            self.data.segment_lengths[0] if self.data.segment_lengths else 0,
            data.max_detectable_sources,
            data.num_holes,
            w1,
            w2,
            w3
        ]

        self.data.performance_summary_table = pd.DataFrame({'Metrics': metrics, 'Value': values})