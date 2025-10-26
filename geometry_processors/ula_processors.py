import numpy as np
import pandas as pd
from .bases_classes import BaseArrayProcessor

class ULArrayProcessor(BaseArrayProcessor):
    """
    Concrete implementation for a Uniform Linear Array (ULA).
    Inherits from BaseArrayProcessor and implements all abstract methods.
    """
    def __init__(self, M: int, d: int = 1.0):
        # Generate ULA positions: [0, d, 2d, ..., (M-1)d]
        positions = np.arange(M) * d
        super().__init__(
            name="ULA_M{}".format(M),
            array_type="Uniform Linear Array",
            sensor_positions=positions.tolist() # Store as list in ArraySpec
        )
        self.d = d # Unit spacing
        self.M = M # Number of sensors

    # ------------------------------------------------------------------
    # 2. Physical Array Specification
    # ------------------------------------------------------------------
    def compute_array_spacing(self):
        """Sets the sensor spacing."""
        self.data.sensor_spacing = self.d

    # ------------------------------------------------------------------
    # 3. Difference Coarray Computation
    # ------------------------------------------------------------------
    def compute_all_differences(self):
        """Computes all N^2 difference elements and the associated table."""
        positions = np.array(self.data.sensors_positions)
        M = self.M

        # Create all pairs (n_i, n_j)
        n_i, n_j = np.meshgrid(positions, positions)
        
        # Compute differences (n_i - n_j)
        differences = n_i - n_j

        # Flatten arrays for table generation
        diff_flat = differences.flatten()
        ni_flat = n_i.flatten()
        nj_flat = n_j.flatten()
        
        # Generate S_i and S_j indices (1-based for S_i, S_j columns)
        s_i = np.repeat(np.arange(1, M + 1), M)
        s_j = np.tile(np.arange(1, M + 1), M)

        # 3.1. all_differences_table
        table_data = {
            'S_i': s_i,
            'S_j': s_j,
            'n_i': ni_flat,
            'n_j': nj_flat,
            'Difference (n_i - n_j)': diff_flat
        }
        self.data.all_differences_table = pd.DataFrame(table_data)

        # 3.2. all_differences_with_duplicates
        self.data.all_differences_with_duplicates = diff_flat

        # 3.3. total_diff_computations
        self.data.total_diff_computations = M * M

    def analyze_coarray(self):
        """Processes the differences to find unique elements, physical, and virtual-only positions."""
        all_diffs = self.data.all_differences_with_duplicates

        # 3.4. unique_differences (Coarray positions)
        unique_diffs = np.unique(all_diffs)
        self.data.unique_differences = unique_diffs

        # 3.5. num_unique_positions
        self.data.num_unique_positions = len(unique_diffs)

        # 3.6. physical_positions
        self.data.physical_positions = np.array(self.data.sensors_positions)

        # 3.8. coarray_positions (Same as unique_differences for ULA)
        self.data.coarray_positions = unique_diffs

        # 3.7. virtual_only_positions
        # Set difference: Coarray - Physical
        self.data.virtual_only_positions = np.setdiff1d(
            self.data.coarray_positions, 
            self.data.physical_positions
        )

        # ULA is always contiguous, so holes and segmentation are simple here
        self.data.coarray_holes = np.array([]) # 3.9.
        self.data.segmentation = [unique_diffs] # 3.10.
        self.data.num_segmentation = 1 # 3.11.

    def plot_coarray(self):
        """
        3.12. Placeholder for plotting the physical, virtual-only, 
        and coarray positions with labels.
        """
        # In a real implementation, you would use Matplotlib here.
        
        print("\n[Plotting Placeholder]: Coarray Visualization")
        # Example representation:
        coarray = self.data.coarray_positions
        physical = set(self.data.physical_positions)
        
        vis_str = "Position: "
        label_str = "Label:    "
        
        for pos in coarray:
            vis_str += f"{pos:^4}"
            if pos in physical:
                label_str += " P  "
            else:
                label_str += " V  "
        
        print(vis_str)
        print(label_str)
        print("Legend: P = Physical Sensor, V = Virtual-Only Position")
        print("-" * 50)


    # ------------------------------------------------------------------
    # 4. Weight Distribution
    # ------------------------------------------------------------------
    def compute_weight_distribution(self):
        """Calculates the weight (count) of each unique lag."""
        all_diffs = self.data.all_differences_with_duplicates
        unique_diffs = self.data.unique_differences
        M = self.M
        
        # Count the frequency of each difference (weight)
        lags, counts = np.unique(all_diffs, return_counts=True)
        weight_dist = dict(zip(lags, counts))
        
        # 4.2. coarray_weight_distribution (sorted by lag)
        weights = [weight_dist.get(lag, 0) for lag in unique_diffs]
        self.data.coarray_weight_distribution = np.array(weights)
        
        # 4.1. weight_table (Simplified: without sensor pairs for brevity)
        table_data = {'Differences (Lag)': lags, 'Count (Weight)': counts}
        # Note: Sensor Pairs column requires looping through all_differences_table, 
        # which is complex. Simplifying to show essential data.
        self.data.weight_table = pd.DataFrame(table_data)

    # ------------------------------------------------------------------
    # 5. Contiguous Segment Analysis
    # ------------------------------------------------------------------
    def analyze_contiguous_segments(self):
        """Identifies the contiguous segments and max detectable sources."""
        coarray = self.data.coarray_positions
        
        # For ULA, the entire coarray is contiguous
        if len(coarray) > 0 and (coarray[-1] - coarray[0] + 1) == len(coarray):
            segment = coarray
            self.data.all_contiguous_segments = [segment] # 5.1.
            self.data.largest_contiguous_segment = segment # 5.2.
            self.data.shortest_contiguous_segment = segment # 5.3.
            segment_length = len(segment)
        else: # Should not happen for ULA, but good for robustness
            segment_length = 0 
            self.data.all_contiguous_segments = [] 
            self.data.largest_contiguous_segment = np.array([])
            self.data.shortest_contiguous_segment = np.array([])
        
        # 5.4. Length of segments
        self.data.segment_lengths = [len(s) for s in self.data.all_contiguous_segments]
        
        # 5.5. Maximum detectable sources
        self.data.max_detectable_sources = np.floor(segment_length / 2.0).astype(int)
        
        # 5.6/5.7. Range of each segments
        self.data.segment_ranges = [(s[0], s[-1]) for s in self.data.all_contiguous_segments]

    # ------------------------------------------------------------------
    # 6. Holes Analysis
    # ------------------------------------------------------------------
    def analyze_holes(self):
        """Analyzes the missing positions in the coarray."""
        coarray = self.data.coarray_positions
        
        if len(coarray) == 0:
            self.data.missing_virtual_positions = np.array([])
            self.data.num_holes = 0
            return

        min_pos = coarray[0]
        max_pos = coarray[-1]
        
        # Ideal contiguous range
        ideal_range = np.arange(min_pos, max_pos + 1)
        
        # 6.1. missing_virtual_positions (Holes)
        holes = np.setdiff1d(ideal_range, coarray)
        self.data.missing_virtual_positions = holes
        
        # 6.2. number of holes
        self.data.num_holes = len(holes)

    # ------------------------------------------------------------------
    # 7. Performance Summary
    # ------------------------------------------------------------------
    def generate_performance_summary(self):
        """Generates the final performance summary table."""
        data = self.data
        weights_dict = dict(zip(data.unique_differences, data.coarray_weight_distribution))
        
        # Max - Min positions for aperture
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
            self.M,
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