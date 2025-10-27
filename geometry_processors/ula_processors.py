# geometry_processors/ula_processors.py

import numpy as np
import pandas as pd
from .bases_classes import BaseArrayProcessor

class ULArrayProcessor(BaseArrayProcessor):
    """
    Uniform Linear Array (ULA) processor (refactored).

    Conventions:
      - sensor positions are stored in **grid units** (integers): [0, 1, 2, ..., N_total-1]
      - physical spacing is `self.d` (float/int). Physical positions = grid_positions * d.
      - all difference/coarray computations are done on the integer lag grid
        so formulas like w(1), segment L1/L2, and K_max are exact and d-independent.
    """

    def __init__(self, N: int, d: float = 1):
        """
        Parameters
        ----------
        N : int
            Number of physical sensors (>= 2).
        d : float
            Physical inter-sensor spacing (default 1). Does not affect lag math.
        """
        if N < 2:
            raise ValueError("ULA requires N >= 2 sensors.")

        # Grid positions (integers). Physical positions are grid*d if needed for plotting.
        positions_grid = np.arange(N, dtype=int)

        super().__init__(
            name=f"ULA_N{N}",
            array_type="Uniform Linear Array",
            sensor_positions=positions_grid.tolist()
        )

        self.N_total = N
        self.d = d

    # ------------------------------------------------------------------
    # 2) Physical Array Specification
    # ------------------------------------------------------------------
    def compute_array_spacing(self):
        """Sets the (physical) sensor spacing."""
        self.data.sensor_spacing = self.d

    # ------------------------------------------------------------------
    # 3) Difference Coarray Computation  (on integer lag grid)
    # ------------------------------------------------------------------
    def compute_all_differences(self):
        """
        Computes all N^2 difference elements and the associated table (on grid).
        Since positions are in grid units, the differences are integer lags.
        """
        pos_grid = np.asarray(self.data.sensors_positions, dtype=int)
        N = self.N_total

        # Create all pairs (n_i, n_j) on the grid
        n_i, n_j = np.meshgrid(pos_grid, pos_grid, indexing="ij")

        # Integer lag grid (differences)
        diffs = n_i - n_j

        # Flatten for tables
        diff_flat = diffs.ravel()
        ni_flat = n_i.ravel()
        nj_flat = n_j.ravel()

        # 3.1. all_differences_table (grid units)
        self.data.all_differences_table = pd.DataFrame({
            "S_i": np.repeat(np.arange(1, N + 1), N),
            "S_j": np.tile(np.arange(1, N + 1), N),
            "n_i(grid)": ni_flat,
            "n_j(grid)": nj_flat,
            "Lag (n_i - n_j)": diff_flat
        })

        # 3.2. all_differences_with_duplicates
        self.data.all_differences_with_duplicates = diff_flat

        # 3.3. total_diff_computations
        self.data.total_diff_computations = int(N * N)

    def analyze_coarray(self):
        """
        3.4–3.11. Build coarray sets on the integer lag grid.
        For a ULA, the coarray is contiguous from -(N-1) to (N-1).
        """
        all_diffs = np.asarray(self.data.all_differences_with_duplicates, dtype=int)

        # 3.4. unique_differences (sorted integer lags)
        unique_diffs = np.unique(all_diffs)
        self.data.unique_differences = unique_diffs

        # 3.5. num_unique_positions
        self.data.num_unique_positions = int(unique_diffs.size)

        # 3.6. physical_positions (grid)
        self.data.physical_positions = np.asarray(self.data.sensors_positions, dtype=int)

        # 3.8. coarray_positions = unique lags
        self.data.coarray_positions = unique_diffs

        # 3.7. virtual_only_positions = coarray \ physical (on lag grid)
        self.data.virtual_only_positions = np.setdiff1d(
            self.data.coarray_positions,
            self.data.physical_positions
        )

        # 3.9–3.11. Holes & segmentation
        # ULA coarray is contiguous on the *two-sided* grid; no holes.
        self.data.coarray_holes = np.array([], dtype=int)
        self.data.segmentation = [unique_diffs]            # one contiguous block (two-sided)
        self.data.num_segmentation = 1

    def plot_coarray(self):
        """
        Placeholder plot. Keep side-effect-free for batch runs.
        Return None to stay compatible with your current runners.
        (Optional) You can implement a matplotlib figure here.
        """
        # No-op (your graphical_demo handles plotting comprehensively)
        return None

    # ------------------------------------------------------------------
    # 4) Weight Distribution
    # ------------------------------------------------------------------
    def compute_weight_distribution(self):
        """
        Count the occurrences of each integer lag (w(k)).
        For ULA, the expected closed form is w(k) = N_total - |k| for k in [-(N-1),...,N-1].
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

    # ------------------------------------------------------------------
    # 5) Contiguous Segment Analysis (use one-sided lags for DOF)
    # ------------------------------------------------------------------
    def analyze_contiguous_segments(self):
        """
        Define the MUSIC-relevant one-sided contiguous segment on non-negative lags.
        For ULA: non-negative lags are {0, 1, ..., N_total-1}, length L = N_total.
        Then max identifiable sources K_max = floor(L/2).
        """
        lags = np.asarray(self.data.unique_differences, dtype=int)
        lags_pos = lags[lags >= 0]
        lags_pos.sort()

        # Identify contiguous runs on non-negative lags (should be a single run for ULA)
        if lags_pos.size > 0:
            # Check contiguity
            dif = np.diff(lags_pos)
            breaks = np.where(dif != 1)[0]
            starts = np.insert(breaks + 1, 0, 0)
            ends = np.append(breaks, lags_pos.size - 1)

            segments = [lags_pos[s:e + 1] for s, e in zip(starts, ends)]
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

    # ------------------------------------------------------------------
    # 6) Holes Analysis
    # ------------------------------------------------------------------
    def analyze_holes(self):
        """
        On the one-sided lag grid, ULA has no holes in {0..N_total-1}.
        """
        lags = np.asarray(self.data.unique_differences, dtype=int)
        lags_pos = lags[lags >= 0]
        ideal = np.arange(0, self.N_total, dtype=int)
        holes = np.setdiff1d(ideal, lags_pos, assume_unique=False)

        self.data.missing_virtual_positions = holes
        self.data.num_holes = int(holes.size)

    # ------------------------------------------------------------------
    # 7) Performance Summary
    # ------------------------------------------------------------------
    def generate_performance_summary(self):
        """
        Summarize key geometry/coarray metrics.
        Aperture is reported on the lag grid (integer units). Multiply by d if physical units are needed.
        """
        data = self.data

        # Build a dict for quick w(k) lookups
        wd = dict(zip(data.weight_table["Lag"].astype(int), data.weight_table["Weight"].astype(int)))

        # Aperture on lag grid (two-sided range)
        aperture_lag = int(data.coarray_positions[-1] - data.coarray_positions[0]) if data.num_unique_positions else 0

        # One-sided largest segment length (L) already computed
        L = int(len(data.largest_contiguous_segment)) if data.largest_contiguous_segment is not None else 0

        metrics = [
            "Physical Sensors (N)",
            "Virtual Elements (Unique Lags)",
            "Virtual-only Elements",
            "Coarray Aperture (lags)",          # two-sided span in lag units
            "Contiguous Segment Length (L)",    # one-sided (non-negative)
            "Maximum Detectable Sources (K_max)",
            "Holes (one-sided)",
            "Weight at Lag 0 (w(0))",
            "Weight at Lag 1 (w(1))",
            "Weight at Lag 2 (w(2))"
        ]

        values = [
            self.N_total,
            data.num_unique_positions,
            int(len(data.virtual_only_positions)) if data.virtual_only_positions is not None else 0,
            aperture_lag,
            L,
            int(data.max_detectable_sources) if data.max_detectable_sources is not None else 0,
            int(data.num_holes) if data.num_holes is not None else 0,
            wd.get(0, 0),
            wd.get(1, 0),
            wd.get(2, 0),
        ]

        self.data.performance_summary_table = pd.DataFrame({"Metrics": metrics, "Value": values})
