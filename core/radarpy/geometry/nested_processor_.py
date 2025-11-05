import numpy as np
import pandas as pd
from .bases_classes import BaseArrayProcessor

class NestedArrayProcessor(BaseArrayProcessor):
    """
    Standard two-level Nested Array (NA), per Pal & Vaidyanathan (2010):

      P1 (dense):  {0, 1, ..., N1-1}         (grid units)
      P2 (sparse): {(N1+1), 2(N1+1), ..., N2(N1+1)}

    We store positions in **grid units**; physical spacing is `self.d`.
    All coarray math is done on the integer lag grid.
    """

    def __init__(self, N1: int, N2: int, d: float = 1.0):
        if N1 < 1 or N2 < 1:
            raise ValueError("Nested array requires N1 >= 1 and N2 >= 1.")

        self.N1 = int(N1)
        self.N2 = int(N2)
        self.d = float(d)

        # Grid-unit positions
        P1 = np.arange(self.N1, dtype=int)
        P2 = (np.arange(1, self.N2 + 1, dtype=int)) * (self.N1 + 1)
        positions_grid = np.unique(np.concatenate([P1, P2]))
        positions_grid.sort()

        super().__init__(
            name=f"Nested Array N={positions_grid.size}",
            array_type="Nested Array",
            sensor_positions=positions_grid.tolist()
        )

        # Actual number of sensors (accounts for any accidental overlap if d not 1)
        self.N_total = int(positions_grid.size)

    # ---------------- 2) Physical specification ----------------
    def compute_array_spacing(self):
        self.data.sensor_spacing = self.d

    # ---------------- 3) Difference coarray (integer lag grid) ----------------
    def compute_all_differences(self):
        pos = np.asarray(self.data.sensors_positions, dtype=int)
        N = self.N_total

        n_i, n_j = np.meshgrid(pos, pos, indexing="ij")
        diffs = (n_i - n_j).ravel()

        self.data.all_differences_with_duplicates = diffs
        self.data.total_diff_computations = int(N * N)
        self.data.all_differences_table = pd.DataFrame({
            "n_i(grid)": n_i.ravel(),
            "n_j(grid)": n_j.ravel(),
            "Lag (n_i - n_j)": diffs
        })

    def analyze_coarray(self):
        lags = np.asarray(self.data.all_differences_with_duplicates, dtype=int)
        uniq = np.unique(lags)

        self.data.unique_differences = uniq
        self.data.num_unique_positions = int(uniq.size)
        self.data.physical_positions = np.asarray(self.data.sensors_positions, dtype=int)
        self.data.coarray_positions = uniq

        self.data.virtual_only_positions = np.setdiff1d(
            self.data.coarray_positions, self.data.physical_positions
        )

        # two-sided set is often (nearly) contiguous; we'll compute holes later on one-sided
        self.data.coarray_holes = np.array([], dtype=int)
        self.data.segmentation = [uniq]
        self.data.num_segmentation = 1

    def plot_coarray(self):
        # No-op here; your graphical script handles plotting if needed
        return None

    # ---------------- 4) Weights ----------------
    def compute_weight_distribution(self):
        lags = np.asarray(self.data.all_differences_with_duplicates, dtype=int)
        u, c = np.unique(lags, return_counts=True)
        weight_dict = dict(zip(u.tolist(), c.tolist()))
        weights = [weight_dict.get(int(k), 0) for k in self.data.unique_differences]

        self.data.coarray_weight_distribution = np.asarray(weights, dtype=int)
        self.data.weight_table = pd.DataFrame({
            "Lag": u.astype(int),
            "Weight": c.astype(int)
        })

    # ---------------- 5) One-sided contiguous segment & DOF ----------------
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

    # ---------------- 6) Holes (on one-sided grid) ----------------
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

    # ---------------- 7) Summary ----------------
    def generate_performance_summary(self):
        data = self.data

        # Build weight lookup safely
        wd = {}
        if data.weight_table is not None and "Lag" in data.weight_table and "Weight" in data.weight_table:
            wd = dict(zip(data.weight_table["Lag"].astype(int), data.weight_table["Weight"].astype(int)))

        # Two-sided lags and one-sided (>=0)
        lags = np.asarray(data.coarray_positions, dtype=int) if data.coarray_positions is not None else np.array([], dtype=int)
        lags_pos = lags[lags >= 0]

        aperture_two_sided = int(lags.max() - lags.min()) if lags.size else 0
        max_positive_lag   = int(lags_pos.max()) if lags_pos.size else 0

        segment = np.asarray(data.largest_contiguous_segment, dtype=int) if data.largest_contiguous_segment is not None else np.array([], dtype=int)
        L = int(len(segment))
        seg_range_str = f"[{int(segment[0])}:{int(segment[-1])}]" if L > 0 else "[]"

        num_holes = int(data.num_holes) if data.num_holes is not None else 0

        metrics = [
            "Physical Sensors (N)",
            "Virtual Elements (Unique Lags)",
            "Virtual-only Elements",
            "Coarray Aperture (two-sided span, lags)",
            "Max Positive Lag (one-sided)",
            "Contiguous Segment Length (L, one-sided)",
            "Maximum Detectable Sources (K_max)",
            "Holes (one-sided)",
            "Largest One-sided Segment Range [L1:L2]",
            "Weight at Lag 0 (w(0))",
            "Weight at Lag 1 (w(1))",
            "Weight at Lag 2 (w(2))",
            "Weight at Lag 3 (w(3))",
        ]

        values = [
            getattr(self, "N_total", data.num_sensors),                 # robust across processors
            int(data.num_unique_positions) if data.num_unique_positions is not None else 0,
            int(len(data.virtual_only_positions)) if data.virtual_only_positions is not None else 0,
            aperture_two_sided,
            max_positive_lag,
            L,
            int(data.max_detectable_sources) if data.max_detectable_sources is not None else 0,
            num_holes,
            seg_range_str,
            int(wd.get(0, 0)),
            int(wd.get(1, 0)),
            int(wd.get(2, 0)),
            int(wd.get(3, 0)),
        ]

        data.performance_summary_table = pd.DataFrame({"Metrics": metrics, "Value": values})
