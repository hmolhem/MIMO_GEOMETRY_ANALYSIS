# geometry_processors/z4_processor.py
from __future__ import annotations
import numpy as np
import pandas as pd

from .bases_classes import BaseArrayProcessor  # your base class


class Z4ArrayProcessor(BaseArrayProcessor):
    """
    Z4 (w(1)=w(2)=0) weight-constrained sparse array.

    Conventions
    -----------
    * N_total (int): number of sensors
    * d (float): physical spacing multiplier for the integer grid
    * All coarray math is done on the INTEGER LAG GRID.
      Physical values are derived by multiplying by d (for display only).
    """

    def __init__(self, N: int = 7, d: float = 1.0):
        if N < 5:
            raise ValueError("Z4 requires N >= 5")

        self.N_total = int(N)
        self.d = float(d)

        # Canonical N=7 layout to match your logs:
        # [0, 5, 8, 11, 14, 17, 21]
        if self.N_total == 7:
            sensors_grid = np.array([0, 5, 8, 11, 14, 17, 21], dtype=int)
        else:
            # Generic construction (keeps w(1)=w(2)=0 for small N via spacing heuristics)
            core = [3 * k for k in range(5)]  # 0,3,6,9,12
            extra = []
            cursor = core[-1]
            bumps = [5, 3, 4, 3, 4, 5]  # cycle safely
            bi = 0
            needed = self.N_total - len(core)
            for _ in range(max(0, needed)):
                cursor += bumps[bi % len(bumps)]
                extra.append(cursor)
                bi += 1
            sensors_grid = np.array(core + extra, dtype=int)

        sensors_grid = np.unique(sensors_grid)
        if sensors_grid.size != self.N_total:
            raise RuntimeError("Z4 generator produced wrong sensor count; adjust heuristic.")

        # >>> FIX: pass d=self.d to BaseArrayProcessor
        super().__init__(
            name=f"Array Z4 (N={self.N_total})",
            array_type="Weight-Constrained Sparse Array (Z4)",
            sensor_positions=sensors_grid.tolist(),
            d=self.d,
        )

        # Fill metadata
        self.data.name = f"Array Z4 (N={self.N_total})"
        self.data.num_sensors = self.N_total
        self.data.sensor_spacing = self.d  # physical multiplier

    # ---------------- Required abstract methods ---------------- #

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(N={self.N_total}, d={self.d})"

    def compute_array_spacing(self):
        return self.d

    def compute_all_differences(self):
        # integer grid differences (two-sided)
        g = np.asarray(self.data.sensors_positions, dtype=int)
        diffs = []
        for i in range(g.size):
            for j in range(g.size):
                diffs.append(g[i] - g[j])
        self.data.all_differences_with_duplicates = np.array(diffs, dtype=int)
        self.data.total_diff_computations = int(len(diffs))

    def analyze_coarray(self):
        lags = np.asarray(self.data.all_differences_with_duplicates, dtype=int)
        uniq = np.unique(lags)  # two-sided unique integer lags

        self.data.unique_differences = uniq
        self.data.num_unique_positions = int(uniq.size)
        self.data.physical_positions = np.asarray(self.data.sensors_positions, dtype=int)
        self.data.coarray_positions = uniq  # two-sided

        self.data.virtual_only_positions = np.setdiff1d(
            self.data.coarray_positions, self.data.physical_positions
        )

        # aperture on integer grid
        self.data.aperture = int(uniq.max() - uniq.min())

    def compute_weight_distribution(self):
        lags = np.asarray(self.data.all_differences_with_duplicates, dtype=int)
        u, c = np.unique(lags, return_counts=True)
        self.data.weight_table = pd.DataFrame({"Lag": u, "Weight": c})

    def analyze_contiguous_segments(self):
        u = np.asarray(self.data.coarray_positions, dtype=int)
        nonneg = u[u >= 0]
        nonneg.sort()
        if nonneg.size == 0:
            self.data.largest_contiguous_segment = np.array([], dtype=int)
            self.data.segment_ranges = []
            self.data.max_detectable_sources = 0
            return

        best_start = nonneg[0]
        best_len = 1
        cur_start = nonneg[0]
        cur_len = 1
        for a, b in zip(nonneg[:-1], nonneg[1:]):
            if b == a + 1:
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_len = cur_len
                    best_start = cur_start
                cur_start = b
                cur_len = 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start

        segment = np.arange(best_start, best_start + best_len, dtype=int)
        self.data.largest_contiguous_segment = segment
        self.data.segment_ranges = [[int(segment[0]), int(segment[-1])]] if segment.size else []
        self.data.max_detectable_sources = int(best_len // 2)

    def analyze_holes(self):
        # Z4: canonical A = 3N - 7; ensure w(1)=w(2)=0
        u = np.asarray(self.data.coarray_positions, dtype=int)
        A = 3 * self.N_total - 7

        pos = u[u >= 0]
        pos_set = set(pos.tolist())
        one_sided_holes = sorted([x for x in range(0, A) if x not in pos_set])

        self.data.missing_virtual_positions = np.array(one_sided_holes, dtype=int)
        self.data.num_holes = int(len(one_sided_holes))

        full_span = np.arange(u.min(), u.max() + 1)
        holes_2s = sorted([x for x in full_span if x not in set(u.tolist())])
        self.data.holes_two_sided = np.array(holes_2s, dtype=int)

        wt = self._weight_dict()
        assert wt.get(1, 0) == 0, "Z4 requires w(1)=0"
        assert wt.get(2, 0) == 0, "Z4 requires w(2)=0"

    def generate_performance_summary(self):
        wt = self._weight_dict()
        u = np.asarray(self.data.coarray_positions, dtype=int)

        seg = getattr(self.data, "largest_contiguous_segment", None)
        if seg is None or (isinstance(seg, np.ndarray) and seg.size == 0):
            L = 0
            seg_range = "[]"
        else:
            L = int(len(seg))
            seg_range = f"[{int(seg[0])}:{int(seg[-1])}]"

        pos = u[u >= 0]
        max_pos = int(pos.max()) if pos.size else 0
        A = 3 * self.N_total - 7

        rows = [
            ("Physical Sensors (N)", self.N_total),
            ("Virtual Elements (Unique Lags)", int(u.size)),
            ("Virtual-only Elements", int(len(self.data.virtual_only_positions))),
            ("Coarray Aperture (two-sided span, lags)", int(u.max() - u.min())),
            ("Max Positive Lag (one-sided)", max_pos),
            ("Contiguous Segment Length (L, one-sided)", L),
            ("Maximum Detectable Sources (K_max)", int(L // 2)),
            ("Holes (one-sided)", int(self.data.num_holes)),
            ("Weight at Lag 0 (w(0))", int(wt.get(0, 0))),
            ("Weight at Lag 1 (w(1))", int(wt.get(1, 0))),
            ("Weight at Lag 2 (w(2))", int(wt.get(2, 0))),
            ("Weight at Lag 3 (w(3))", int(wt.get(3, 0))),
            ("Largest One-sided Segment Range [L1:L2]", seg_range),
            ("Canonical A (3N-7)", A),
        ]
        self.data.performance_summary_table = pd.DataFrame(rows, columns=["Metrics", "Value"])

    def plot_coarray(self):
        seg = getattr(self.data, "largest_contiguous_segment", np.array([], dtype=int))
        return {
            "title": "Array Z4 Visualization (w(1)=w(2)=0)",
            "segment": [int(seg[0]), int(seg[-1])] if seg.size else None,
            "L": int(len(seg)),
        }

    # ---------------- helpers ---------------- #

    def _weight_dict(self) -> dict[int, int]:
        if getattr(self.data, "weight_table", None) is None:
            return {}
        d = {}
        for _, row in self.data.weight_table.iterrows():
            d[int(row["Lag"])] = int(row["Weight"])
        return d
