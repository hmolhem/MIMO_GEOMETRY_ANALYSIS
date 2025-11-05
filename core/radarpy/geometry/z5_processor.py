# geometry_processors/z5_processor.py
from __future__ import annotations
import numpy as np
import pandas as pd
from geometry_processors.bases_classes import BaseArrayProcessor

class Z5ArrayProcessor(BaseArrayProcessor):
    """
    Z5 processor:
      • Enforce w(1)=0 and w(2)=0 (no unit/2-unit differences)
      • Break mod-3 symmetry to enlarge the one-sided contiguous segment
      • Construction: alternate gaps [3,4,3,4,...] (keeps w(1)=w(2)=0 but adds diversity)
      
    Note: Reported A in "holes (< A_obs)" refers to observed max positive lag,
    not a closed-form expression like 3N-7 (which applies to Z4 arrays).
    """

    def __init__(self, N: int, d: float = 1.0):
        self.N_total = int(N)
        self.d = float(d)

        sensors_grid = self._build_z5_positions(self.N_total)

        super().__init__(
            name=f"Array Z5 (N={self.N_total})",
            array_type="Weight-Constrained Sparse Array (Z5)",
            sensor_positions=sensors_grid.tolist(),
            d=self.d,
        )

        # Pre-create fields used later
        self.data.coarray_positions = None
        self.data.largest_contiguous_segment = None
        self.data.missing_virtual_positions = None
        self.data.weight_table = pd.DataFrame(columns=["Lag", "Weight"])

    def __repr__(self) -> str:
        # Required by BaseArrayProcessor (abstract)
        return f"Z5ArrayProcessor(N={self.N_total}, d={self.d})"

    # ---------- construction ----------
    def _build_z5_positions(self, N: int) -> np.ndarray:
        if N <= 1:
            return np.zeros((N,), dtype=int)
        # Hand-tuned seed that keeps w(1)=w(2)=0 but fattens the coarray near small lags
        base_gaps = [3, 4, 3, 5, 3, 4]  # good for N=7
        if N <= len(base_gaps) + 1:
            gaps = base_gaps[:N-1]
        else:
            # For N>7, extend by repeating [3,4,3,5] blocks; still avoids {1,2}
            ext = []
            while len(ext) < N - 1:
                ext += [3, 4, 3, 5]
            gaps = ext[:N-1]

        pos = [0]
        for g in gaps:
            pos.append(pos[-1] + g)
        pos = np.asarray(pos, dtype=int)
        
        # Apply local perturbation to improve L while keeping w(1)=w(2)=0
        pos = self._improve_once_by_perturb(pos)
        return pos

    def _preserves_constraints(self, grid: np.ndarray) -> bool:
        diff = np.abs(grid[:, None] - grid[None, :]).ravel()
        return not (np.any(diff == 1) or np.any(diff == 2))

    def _largest_contig_segment_nonneg(self, lags_nonneg: np.ndarray) -> np.ndarray:
        if lags_nonneg.size == 0:
            return np.array([], dtype=int)
        lags = np.unique(lags_nonneg)
        best = (0, 0)  # (start_idx, length)
        start = 0
        for i in range(1, lags.size + 1):
            if i == lags.size or lags[i] != lags[i-1] + 1:
                length = i - start
                if length > best[1]:
                    best = (start, length)
                start = i
        s, m = best
        return lags[s:s+m]

    def _score_L_and_holes(self, grid: np.ndarray):
        diff = grid[:, None] - grid[None, :]
        lags = np.unique(diff.ravel().astype(int))
        one = lags[lags >= 0]
        seg = self._largest_contig_segment_nonneg(one)
        A_obs = int(one.max()) if one.size else 0
        full = np.arange(0, A_obs + 1, dtype=int)
        holes = np.setdiff1d(full, one)
        return int(seg.size), holes

    def _improve_once_by_perturb(self, pos: np.ndarray) -> np.ndarray:
        best = pos.copy()
        best_L, best_holes = self._score_L_and_holes(best)
        for i in range(1, len(pos) - 1):  # internal sensors only
            for delta in (-1, 1):
                cand = best.copy()
                cand[i] += delta
                cand.sort()
                if np.any(np.diff(cand) <= 0):
                    continue
                if not self._preserves_constraints(cand):
                    continue
                L, holes = self._score_L_and_holes(cand)
                if (L > best_L) or (L == best_L and len(holes) < len(best_holes)):
                    best, best_L, best_holes = cand, L, holes
        return best

    # ---------- analysis pipeline ----------
    def analyze_geometry(self):
        # All geometry derives from sensor positions; nothing extra here.
        return

    def analyze_coarray(self):
        grid = np.asarray(self.data.sensors_positions, dtype=int)

        # pairwise differences (lag units)
        diff = grid[:, None] - grid[None, :]
        lags_2s = np.unique(diff.ravel())
        self.data.coarray_positions = lags_2s

        # weight table
        uniq, counts = np.unique(diff.ravel(), return_counts=True)
        self.data.weight_table = pd.DataFrame({"Lag": uniq.astype(int), "Weight": counts.astype(int)})

        # one-sided set for segment/holes
        one = np.unique(lags_2s[lags_2s >= 0])
        seg = self._largest_contiguous_segment(one)
        self.data.largest_contiguous_segment = seg

        # Holes relative to observed [0..A_obs]
        A_obs = int(one.max()) if one.size else 0
        full_one = np.arange(0, A_obs + 1, dtype=int)
        holes_one = np.setdiff1d(full_one, one)
        self.data.missing_virtual_positions = holes_one

    def _largest_contiguous_segment(self, nonneg_lags: np.ndarray) -> np.ndarray:
        if nonneg_lags.size == 0:
            return np.array([], dtype=int)
        best_start = best_len = 0
        start = 0
        for i in range(1, nonneg_lags.size + 1):
            if i == nonneg_lags.size or nonneg_lags[i] != nonneg_lags[i - 1] + 1:
                seg_len = i - start
                if seg_len > best_len:
                    best_len = seg_len
                    best_start = start
                start = i
        return nonneg_lags[best_start:best_start + best_len]

    def compute_weight_distribution(self):
        # Already computed in analyze_coarray; keep for API symmetry.
        return

    def generate_performance_summary(self):
        lags = np.asarray(self.data.coarray_positions if self.data.coarray_positions is not None else [], dtype=int)
        one = lags[lags >= 0] if lags.size else np.array([], dtype=int)
        A_obs = int(one.max()) if one.size else 0

        seg = np.asarray(self.data.largest_contiguous_segment if self.data.largest_contiguous_segment is not None else [], dtype=int)
        if seg.size > 0:
            L1, L2 = int(seg[0]), int(seg[-1])
            L = int(seg.size)
        else:
            L1 = L2 = None
            L = 0

        wt_df = self.data.weight_table if isinstance(self.data.weight_table, pd.DataFrame) else pd.DataFrame(columns=["Lag", "Weight"])
        wt = {int(r["Lag"]): int(r["Weight"]) for _, r in wt_df.iterrows()} if not wt_df.empty else {}

        holes_one = np.asarray(self.data.missing_virtual_positions if self.data.missing_virtual_positions is not None else [], dtype=int)

        num_unique = int(np.unique(lags).size) if lags.size else 0
        num_virtual_only = num_unique - int(np.asarray(self.data.sensors_positions).size)
        aperture = int(lags.max() - lags.min()) if lags.size else 0
        k_max = L // 2

        rows = [
            ["Physical Sensors (N)",                  int(self.N_total)],
            ["Virtual Elements (Unique Lags)",       num_unique],
            ["Virtual-only Elements",                max(0, num_virtual_only)],
            ["Coarray Aperture (two-sided span, lags)", aperture],
            ["Max Positive Lag (one-sided)",         A_obs],
            ["Contiguous Segment Length (L, one-sided)", L],
            ["Maximum Detectable Sources (K_max)",   k_max],
            ["Holes (one-sided)",                    int(holes_one.size)],
            ["Weight at Lag 0 (w(0))",               wt.get(0, 0)],
            ["Weight at Lag 1 (w(1))",               wt.get(1, 0)],
            ["Weight at Lag 2 (w(2))",               wt.get(2, 0)],
            ["Weight at Lag 3 (w(3))",               wt.get(3, 0)],
            ["Largest One-sided Segment Range [L1:L2]", f"[{L1}:{L2}]" if L else "NA"],
        ]
        self.data.performance_summary_table = pd.DataFrame(rows, columns=["Metrics", "Value"])
