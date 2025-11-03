# geometry_processors/z6_processor.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from .bases_classes import BaseArrayProcessor  # uses your existing base

@dataclass
class Z6Data:
    """Data container for Z6 array analysis results - extends base ArraySpec fields."""
    # Core identity fields (required by base class)
    name: str = ""
    array_type: str = ""
    sensors_positions: List[int] = field(default_factory=list)  # Note: 'sensors' with 's' to match base class
    num_sensors: int = 0
    sensor_spacing: float = 1.0  # d (physical spacing unit)
    
    # Additional Z6-specific fields
    sensor_positions_physical: List[float] = field(default_factory=list)
    sensor_positions_grid: Optional[np.ndarray] = None
    coarray_lags: List[int] = field(default_factory=list)
    coarray_positions: Optional[np.ndarray] = None
    coarray_one_sided: Optional[np.ndarray] = None
    weights_two_sided: Dict[int, int] = field(default_factory=dict)
    weights_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["Lag", "Weight"]))
    weight_table: Optional[pd.DataFrame] = None
    A_obs: int = 0
    largest_contiguous_segment: Optional[np.ndarray] = None
    contiguous_L: int = 0
    holes_one_sided: List[int] = field(default_factory=list)
    holes_two_sided: List[int] = field(default_factory=list)
    summary_rows: List[Tuple[str, Any]] = field(default_factory=list)
    weight_table_rows: List[Tuple[int, int]] = field(default_factory=list)
    
    # Additional fields from ArraySpec that base class may expect
    all_differences_with_duplicates: Optional[np.ndarray] = None
    unique_differences: Optional[np.ndarray] = None
    physical_positions: Optional[np.ndarray] = None
    missing_virtual_positions: Optional[np.ndarray] = None

class Z6ArrayProcessor(BaseArrayProcessor):
    """
    Z6: Weight-constrained sparse array (next design in the Z-family).

    Design goals (pragmatic):
      - Keep w(1)=0 and w(2)=0 (no unit/2-unit differences).
      - Grow the one-sided contiguous coarray segment starting near 3.
      - Avoid touching base classes or other processors.

    Geometry generator (heuristic, N>=5 recommended):
      - Start at 0
      - Use gap pattern that never equals 1 or 2 to keep w(1)=w(2)=0.
      - Pattern chosen to extend the contiguous region while pushing aperture.

      Default gap cycle: [4, 3, 3, 4, 4, 3, 4, 3] (repeats as needed)
      For N=7 => sensors (grid): [0, 4, 7, 10, 14, 18, 21]
      (You can tweak _build_geometry() later without touching runner.)
    """

    def __init__(self, N: int, d: float = 1.0) -> None:
        if N < 5:
            raise ValueError("Z6 expects N>=5.")
        self.N_total = int(N)
        self.d = float(d)

        sensors_grid = self._build_geometry(self.N_total)
        sensors_phys = (np.asarray(sensors_grid, dtype=float) * self.d).tolist()

        name = f"Array Z6 (N={self.N_total})"
        array_type = "Weight-Constrained Sparse Array (Z6)"

        super().__init__(
            name=name,
            array_type=array_type,
            sensor_positions=sensors_phys,
            d=self.d,
        )
        # Persist a grid (integer) version for coarray math
        self.sensors_grid = np.asarray(sensors_grid, dtype=int)

        # IMPORTANT: construct Z6Data with name/array_type set
        self.data = Z6Data(
            name=name,
            array_type=array_type,
            sensors_positions=sensors_grid if isinstance(sensors_grid, list) else sensors_grid.tolist(),
            num_sensors=len(sensors_grid),
            sensor_spacing=self.d,
            sensor_positions_physical=sensors_phys,
            sensor_positions_grid=self.sensors_grid,
            largest_contiguous_segment=np.array([], dtype=int),
        )

    # ---------- Pretty ----------
    def __repr__(self) -> str:
        return f"Z6ArrayProcessor(N={self.N_total}, d={self.d})"

    # ---------- Geometry ----------
    def _build_geometry(self, N: int) -> List[int]:
        """
        Produce an integer-grid geometry with w(1)=w(2)=0 by forbidding 1- and 2-sized gaps.
        Heuristic gap cycle is tunable; it tries to keep a decent L while expanding aperture.
        """
        gap_cycle = [4, 3, 3, 4, 4, 3, 4, 3]  # never 1 or 2
        pos = [0]
        gi = 0
        while len(pos) < N:
            pos.append(pos[-1] + gap_cycle[gi % len(gap_cycle)])
            gi += 1
        return pos

    # ---------- Core Analysis ----------
    def analyze_coarray(self) -> None:
        x = self.sensors_grid
        # All ordered pair differences (two-sided)
        diffs = (x.reshape(-1, 1) - x.reshape(1, -1)).ravel()
        uniq = np.unique(diffs.astype(int))
        self.data.coarray_positions = uniq

        # One-sided (>=0)
        pos = np.unique(uniq[uniq >= 0])
        self.data.coarray_one_sided = pos

        # Weights (difference multiplicity)
        # Count exact occurrences per lag
        _, counts = np.unique(diffs.astype(int), return_counts=True)
        wt_df = pd.DataFrame({"Lag": uniq, "Weight": counts}).sort_values("Lag", ascending=True).reset_index(drop=True)
        self.data.weights_df = wt_df

        # Largest one-sided contiguous run (integers, step=1)
        seg, L = self._largest_contiguous_run(pos)
        self.data.largest_contiguous_segment = seg
        self.data.contiguous_L = int(L)

        # Observed max positive lag
        A_obs = int(pos.max()) if pos.size else 0
        self.data.A_obs = A_obs

        # Holes: one-sided in [0..A_obs], two-sided in [-A_obs..A_obs]
        full_1s = np.arange(0, A_obs + 1, dtype=int) if A_obs >= 0 else np.array([], dtype=int)
        holes_1s = np.setdiff1d(full_1s, pos, assume_unique=False)
        self.data.holes_one_sided = holes_1s.tolist()

        full_2s = np.arange(-A_obs, A_obs + 1, dtype=int) if A_obs > 0 else np.array([], dtype=int)
        holes_2s = np.setdiff1d(full_2s, uniq, assume_unique=False)
        self.data.holes_two_sided = holes_2s.tolist()

    def _largest_contiguous_run(self, sorted_nonneg: np.ndarray) -> Tuple[np.ndarray, int]:
        if sorted_nonneg.size == 0:
            return np.array([], dtype=int), 0
        best_start = best_len = 0
        cur_start = 0
        cur_len = 1
        for i in range(1, sorted_nonneg.size):
            if sorted_nonneg[i] == sorted_nonneg[i - 1] + 1:
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
                cur_start, cur_len = i, 1
        if cur_len > best_len:
            best_len, best_start = cur_len, cur_start
        seg = sorted_nonneg[best_start: best_start + best_len]
        return seg, int(best_len)

    # ---------- Summary ----------
    def generate_performance_summary(self) -> Dict[str, Any]:
        wt_df = self.data.weights_df
        wt = {int(r["Lag"]): int(r["Weight"]) for _, r in wt_df.iterrows()} if not wt_df.empty else {}

        uniq = np.asarray(self.data.coarray_positions if self.data.coarray_positions is not None else [], dtype=int)
        pos = np.asarray(self.data.coarray_one_sided if self.data.coarray_one_sided is not None else [], dtype=int)

        A_obs = int(self.data.A_obs)
        L = int(self.data.contiguous_L)
        seg = np.asarray(self.data.largest_contiguous_segment, dtype=int) if L > 0 else np.array([], dtype=int)
        L1, L2 = (int(seg[0]), int(seg[-1])) if L > 0 else ("NA", "NA")

        physical_N = int(self.N_total)
        virtual_unique = int(uniq.size)
        virtual_only = int(max(0, virtual_unique - physical_N))
        aperture_two_sided_span = int(2 * A_obs)  # matches your prior outputs
        K_max = int(np.floor(L / 2))

        summary_rows = [
            ("Physical Sensors (N)", physical_N),
            ("Virtual Elements (Unique Lags)", virtual_unique),
            ("Virtual-only Elements", virtual_only),
            ("Coarray Aperture (two-sided span, lags)", aperture_two_sided_span),
            ("Max Positive Lag (one-sided)", A_obs),
            ("Contiguous Segment Length (L, one-sided)", L),
            ("Maximum Detectable Sources (K_max)", K_max),
            ("Holes (one-sided)", len(self.data.holes_one_sided)),
            ("Weight at Lag 0 (w(0))", wt.get(0, 0)),
            ("Weight at Lag 1 (w(1))", wt.get(1, 0)),
            ("Weight at Lag 2 (w(2))", wt.get(2, 0)),
            ("Weight at Lag 3 (w(3))", wt.get(3, 0)),
            ("Largest One-sided Segment Range [L1:L2]", f"[{L1}:{L2}]"),
        ]
        df = pd.DataFrame(summary_rows, columns=["Metrics", "Value"])
        self.data.summary_df = df
        return {"summary_df": df, "weights_df": wt_df}

    # ---------- Convenience ----------
    def get_two_sided_holes(self) -> List[int]:
        return list(self.data.holes_two_sided)

    def get_one_sided_holes(self) -> List[int]:
        return list(self.data.holes_one_sided)
