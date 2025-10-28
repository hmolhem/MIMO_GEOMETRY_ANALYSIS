# geometry_processors/bases_classes.py
from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class ArraySpec:
    # Identity
    name: str
    array_type: str

    # Design / inputs
    sensors_positions: List[int]  # integer grid positions (NOT multiplied by d)
    num_sensors: int
    sensor_spacing: float  # d (physical spacing unit)

    # Core derived artifacts (all in integer lag units unless noted)
    all_differences_with_duplicates: Optional[np.ndarray] = None  # two-sided integer lags (with dupes)
    unique_differences: Optional[np.ndarray] = None               # sorted unique two-sided integer lags
    num_unique_positions: Optional[int] = None

    coarray_positions: Optional[np.ndarray] = None                # alias for unique_differences (two-sided)
    physical_positions: Optional[np.ndarray] = None               # sensor grid positions as ndarray (ints)
    virtual_only_positions: Optional[np.ndarray] = None           # setdiff1d(coarray, physical)
    coarray_holes: Optional[np.ndarray] = None

    # Segments and holes (one-sided analysis)
    largest_contiguous_segment: Optional[np.ndarray] = None       # non-negative lags contiguous window
    all_contiguous_segments: Optional[List[np.ndarray]] = None
    segment_lengths: Optional[List[int]] = None
    segment_ranges: Optional[List[List[int]]] = None              # [[L1, L2], ...]
    max_detectable_sources: Optional[int] = None

    num_holes: Optional[int] = None
    missing_virtual_positions: Optional[np.ndarray] = None            # all one-sided holes up to max(+)
    missing_virtual_positions_below_A: Optional[np.ndarray] = None    # canonical (< A) holes if applicable

    # Weighting
    weight_table: Optional[pd.DataFrame] = None  # columns: Lag, Weight
    weight_dict: Optional[Dict[int, int]] = None # {lag: count}

    # Misc / meta
    segmentation: Optional[List[np.ndarray]] = None
    num_segmentation: Optional[int] = None
    aperture: Optional[int] = None  # two-sided span (max lag - min lag), integer units

    # Presentation
    performance_summary_table: Optional[pd.DataFrame] = None


class BaseArrayProcessor(abc.ABC):
    """
    Base processor providing a standard analysis pipeline.

    Subclasses must implement:
      - compute_array_spacing (if custom)
      - compute_all_differences (must populate integer lag diffs in data)
      - analyze_coarray (may keep default)
      - compute_weight_distribution (default provided)
      - analyze_contiguous_segments (default provided)
      - analyze_holes (default provided)
      - generate_performance_summary (default provided; subclasses may override to add fields)
      - plot_coarray (optional visualization)
    """

    def __init__(self, name: str, array_type: str, sensor_positions: List[int], d: float):
        self.data = ArraySpec(
            name=name,
            array_type=array_type,
            sensors_positions=list(map(int, sensor_positions)),
            num_sensors=int(len(sensor_positions)),
            sensor_spacing=float(d),
        )

    # ---------- Pipeline ----------

    def run_full_analysis(self, verbose: bool = True) -> ArraySpec:
        if verbose:
            print(f"--- Starting analysis for {self.data.name} ({self.data.array_type}) ---")
        self.compute_array_spacing()
        self.compute_all_differences()
        self.analyze_coarray()
        self.compute_weight_distribution()
        self.analyze_contiguous_segments()
        self.analyze_holes()
        self.generate_performance_summary()
        # Optional plot from subclass
        self.plot_coarray()
        if verbose:
            print("--- Analysis Complete ---\n")
        return self.data

    # ---------- Steps (defaults may be overridden) ----------

    def compute_array_spacing(self):
        # Already stored, but keep method to align interface
        self.data.num_sensors = int(len(self.data.sensors_positions))
        # self.data.sensor_spacing is set in __init__
        return

    def compute_all_differences(self):
        """
        Build integer-lag differences from physical diffs by normalizing with d and rounding.
        """
        grid = np.asarray(self.data.sensors_positions, dtype=float)
        # physical locations = grid * d  (we only need grid differences divided by d => integer lags)
        # Use full difference set with duplicates (two-sided).
        diffs = []
        for i in range(grid.size):
            for j in range(grid.size):
                # physical difference divided by d equals (grid[j]-grid[i])
                diffs.append(grid[j] - grid[i])
        diffs = np.asarray(diffs, dtype=float)

        # Normalize to integer lag grid (divide by d and round)
        d = self.data.sensor_spacing if self.data.sensor_spacing != 0 else 1.0
        lags = np.rint(diffs / d).astype(int)

        self.data.all_differences_with_duplicates = lags

    def analyze_coarray(self):
        """
        Two-sided, unique coarray positions. Also derive physical positions ndarray.
        """
        lags = np.asarray(self.data.all_differences_with_duplicates, dtype=int)
        uniq = np.unique(lags)

        self.data.unique_differences = uniq
        self.data.num_unique_positions = int(uniq.size)
        self.data.coarray_positions = uniq

        self.data.physical_positions = np.asarray(self.data.sensors_positions, dtype=int)
        self.data.virtual_only_positions = np.setdiff1d(uniq, self.data.physical_positions)

        # Aperture (two-sided span)
        if uniq.size:
            self.data.aperture = int(uniq.max() - uniq.min())
        else:
            self.data.aperture = 0

    def compute_weight_distribution(self):
        """
        Create a lag->count table; store both DataFrame and dict.
        """
        lags = np.asarray(self.data.all_differences_with_duplicates, dtype=int)
        lvals, counts = np.unique(lags, return_counts=True)
        df = pd.DataFrame({"Lag": lvals, "Weight": counts})
        df.sort_values("Lag", inplace=True, kind="mergesort")
        df.reset_index(drop=True, inplace=True)
        self.data.weight_table = df
        self.data.weight_dict = dict(zip(lvals.tolist(), counts.tolist()))

    def analyze_contiguous_segments(self):
        """
        Generic: find largest contiguous non-negative segment from available coarray positions.
        """
        uniq = np.asarray(self.data.coarray_positions, dtype=int)
        nonneg = np.sort(uniq[uniq >= 0])

        segments: List[np.ndarray] = []
        if nonneg.size == 0:
            seg = np.array([], dtype=int)
            segments = [seg]
        else:
            # split on gaps greater than 1
            breaks = np.where(np.diff(nonneg) > 1)[0]
            start = 0
            for b in breaks:
                segments.append(nonneg[start:b+1])
                start = b + 1
            segments.append(nonneg[start:])

        self.data.all_contiguous_segments = segments
        self.data.segment_lengths = [int(len(s)) for s in segments]
        self.data.segment_ranges = [[int(s[0]), int(s[-1])] for s in segments if s.size]
        # largest by length (tie-break: earliest)
        if segments:
            seg = max(segments, key=lambda s: (len(s), -s[0] if s.size else 0))
        else:
            seg = np.array([], dtype=int)

        self.data.largest_contiguous_segment = seg
        self.data.max_detectable_sources = int(len(seg) // 2)
        # generic holes (one-sided up to max positive)
        if nonneg.size:
            ideal = np.arange(0, int(nonneg.max()) + 1, dtype=int)
            holes = np.setdiff1d(ideal, nonneg, assume_unique=False)
        else:
            holes = np.array([], dtype=int)
        self.data.missing_virtual_positions = holes
        self.data.num_holes = int(holes.size)

    def analyze_holes(self):
        """
        Default keeps 'missing_virtual_positions' computed in analyze_contiguous_segments.
        Subclasses may compute additional canonical (<A) variants.
        """
        self.data.missing_virtual_positions_below_A = None
        return

    def generate_performance_summary(self):
        """
        Generic summary table; subclasses may append custom rows.
        """
        d = self.data
        rows = [
            ("Physical Sensors (N)", d.num_sensors),
            ("Virtual Elements (Unique Lags)", d.num_unique_positions or 0),
            ("Virtual-only Elements", int(len(d.virtual_only_positions)) if d.virtual_only_positions is not None else 0),
            ("Coarray Aperture (two-sided span, lags)", d.aperture if d.aperture is not None else 0),
            ("Max Positive Lag (one-sided)", int(np.max(d.coarray_positions)) if d.coarray_positions is not None and d.coarray_positions.size else 0),
            ("Contiguous Segment Length (L, one-sided)", int(len(d.largest_contiguous_segment)) if d.largest_contiguous_segment is not None else 0),
            ("Maximum Detectable Sources (K_max)", d.max_detectable_sources if d.max_detectable_sources is not None else 0),
            ("Holes (one-sided)", int(d.num_holes) if d.num_holes is not None else 0),
        ]
        # Add a few weights if available
        if d.weight_dict:
            rows.extend([
                ("Weight at Lag 0 (w(0))", d.weight_dict.get(0, 0)),
                ("Weight at Lag 1 (w(1))", d.weight_dict.get(1, 0)),
                ("Weight at Lag 2 (w(2))", d.weight_dict.get(2, 0)),
                ("Weight at Lag 3 (w(3))", d.weight_dict.get(3, 0)),
            ])
        self.data.performance_summary_table = pd.DataFrame(rows, columns=["Metrics", "Value"])

    # ---------- Optional visualization ----------

    def plot_coarray(self):
        """
        Subclasses can implement a real plot if desired. Keep as a placeholder here.
        """
        return

    # ---------- Abstract marker for subclasses (if they require custom behavior) ----------

    @abc.abstractmethod
    def __repr__(self) -> str:  # not strictly necessary, but keeps subclasses explicit
        return f"<{self.__class__.__name__}: {self.data.name}>"
