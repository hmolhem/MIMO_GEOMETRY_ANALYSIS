import sys
import os
import pandas as pd

# Ensure project root is importable (for local test runs)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from geometry_processors.ula_processors import ULArrayProcessor
from geometry_processors.nested_processor import NestedArrayProcessor
from geometry_processors.z1_processor import Z1ArrayProcessor
from geometry_processors.z3_1_processor import Z3_1ArrayProcessor
from geometry_processors.z3_2_processor import Z3_2ArrayProcessor
from geometry_processors.z4_processor import Z4ArrayProcessor
from geometry_processors.z5_processor import Z5ArrayProcessor


def run_basic_pipeline(proc):
    """Run the canonical analysis pipeline methods for a processor instance."""
    # Common pipeline steps (should not raise)
    proc.compute_array_spacing()
    proc.compute_all_differences()
    proc.analyze_coarray()
    proc.compute_weight_distribution()
    proc.analyze_contiguous_segments()
    proc.analyze_holes()
    proc.generate_performance_summary()


def test_geometry_processors_smoke():
    """Smoke test: instantiate each canonical geometry processor and run pipeline."""
    procs = []
    procs.append(ULArrayProcessor(7))
    # Nested uses N1, N2
    procs.append(NestedArrayProcessor(3, 4))
    procs.append(Z1ArrayProcessor(7))
    procs.append(Z3_1ArrayProcessor(7))
    procs.append(Z3_2ArrayProcessor(7))
    procs.append(Z4ArrayProcessor(7))
    procs.append(Z5ArrayProcessor(7))

    for p in procs:
        run_basic_pipeline(p)
        # Basic assertions: performance_summary_table exists and is a DataFrame
        assert hasattr(p.data, 'performance_summary_table'), f"{p} missing performance_summary_table"
        tbl = p.data.performance_summary_table
        assert isinstance(tbl, pd.DataFrame), f"{p}: performance_summary_table not a DataFrame"
        # At least one metric row expected
        assert tbl.shape[0] >= 1
