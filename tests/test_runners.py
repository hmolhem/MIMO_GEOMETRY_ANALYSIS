import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path so tests can import the package in-place
REPO_ROOT = Path(__file__).parent.parent
SRC_PATH = str(REPO_ROOT / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from mimo_geom_analysis.runners import run_benchmarks


def test_run_benchmarks_smoke(tmp_path):
    out = str(tmp_path / "bench_smoke.csv")
    # Minimal arguments: run a single-trial ULA SpatialMUSIC job
    argv = [
        "--arrays", "ULA",
        "--N", "4",
        "--d", "1.0",
        "--algs", "SpatialMUSIC",
        "--snr", "0",
        "--snapshots", "1",
        "--k", "1",
        "--trials", "1",
        "--out", out,
    ]

    saved = run_benchmarks(argv)
    assert saved == out
    assert Path(out).exists()
    # CSV should contain header and at least one row
    with open(out, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if l.strip()]
    assert len(lines) >= 2
