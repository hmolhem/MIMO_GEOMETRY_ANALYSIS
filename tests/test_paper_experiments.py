import sys
from pathlib import Path

# Ensure src is importable
REPO = Path(__file__).parent.parent
SRC = str(REPO / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from mimo_geom_analysis.paper_experiments import run_paper_experiments


def test_run_paper_experiments_smoke(tmp_path):
    outdir = str(tmp_path / "paper_out")
    argv = ["--scenario", "1", "--trials", "1", "--arrays", "ULA", "--output-dir", outdir, "--test"]
    saved_dir = run_paper_experiments(argv)
    assert saved_dir == outdir
    csv_path = Path(outdir) / 'scenario1_baseline.csv'
    assert csv_path.exists()
    # File must have header + at least one data row
    with csv_path.open('r', encoding='utf-8') as f:
        lines = [l for l in f.readlines() if l.strip()]
    assert len(lines) >= 2
