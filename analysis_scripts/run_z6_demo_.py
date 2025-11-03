# analysis_scripts/run_z6_demo.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

# Ensure project root import if run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry_processors.z6_processor import Z6ArrayProcessor  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser(description="Run Z6 array geometry/coarray analysis.")
    ap.add_argument("--N", type=int, default=7, help="Number of sensors (demo is fixed at N=7).")
    ap.add_argument("--d", type=float, default=1.0, help="Physical spacing multiplier.")
    ap.add_argument("--markdown", action="store_true", help="Print summary in Markdown table.")
    ap.add_argument("--save-csv", action="store_true", help="Save summary CSV to results/summaries.")
    ap.add_argument("--show-weights", action="store_true", help="Print weight table.")
    ap.add_argument("--save-json", action="store_true", help="Save small JSON sidecar of key metrics.")
    ap.add_argument("--holes", choices=["one", "both"], default="one", help="Which holes list to print.")
    ap.add_argument("--assert", dest="do_assert", action="store_true", help="Run mild Z6 checks (optional).")
    args = ap.parse_args()

    print(f"--- Starting Array Z6 (N={args.N}, d={args.d}) Analysis Demo ---")

    z6 = Z6ArrayProcessor(N=args.N, d=args.d)
    print(f"[Geometry preview] sensors (grid): {z6.data.sensor_positions}")

    # Do the analysis (calls analyze_coarray() then generate_performance_summary())
    z6.analyze_geometry()
    z6.analyze_coarray()
    z6.generate_performance_summary()
    print("--- Analysis Complete ---\n")

    # ---- Pretty print summary ----
    rows = z6.data.summary_rows
    if args.markdown:
        # markdown table
        print("========================================")
        print("      ARRAY Z6 PERFORMANCE SUMMARY")
        print("========================================")
        print("| Metrics                                  | Value   |")
        print("|:-----------------------------------------|:--------|")
        for k, v in rows:
            print(f"| {k:<41} | {v!s:<7} |")
    else:
        df = pd.DataFrame(rows, columns=["Metrics", "Value"])
        print("========================================")
        print("      ARRAY Z6 PERFORMANCE SUMMARY")
        print("========================================")
        print(df.to_string(index=False))

    # ---- Coarray details ----
    lags_2s = np.asarray(z6.data.coarray_positions, dtype=int)
    seg = z6.data.largest_contiguous_segment if z6.data.largest_contiguous_segment is not None else np.array([], dtype=int)
    L = len(seg)

    print("\n==================================================")
    print("KEY COARRAY DATA (integer lag grid)")
    print("==================================================")
    print(f"Unique lags (two-sided): {sorted(lags_2s.tolist()) if lags_2s.size else []}")
    if L:
        print(f"Largest one-sided contiguous segment: {seg.tolist()}  (L = {L}, range = [{int(seg[0])}:{int(seg[-1])}])")
    else:
        print("Largest one-sided contiguous segment: []  (L = 0, range = [NA:NA])")

    if args.holes == "both":
        print(f"Holes (two-sided): {z6.data.holes_two_sided}  (count = {len(z6.data.holes_two_sided)})")
    else:
        print(f"Holes (one-sided, < A_obs): {z6.data.holes_one_sided}  (count = {len(z6.data.holes_one_sided)})")

    print(f"Max positive lag (observed): {z6.data.A_obs}")

    # ---- weight table (optional) ----
    if args.show_weights:
        print("\n========================================")
        print("      WEIGHT TABLE (Lag, Weight)")
        print("========================================")
        print(" Lag  Weight")
        for lag, w in z6.data.weight_table_rows:
            print(f"{lag:4d}{w:8d}")

    # ---- optional asserts ----
    if args.do_assert:
        wt = dict(z6.data.weight_table_rows)
        # gentle consistency checks (no strict Z6 theorem enforced here)
        assert wt.get(1, 0) == 0 and wt.get(2, 0) == 0, "Expected w(1)=0 and w(2)=0 for sparse Z-family."
        assert z6.data.A_obs >= 10, "Observed aperture looks too small for the chosen Z6 demo geometry."
        print("\n[Asserts] Z6 invariants passed.")

    # ---- save artifacts ----
    if args.save_csv or args.save_json:
        out_root = ROOT / "results"
        out_root.mkdir(parents=True, exist_ok=True)

    if args.save_csv:
        # summary csv
        df = pd.DataFrame(rows, columns=["Metrics", "Value"])
        (out_root / "summaries").mkdir(parents=True, exist_ok=True)
        df.to_csv(out_root / "summaries" / f"z6_summary_N{args.N}_d{args.d}.csv", index=False)

    if args.save_json:
        import json
        mini = {
            "name": z6.data.name,
            "array_type": z6.data.array_type,
            "N": args.N,
            "d": args.d,
            "A_obs": z6.data.A_obs,
            "L": int(L),
            "seg": seg.tolist() if L else [],
            "holes_one_sided": z6.data.holes_one_sided,
        }
        (out_root / "summaries").mkdir(parents=True, exist_ok=True)
        with (out_root / "summaries" / f"z6_run_N{args.N}_d{args.d}.json").open("w", encoding="utf-8") as f:
            json.dump(mini, f, indent=2)

    # Always dump geometry + coarray for reproducibility
    geo_dir = ROOT / "results" / "geometries"
    co_dir = ROOT / "results" / "coarrays"
    geo_dir.mkdir(parents=True, exist_ok=True)
    co_dir.mkdir(parents=True, exist_ok=True)
    z6.save_geometry_csv(geo_dir / f"z6_N{args.N}_d{args.d}_sensors.csv")
    z6.save_coarray_txt(co_dir / f"z6_N{args.N}_d{args.d}_lags.txt")
    z6.save_two_sided_holes_json(co_dir / f"z6_N{args.N}_d{args.d}_holes_two_sided.json")


if __name__ == "__main__":
    main()