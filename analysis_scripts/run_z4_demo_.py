# analysis_scripts/run_z4_demo.py

import os
import sys
import json
import argparse
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_processors.z4_processor import Z4ArrayProcessor  # noqa: E402


def pretty_phys_positions(grid_pos, d):
    phys = grid_pos * d
    if abs(d - round(d)) < 1e-12:
        return [int(x) for x in phys]
    return [float(f"{x:.6g}") for x in phys]


def main():
    parser = argparse.ArgumentParser(description="Run Z4 array geometry/coarray analysis.")
    parser.add_argument("--N", type=int, default=7, help="Number of sensors (>=5).")
    parser.add_argument("--d", type=float, default=1.0, help="Physical spacing.")
    parser.add_argument("--markdown", action="store_true", help="Print summary in Markdown if available.")
    parser.add_argument("--save-csv", action="store_true", help="Save summary CSV to results/summaries.")
    parser.add_argument("--show-weights", action="store_true", help="Print weight table.")
    parser.add_argument("--save-json", action="store_true", help="Save a small JSON sidecar of key metrics.")

    # NEW flags
    parser.add_argument("--holes", choices=["one", "both"], default="one",
                        help="Print holes as one-sided (<A) or two-sided.")
    parser.add_argument("--assert", dest="do_asserts", action="store_true",
                        help="Run Z4 invariants: w(1)=w(2)=0, L1=3, L2=3N-7 (only for N>=7).")
    args = parser.parse_args()

    print(f"--- Starting Array Z4 (N={args.N}, d={args.d}) Analysis Demo ---")

    z4 = Z4ArrayProcessor(N=args.N, d=args.d)
    data = z4.run_full_analysis()

    print(f"--- Starting analysis for Array Z4 (N={args.N}) (Weight-Constrained Sparse Array (Z4)) ---")
    print("--- Analysis Complete ---\n")

    # Physical positions
    grid = np.asarray(data.sensors_positions, dtype=int)
    phys = pretty_phys_positions(grid, args.d)

    print("\n========================================")
    print(f"Physical Sensor Positions (N={data.num_sensors}):")
    print(grid.tolist())
    print(f"Physical positions (grid*d): {phys}\n")

    # Summary table (exactly once)
    print("========================================")
    print("      ARRAY Z4 PERFORMANCE SUMMARY")
    print("========================================")
    df = data.performance_summary_table
    if args.markdown and hasattr(df, "to_markdown"):
        try:
            print(df.to_markdown(index=False))
        except Exception:
            print(df.to_string(index=False))
    else:
        print(df.to_string(index=False))

    # Key coarray data
    lags_2s = np.asarray(data.coarray_positions, dtype=int)  # two-sided lags, sorted
    seg = np.asarray(data.largest_contiguous_segment, dtype=int)  # one-sided segment
    L = int(len(seg))
    seg_range = f"[{int(seg[0])}:{int(seg[-1])}]" if L > 0 else "[]"

    # Canonical A = 3N - 7
    A = 3 * args.N - 7

    # One-sided holes (< A) from the processor
    holes_one = np.asarray(getattr(data, "missing_virtual_positions", []), dtype=int)

    # Two-sided holes: either provided or computed
    if hasattr(data, "holes_two_sided"):
        holes_two = np.asarray(data.holes_two_sided, dtype=int)
    else:
        Lmax = int(np.max(np.abs(lags_2s))) if lags_2s.size else 0
        full = np.arange(-Lmax, Lmax + 1, dtype=int)
        holes_two = np.setdiff1d(full, lags_2s)

    print("\n==================================================")
    print("KEY COARRAY DATA (integer lag grid)")
    print("==================================================")
    print(f"Unique lags (two-sided): {lags_2s.tolist()}")
    print(f"Largest one-sided contiguous segment: {seg.tolist()}  (L = {L}, range = {seg_range})")
    print(f"K_max (floor(L/2)): {int(data.max_detectable_sources)}")

    if args.holes == "both":
        holes_label = "Holes (two-sided)"
        holes_list = holes_two.tolist()
    else:
        holes_label = "Holes (one-sided, < A)"
        holes_list = holes_one.tolist()

    print(f"{holes_label}: {holes_list}  (count = {len(holes_list)})")
    print(f"Max positive lag (observed): {int(np.max(lags_2s)) if lags_2s.size else 0}")

    # Optional: show weight table
    if args.show_weights:
        wt_df = getattr(data, "weight_table", None)
        if wt_df is not None and len(wt_df) > 0:
            print("\n========================================")
            print("      WEIGHT TABLE (Lag, Weight)")
            print("========================================")
            # Reference by column names; avoid duplicate prints
            print(wt_df[["Lag", "Weight"]].to_string(index=False))

    # Optional artifacts
    if args.save_csv:
        os.makedirs("results/summaries", exist_ok=True)
        out_csv = f"results/summaries/z4_summary_N{args.N}_d{args.d}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\n[Saved] Performance summary CSV → {out_csv}")

    if args.save_json:
        os.makedirs("results/summaries", exist_ok=True)
        run_json = {
            "name": getattr(data, "name", f"Array Z4 (N={args.N})"),
            "type": getattr(data, "array_type", "Z4"),
            "N": int(args.N),
            "d": float(args.d),
            "grid_positions": grid.tolist(),
            "phys_positions": phys,
            "coarray_two_sided": lags_2s.tolist(),
            "segment_one_sided": seg.tolist(),
            "L": int(L),
            "A": int(A),
            "holes_one_sided": holes_one.tolist(),
            "holes_two_sided": holes_two.tolist(),
            "K_max": int(data.max_detectable_sources),
        }
        out_json = f"results/summaries/z4_run_N{args.N}_d{args.d}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(run_json, f, indent=2)
        print(f"[Saved] Run JSON → {out_json}")

    # Optional Z4 invariants (only meaningful for N >= 7)
    if args.do_asserts and args.N >= 7:
        wt_df = getattr(data, "weight_table", None)
        wt = {int(r["Lag"]): int(r["Weight"]) for _, r in wt_df.iterrows()} if (wt_df is not None and not wt_df.empty) else {}
        assert wt.get(1, 0) == 0 and wt.get(2, 0) == 0, "Z4 requires w(1)=0 and w(2)=0."
        assert L > 0, "No one-sided contiguous segment found."
        assert int(seg[0]) == 3, f"L1 must be 3; got {int(seg[0]) if L else 'NA'}"
        assert int(seg[-1]) == A, f"L2 must be A=3N-7={A}; got {int(seg[-1]) if L else 'NA'}"
        print("\n[Asserts] Z4 invariants passed: w(1)=w(2)=0, L1=3, L2=3N-7.")


if __name__ == "__main__":
    main()
