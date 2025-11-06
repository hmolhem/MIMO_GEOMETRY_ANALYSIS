# analysis_scripts/run_z6_demo.py
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from core.radarpy.geometry.z6_processor import Z6ArrayProcessor

def _safe_df_from_rows(rows, cols):
    if rows is None:
        return pd.DataFrame(columns=cols)
    if isinstance(rows, pd.DataFrame):
        return rows
    return pd.DataFrame(rows, columns=cols)

def main():
    parser = argparse.ArgumentParser(
        description="Run Z6 array geometry/coarray analysis."
    )
    parser.add_argument("--N", type=int, default=7, help="Number of sensors (>=5).")
    parser.add_argument("--d", type=float, default=1.0, help="Physical spacing.")
    parser.add_argument("--markdown", action="store_true", help="Print summary in Markdown if available.")
    parser.add_argument("--save-csv", action="store_true", help="Save summary CSV to results/summaries.")
    parser.add_argument("--show-weights", action="store_true", help="Print weight table.")
    parser.add_argument("--save-json", action="store_true", help="Save a small JSON sidecar of key metrics.")
    parser.add_argument("--holes", choices=["one","both"], default="one",
                        help="Print holes as one-sided (<A_obs) or two-sided.")
    parser.add_argument("--assert", dest="do_assert", action="store_true",
                        help="Run Z6 invariants if defined.")
    args = parser.parse_args()

    print(f"--- Starting Array Z6 (N={args.N}, d={args.d}) Analysis Demo ---")

    # Instantiate processor
    z6 = Z6ArrayProcessor(N=args.N, d=args.d)

    # Preview geometry if available
    if hasattr(z6, "data") and getattr(z6.data, "sensors_positions", None):
        print(f"[Geometry preview] sensors (grid): {z6.data.sensors_positions}")

    # Run analysis (fills z6.data)
    data_obj = z6.run_full_analysis()   # returns Z6Data object (same as z6.data)
    d = z6.data if hasattr(z6, "data") and z6.data is not None else data_obj

    # ---- Pull commonly used attributes safely ----
    name = getattr(d, "name", f"Array Z6 (N={args.N})")
    array_type = getattr(d, "array_type", "Weight-Constrained Sparse Array (Z6)")
    sensors_grid = getattr(d, "sensors_positions", getattr(d, "sensor_positions", []))
    sensors_phys = getattr(d, "sensor_positions_physical", [s * args.d for s in sensors_grid])

    # coarray info
    lags_2s = np.asarray(getattr(d, "coarray_positions", getattr(d, "coarray_lags", [])), dtype=int)
    A_obs = int(getattr(d, "A_obs", np.max(lags_2s) if lags_2s.size else 0))
    seg = getattr(d, "largest_contiguous_segment", np.array([], dtype=int))
    L = int(len(seg))

    # holes (one-sided and two-sided)
    holes_one = np.asarray(getattr(d, "holes_one_sided", []), dtype=int)
    holes_two = np.asarray(getattr(d, "holes_two_sided", []), dtype=int)
    if holes_two.size == 0:
        # derive two-sided holes from observed lags
        Lmax = int(np.max(np.abs(lags_2s))) if lags_2s.size else 0
        full = np.arange(-Lmax, Lmax + 1, dtype=int)
        holes_two = np.setdiff1d(full, lags_2s)

    # weight table
    weight_rows = getattr(d, "weight_table_rows", None)
    if hasattr(d, "weights_df"):
        wt_df = d.weights_df
    elif hasattr(d, "weight_df"):
        wt_df = d.weight_df
    else:
        wt_df = _safe_df_from_rows(weight_rows, cols=["Lag", "Weight"])
    wt = {int(r["Lag"]): int(r["Weight"]) for _, r in wt_df.iterrows()} if not wt_df.empty else {}

    # summary table
    if hasattr(d, "summary_df"):
        summary_df = d.summary_df
    else:
        # build from rows if needed
        rows = getattr(d, "summary_rows", None)
        summary_df = _safe_df_from_rows(rows, cols=["Metrics", "Value"])

    # Optional asserts (only if your Z6 invariants make sense)
    if args.do_assert:
        # Example invariants (adjust if your Z6 spec differs)
        # Here we enforce w(1)=w(2)=0 and L >= 1 (since Z6 isn't necessarily Z4/Z5)
        if wt:
            assert wt.get(1, 0) == 0 and wt.get(2, 0) == 0, "Z6 requires w(1)=0 and w(2)=0."
        assert L >= 1, "Contiguous segment length L must be >= 1 for Z6."
        print("[Asserts] Z6 invariants passed.")

    # ---- Print human-readable outputs ----
    print("\n========================================")
    print(f"Physical Sensor Positions (N={args.N}):")
    print(sensors_grid)
    print(f"Physical positions (grid*d): {sensors_phys}")

    print("\n========================================")
    print("      ARRAY Z6 PERFORMANCE SUMMARY")
    print("========================================")
    if args.markdown:
        # Pretty markdown if desired
        print(summary_df.to_markdown(index=False))
    else:
        print(summary_df.to_string(index=False))

    print("\n==================================================")
    print("KEY COARRAY DATA (integer lag grid)")
    print("==================================================")
    uniq = np.unique(lags_2s).tolist() if lags_2s.size else []
    print(f"Unique lags (two-sided): {uniq}")

    if L > 0:
        l1, l2 = int(seg[0]), int(seg[-1])
        print(f"Largest one-sided contiguous segment: {seg.tolist()}  (L = {L}, range = [{l1}:{l2}])")
    else:
        print("Largest one-sided contiguous segment: []  (L = 0)")

    # Choose which holes to display
    if args.holes == "one":
        holes_label = "Holes (one-sided, < A_obs)"
        holes_list = holes_one.tolist() if holes_one.size else []
    else:
        holes_label = "Holes (two-sided)"
        holes_list = holes_two.tolist()
    print(f"{holes_label}: {holes_list}  (count = {len(holes_list)})")

    if lags_2s.size:
        print(f"Max positive lag (observed): {int(np.max(lags_2s))}")

    # ---- Optional prints ----
    if args.show_weights and not wt_df.empty:
        print("\n========================================")
        print("      WEIGHT TABLE (Lag, Weight)")
        print("========================================")
        # fixed width
        print(" Lag  Weight")
        for _, row in wt_df.iterrows():
            print(f"{int(row['Lag']):>4} {int(row['Weight']):>7}")

    # ---- Saves ----
    out_root = Path("results")
    if args.save_csv:
        (out_root / "summaries").mkdir(parents=True, exist_ok=True)
        summary_path = out_root / "summaries" / f"z6_summary_N{args.N}_d{args.d}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[Saved] Performance summary CSV -> {summary_path}")

        (out_root / "geometries").mkdir(parents=True, exist_ok=True)
        sensors_path = out_root / "geometries" / f"z6_N{args.N}_d{args.d}_sensors.csv"
        pd.DataFrame({"grid": sensors_grid, "physical": sensors_phys}).to_csv(sensors_path, index=False)
        print(f"[Saved] Sensor geometry CSV -> {sensors_path}")

        (out_root / "coarrays").mkdir(parents=True, exist_ok=True)
        lags_path = out_root / "coarrays" / f"z6_N{args.N}_d{args.d}_lags.txt"
        with open(lags_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(map(str, uniq)))
        print(f"[Saved] Coarray lags -> {lags_path}")

        holes2_path = out_root / "coarrays" / f"z6_N{args.N}_d{args.d}_holes_two_sided.json"
        with open(holes2_path, "w", encoding="utf-8") as fh:
            json.dump({"holes_two_sided": holes_list if args.holes == "both" else holes_two.tolist()}, fh, indent=2)
        print(f"[Saved] Two-sided holes -> {holes2_path}")

    if args.save_json:
        payload = {
            "name": name,
            "array_type": array_type,
            "N": args.N,
            "d": args.d,
            "sensors_grid": sensors_grid,
            "sensors_physical": sensors_phys,
            "unique_lags_two_sided": uniq,
            "largest_segment": seg.tolist() if L > 0 else [],
            "L": L,
            "holes_one_sided": holes_one.tolist() if holes_one.size else [],
            "holes_two_sided": holes_two.tolist(),
            "A_obs": A_obs,
            "weights": wt,
        }
        (out_root / "summaries").mkdir(parents=True, exist_ok=True)
        jpath = out_root / "summaries" / f"z6_run_N{args.N}_d{args.d}.json"
        with open(jpath, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[Saved] Run JSON -> {jpath}")

if __name__ == "__main__":
    main()


