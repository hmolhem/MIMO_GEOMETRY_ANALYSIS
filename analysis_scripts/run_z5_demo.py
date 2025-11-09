# analysis_scripts/run_z5_demo.py
import os
import sys
import argparse
import numpy as np
import pandas as pd
import inspect

# add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_processors.z5_processor import Z5ArrayProcessor

def pretty_physical_positions(grid_pos, d):
    phys = grid_pos * d
    if abs(d - round(d)) < 1e-12:
        return phys.astype(int).tolist()
    return [float(f"{p:.6g}") for p in phys]

def main():
    parser = argparse.ArgumentParser(description="Run Z5 array geometry/coarray analysis.")
    parser.add_argument("--N", type=int, default=7, help="Number of sensors (>=5).")
    parser.add_argument("--d", type=float, default=1.0, help="Physical spacing.")
    parser.add_argument("--markdown", action="store_true", help="Print summary in Markdown if available.")
    parser.add_argument("--save-csv", action="store_true", help="Save summary CSV to results/summaries.")
    parser.add_argument("--show-weights", action="store_true", help="Print weight table.")
    parser.add_argument("--save-json", action="store_true", help="Save a small JSON sidecar of key metrics.")
    parser.add_argument("--holes", choices=["one", "both"], default="one", help="Show holes as one-sided (<A_obs) or two-sided.")
    parser.add_argument("--assert", dest="assert_", action="store_true",
                        help="Check Z5 invariants (w1=w2=0, w3>=1, nonempty segment).")
    parser.add_argument("--latex", action="store_true",
                        help="Output single LaTeX table row: N & |S| & |Λ| & L & K_max & A_obs & holes_1s \\\\")
    args = parser.parse_args()

    print(f"--- Starting Array Z5 (N={args.N}, d={args.d}) Analysis Demo ---")
    print(f"[Import] Z5 source file: {inspect.getsourcefile(Z5ArrayProcessor)}")
    print(f"--- Starting analysis for Array Z5 (N={args.N}) (Weight-Constrained Sparse Array (Z5)) ---")

    z5 = Z5ArrayProcessor(N=args.N, d=args.d)
    # quick peek at the constructed geometry BEFORE analysis
    print(f"[Geometry preview] sensors (grid): {z5.data.sensors_positions}")

    data = z5.run_full_analysis(verbose=False)  # Quiet mode to avoid duplicate banners
    
    # Optional asserts block (mirroring Z4) - Z5 invariants validation
    if args.assert_:
        wt_df = getattr(data, "weight_table", None)
        if wt_df is not None and not wt_df.empty:
            w = {int(r["Lag"]): int(r["Weight"]) for _, r in wt_df.iterrows()}
            assert w.get(1, 0) == 0 and w.get(2, 0) == 0, "Z5 requires w(1)=0 and w(2)=0."
            assert w.get(3, 0) >= 1, "Z5 expects nonzero weight at lag 3."
        seg = getattr(data, "largest_contiguous_segment", None)
        assert seg is not None and len(seg) > 0, "No one-sided contiguous segment found."
        print("[Asserts] Z5 invariants passed.")
    
    # Basic validation (always runs)
    seg = getattr(data, "largest_contiguous_segment", None)
    if seg is None or len(seg) == 0:
        raise AssertionError("Z5: No one-sided contiguous segment found.")
    
    print("--- Analysis Complete ---\n")

    grid_pos = np.asarray(data.sensors_positions, dtype=int)
    phys_pos = pretty_physical_positions(grid_pos, args.d)

    print("\n" + "="*40)
    print(f"Physical Sensor Positions (N={data.num_sensors}):")
    print(grid_pos.tolist())
    print(f"Physical positions (grid*d): {phys_pos}")

    print("\n" + "="*40)
    print("      ARRAY Z5 PERFORMANCE SUMMARY")
    print("="*40)
    if args.markdown and hasattr(data.performance_summary_table, "to_markdown"):
        try:
            print(data.performance_summary_table.to_markdown(index=False))
        except Exception:
            print(data.performance_summary_table.to_string(index=False))
    else:
        print(data.performance_summary_table.to_string(index=False))

    lags = np.asarray(data.coarray_positions if data.coarray_positions is not None else [], dtype=int)
    one = np.unique(lags[lags >= 0]) if lags.size else np.array([], dtype=int)
    seg = np.asarray(data.largest_contiguous_segment if data.largest_contiguous_segment is not None else [], dtype=int)
    holes_one = np.asarray(data.missing_virtual_positions if data.missing_virtual_positions is not None else [], dtype=int)
    L = int(seg.size)
    seg_range = f"[{int(seg[0])}:{int(seg[-1])}]" if L > 0 else "[]"
    A_obs = int(one.max()) if one.size else 0
    
    # Two-sided holes (for --holes both)
    if args.holes == "both":
        # Compute two-sided holes: [-A_obs .. A_obs] minus observed lags
        domain_two = np.arange(-A_obs, A_obs + 1, dtype=int) if A_obs > 0 else np.array([0], dtype=int)
        holes_two = np.setdiff1d(domain_two, lags) if lags.size else domain_two.copy()
        holes_label = "Holes (two-sided)"
        holes_list = holes_two.tolist()
    else:
        holes_label = "Holes (one-sided, < A_obs)"
        holes_list = holes_one.tolist()

    print("\n" + "="*50)
    print("KEY COARRAY DATA (integer lag grid)")
    print("="*50)
    print(f"Unique lags (two-sided): {lags.tolist() if lags.size else []}")
    print(f"Largest one-sided contiguous segment: {seg.tolist() if L else []}  (L = {L}, range = {seg_range})")
    print(f"{holes_label}: {holes_list}  (count = {len(holes_list)})")
    print(f"Max positive lag (observed): {A_obs}")

    if args.show_weights:
        wt_df = data.weight_table if isinstance(data.weight_table, pd.DataFrame) else pd.DataFrame(columns=["Lag", "Weight"])
        print("\n" + "="*40)
        print("      WEIGHT TABLE (Lag, Weight)")
        print("="*40)
        print(wt_df.to_string(index=False))

    if args.save_csv:
        os.makedirs("results/summaries", exist_ok=True)
        out_csv = f"results/summaries/z5_summary_N{args.N}_d{args.d}.csv"
        data.performance_summary_table.to_csv(out_csv, index=False)
        print(f"\n[Saved] Performance summary CSV → {out_csv}")

    if args.save_json:
        os.makedirs("results/summaries", exist_ok=True)
        import json
        wt = {int(r["Lag"]): int(r["Weight"]) for _, r in data.weight_table.iterrows()} if isinstance(data.weight_table, pd.DataFrame) and not data.weight_table.empty else {}
        payload = {
            "name": data.name if hasattr(data, "name") else f"Array Z5 (N={args.N})",
            "N": args.N,
            "d": args.d,
            "sensors_grid": grid_pos.tolist(),
            "lags_two_sided": lags.tolist() if lags.size else [],
            "largest_segment_one_sided": seg.tolist() if L else [],
            "holes_one_sided": holes_one.tolist(),
            "A_obs": A_obs,
            "w1": wt.get(1, 0),
            "w2": wt.get(2, 0),
            "w3": wt.get(3, 0),
        }
        out_json = f"results/summaries/z5_run_N{args.N}_d{args.d}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[Saved] Run JSON → {out_json}")

    # LaTeX table row output
    if args.latex:
        # Extract key metrics: N & |S| & |Λ| & L & K_max & A_obs & holes_1s \\
        N_val = int(args.N)
        S_val = int(data.num_sensors)  # |S|
        Lambda_val = int(len(lags)) if lags.size else 0  # |Λ| (unique lags)
        L_val = int(L)  # L (contiguous segment length)  
        K_max_val = int(data.max_detectable_sources) if hasattr(data, 'max_detectable_sources') else int(L // 2)  # K_max
        A_obs_val = int(A_obs)  # A_obs (max positive lag observed)
        holes_1s_val = int(len(holes_one))  # holes count (one-sided)
        
        latex_row = f"{N_val} & {S_val} & {Lambda_val} & {L_val} & {K_max_val} & {A_obs_val} & {holes_1s_val} \\\\"
        print(latex_row)

    # Persist geometry/lag artifacts for future comparison
    if args.save_csv or args.save_json:
        import json
        # Sensor geometry
        os.makedirs("results/geometries", exist_ok=True)
        geom_csv = f"results/geometries/z5_N{args.N}_d{args.d}_sensors.csv"
        pd.DataFrame({"sensor_index": range(len(grid_pos)), "grid_position": grid_pos}).to_csv(geom_csv, index=False)
        print(f"[Saved] Sensor geometry CSV → {geom_csv}")
        
        # Coarray lags
        os.makedirs("results/coarrays", exist_ok=True)
        lags_txt = f"results/coarrays/z5_N{args.N}_d{args.d}_lags.txt" 
        with open(lags_txt, "w") as f:
            f.write("# Z5 Two-sided lags (sorted unique)\n")
            f.write(" ".join(map(str, sorted(lags.tolist()))))
        print(f"[Saved] Coarray lags → {lags_txt}")
        
        # Optional: holes artifacts
        holes_one_json = f"results/coarrays/z5_N{args.N}_d{args.d}_holes_one_sided.json"
        with open(holes_one_json, "w") as f:
            json.dump({"holes_one_sided": holes_one.tolist(), "A_obs": A_obs}, f, indent=2)
        
        if args.holes == "both":
            holes_two_json = f"results/coarrays/z5_N{args.N}_d{args.d}_holes_two_sided.json"
            domain_two = np.arange(-A_obs, A_obs + 1, dtype=int) if A_obs > 0 else np.array([0], dtype=int)
            holes_two = np.setdiff1d(domain_two, lags) if lags.size else domain_two.copy()
            with open(holes_two_json, "w") as f:
                json.dump({"holes_two_sided": holes_two.tolist(), "domain": f"[-{A_obs}:{A_obs}]"}, f, indent=2)
            print(f"[Saved] Two-sided holes → {holes_two_json}")

if __name__ == "__main__":
    main()
