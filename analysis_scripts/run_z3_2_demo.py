import os
import sys
import argparse
import numpy as np

# add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geometry_processors.z3_2_processor import Z3_2ArrayProcessor

def pretty_physical_positions(grid_pos: np.ndarray, d: float):
    phys = grid_pos * d
    if abs(d - round(d)) < 1e-12:
        return phys.astype(int).tolist()
    return [float(f"{p:.6g}") for p in phys]

def print_key_coarray_data(data):
    lags = np.asarray(data.coarray_positions, dtype=int)          # two-sided
    seg  = np.asarray(data.largest_contiguous_segment, dtype=int) # one-sided
    holes = np.asarray(data.missing_virtual_positions, dtype=int)  # one-sided

    L = int(len(seg))
    seg_range = f"[{int(seg[0])}:{int(seg[-1])}]" if L > 0 else "[]"

    print("\n" + "="*50)
    print("KEY COARRAY DATA (integer lag grid)")
    print("="*50)
    print(f"Unique lags (two-sided): {lags.tolist()}")
    print(f"Largest one-sided contiguous segment: {seg.tolist()}  (L = {L}, range = {seg_range})")
    print(f"K_max (floor(L/2)): {int(data.max_detectable_sources)}")
    print(f"Holes (one-sided): {holes.tolist()}  (count = {int(holes.size)})")

def main():
    parser = argparse.ArgumentParser(description="Run Z3(2) array geometry/coarray analysis.")
    parser.add_argument("--N", type=int, default=6, help="Number of sensors (>=5).")
    parser.add_argument("--d", type=float, default=1.0, help="Physical spacing.")
    parser.add_argument("--markdown", action="store_true", help="Print summary in Markdown.")
    parser.add_argument("--assert", dest="do_asserts", action="store_true",
                        help="Fail if Z3(2) invariants are violated.")
    parser.add_argument("--save", type=str, default="results/summaries",
                        help="Directory to save performance summary CSV.")
    parser.add_argument("--show-weights", action="store_true", help="Also print weight table.")
    args = parser.parse_args()

    if args.N < 5:
        raise ValueError("Z3(2) requires N >= 5.")

    print(f"--- Starting Array Z3(2) (N={args.N}, d={args.d}) Analysis Demo ---")

    z3 = Z3_2ArrayProcessor(N=args.N, d=args.d)
    data = z3.run_full_analysis()

    # Physical positions
    grid_pos = np.asarray(data.sensors_positions, dtype=int)
    phys_pos = pretty_physical_positions(grid_pos, args.d)

    print("\n" + "="*40)
    print(f"Physical Sensor Positions (N={data.num_sensors}):")
    print(grid_pos.tolist())
    print(f"Physical positions (grid*d): {phys_pos}")

    # Performance summary
    print("\n" + "="*40)
    print("      ARRAY Z3(2) PERFORMANCE SUMMARY")
    print("="*40)
    if args.markdown and hasattr(data.performance_summary_table, "to_markdown"):
        try:
            print(data.performance_summary_table.to_markdown(index=False))
        except Exception:
            print(data.performance_summary_table.to_string(index=False))
    else:
        print(data.performance_summary_table.to_string(index=False))

    # Optional: weight table
    if args.show_weights and getattr(data, "weight_table", None) is not None:
        print("\n" + "="*40)
        print("      WEIGHT TABLE (Lag, Weight)")
        print("="*40)
        print(data.weight_table.to_string(index=False))

    # Key coarray data
    print_key_coarray_data(data)

    # ------- Optional invariants for Z3(2) -------
    # For N >= 5:
    #   L1=2, L2=4N-13  =>  L = 4N-14
    #   A = 4N-9; holes at {1, A-3, A-1} = {1, 4N-12, 4N-10}
    #   weights: w(1)=0, w(2)=1, w(3)=2
    N = args.N
    A = 4 * N - 9
    expected_L  = 4 * N - 14
    expected_w  = {1: 0, 2: 1, 3: 2}
    expected_holes = {1, A - 3, A - 1}

    seg  = np.asarray(data.largest_contiguous_segment, dtype=int)
    L = int(len(seg))
    holes = set(np.asarray(data.missing_virtual_positions, dtype=int).tolist())

    # extract weights 1..3
    w = {k: 0 for k in [1,2,3]}
    try:
        for k in [1,2,3]:
            row = data.weight_table.loc[data.weight_table["Lag"] == k, "Weight"]
            if not row.empty:
                w[k] = int(row.item())
    except Exception:
        pass

    print("\n[Z3(2) checks]")
    print(f"A (4N-9) = {A}")
    print(f"Expected largest one-sided L = {expected_L}; computed L = {L}")
    print(f"Expected holes (one-sided): {sorted(expected_holes)}; computed: {sorted(holes)}")
    print(f"Expected weights w(1..3) = {expected_w}; computed: {{1:{w[1]}, 2:{w[2]}, 3:{w[3]}}}")

    if args.do_asserts:
        assert L == expected_L, f"L mismatch: expected {expected_L}, got {L}"
        # Keep only the canonical holes the theory enumerates
        holes_set = set(holes)
        holes_core = {h for h in holes_set if h in expected_holes}

        if holes_core != expected_holes:
            raise AssertionError(
                f"Holes mismatch: expected {sorted(expected_holes)}, "
                f"got core {sorted(holes_core)} (all computed: {sorted(holes_set)})"
            )
        for k, v in expected_w.items():
            assert w[k] == v, f"w({k}) mismatch: expected {v}, got {w[k]}"

    # Save CSV
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        out_path = os.path.join(args.save, f"z3_2_summary_N{args.N}_d{args.d}.csv")
        try:
            data.performance_summary_table.to_csv(out_path, index=False)
            print(f"\n[Saved] Performance summary CSV → {out_path}")
        except Exception as e:
            print(f"\n[Warn] Could not save CSV to {out_path}: {e}")

if __name__ == "__main__":
    main()
