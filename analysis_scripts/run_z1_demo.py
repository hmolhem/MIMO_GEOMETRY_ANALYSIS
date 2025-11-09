import os
import sys
import argparse
import numpy as np

# add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geometry_processors.z1_processor import Z1ArrayProcessor

def pretty_physical_positions(grid_pos: np.ndarray, d: float):
    phys = grid_pos * d
    if abs(d - round(d)) < 1e-12:
        return phys.astype(int).tolist()
    return [float(f"{p:.6g}") for p in phys]

def print_key_coarray_data(data):
    lags = np.asarray(data.coarray_positions, dtype=int)          # two-sided
    seg  = np.asarray(data.largest_contiguous_segment, dtype=int) # one-sided
    holes = np.asarray(data.missing_virtual_positions, dtype=int) # one-sided

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
    parser = argparse.ArgumentParser(description="Run Z1 array geometry/coarray analysis.")
    parser.add_argument("--N", type=int, default=6, help="Number of sensors (>=3).")
    parser.add_argument("--d", type=float, default=1.0, help="Physical spacing.")
    parser.add_argument("--markdown", action="store_true", help="Print summary in Markdown.")
    parser.add_argument("--assert", dest="do_asserts", action="store_true",
                        help="Fail if Z1 invariants are violated (useful for tests).")
    parser.add_argument("--save", type=str, default="results/summaries",
                        help="Directory to save performance summary CSV (created if missing).")
    parser.add_argument("--show-weights", action="store_true", help="Also print weight table.")
    args = parser.parse_args()

    if args.N < 3:
        raise ValueError("Z1 requires N >= 3.")

    print(f"--- Starting Array Z1 (N={args.N}, d={args.d}) Analysis Demo ---")

    z1 = Z1ArrayProcessor(N=args.N, d=args.d)
    data = z1.run_full_analysis()

    # Physical positions
    grid_pos = np.asarray(data.sensors_positions, dtype=int)
    phys_pos = pretty_physical_positions(grid_pos, args.d)

    print("\n" + "="*40)
    print(f"Physical Sensor Positions (N={data.num_sensors}):")
    print(grid_pos.tolist())
    print(f"Physical positions (grid*d): {phys_pos}")

    # Performance summary
    print("\n" + "="*40)
    print("      ARRAY Z1 PERFORMANCE SUMMARY")
    print("="*40)
    if args.markdown and hasattr(data.performance_summary_table, "to_markdown"):
        try:
            print(data.performance_summary_table.to_markdown(index=False))
        except Exception:
            print(data.performance_summary_table.to_string(index=False))
    else:
        print(data.performance_summary_table.to_string(index=False))

    # Optional: print weights table
    if args.show_weights and getattr(data, "weight_table", None) is not None:
        print("\n" + "="*40)
        print("      WEIGHT TABLE (Lag, Weight)")
        print("="*40)
        print(data.weight_table.to_string(index=False))

    # Key coarray data
    print_key_coarray_data(data)

    # Optional invariants for Z1
    #   A = 2N-1 is max positive lag; expected one-sided holes = {1, A-1}
    lags = np.asarray(data.coarray_positions, dtype=int)
    lpos = set([x for x in lags.tolist() if x >= 0])
    A = 2 * args.N - 1
    expected_holes = {1, A - 1}

    # Holes from analysis (one-sided)
    holes = set(np.asarray(data.missing_virtual_positions, dtype=int).tolist())

    # Report differences
    print("\n[Z1 checks]")
    print(f"Max positive lag A = 2N-1 = {A}")
    print(f"Expected one-sided holes: {sorted(expected_holes)}")
    print(f"Computed one-sided holes: {sorted(holes)}")

    # w(1) should be 0
    w1 = 0
    try:
        # weight_table uses columns 'Lag' and 'Weight' in the refactor
        row = data.weight_table.loc[data.weight_table["Lag"] == 1, "Weight"]
        w1 = int(row.item()) if not row.empty else 0
    except Exception:
        pass
    print(f"Weight at Lag 1 (w(1)) = {w1}")

    if args.do_asserts:
        assert w1 == 0, "Z1 invariant failed: w(1) must be 0."
        assert expected_holes == holes, "Z1 invariant failed: holes must be {1, 2N-2}."
        # Basic DOF check: K_max = floor(L/2)
        L = len(np.asarray(data.largest_contiguous_segment, dtype=int))
        assert int(data.max_detectable_sources) == L // 2, "K_max != floor(L/2)."

    # Save CSV
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        out_path = os.path.join(args.save, f"z1_summary_N{args.N}_d{args.d}.csv")
        try:
            data.performance_summary_table.to_csv(out_path, index=False)
            print(f"\n[Saved] Performance summary CSV → {out_path}")
        except Exception as e:
            print(f"\n[Warn] Could not save CSV to {out_path}: {e}")

if __name__ == "__main__":
    main()
