import os
import sys
import argparse
import numpy as np

# add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.radarpy.geometry.nested_processor import NestedArrayProcessor

def pretty_physical_positions(grid_pos, d):
    phys = grid_pos * d
    if abs(d - round(d)) < 1e-12:
        return phys.astype(int).tolist()
    return [float(f"{p:.6g}") for p in phys]

def main():
    parser = argparse.ArgumentParser(description="Run Nested Array geometry/coarray analysis.")
    parser.add_argument("--N1", type=int, default=2, help="Dense subarray size (>=1).")
    parser.add_argument("--N2", type=int, default=3, help="Sparse subarray size (>=1).")
    parser.add_argument("--d",  type=float, default=1.0, help="Physical spacing.")
    parser.add_argument("--markdown", action="store_true", help="Print summary in Markdown if available.")
    args = parser.parse_args()

    print(f"--- Starting Nested Array (N1={args.N1}, N2={args.N2}, d={args.d}) Analysis Demo ---")

    na = NestedArrayProcessor(N1=args.N1, N2=args.N2, d=args.d)
    data = na.run_full_analysis()

    grid_pos = np.asarray(data.sensors_positions, dtype=int)
    phys_pos = pretty_physical_positions(grid_pos, args.d)

    print("\n" + "="*50)
    print("PHYSICAL ARRAY")
    print("="*50)
    print(f"Name: {data.name}")
    print(f"Grid positions: {grid_pos.tolist()}")
    print(f"Physical positions (grid*d): {phys_pos}")
    print(f"Sensor spacing d: {args.d}")

    print("\n" + "="*50)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*50)
    if args.markdown and hasattr(data.performance_summary_table, "to_markdown"):
        try:
            print(data.performance_summary_table.to_markdown(index=False))
        except Exception:
            print(data.performance_summary_table.to_string(index=False))
    else:
        print(data.performance_summary_table.to_string(index=False))

    print("\n" + "="*50)
    lags = np.asarray(data.coarray_positions, dtype=int)          # two-sided, sorted
    lags_pos = lags[lags >= 0]                                    # one-sided (>=0)
    aperture_two_sided = int(lags.max() - lags.min()) if lags.size else 0
    max_positive_lag = int(lags_pos.max()) if lags_pos.size else 0

    segment = np.asarray(data.largest_contiguous_segment, dtype=int)
    L = int(len(segment))
    holes = np.asarray(data.missing_virtual_positions, dtype=int)
    num_holes = int(holes.size)

    print("\n" + "="*50)
    print("KEY COARRAY DATA (integer lag grid)")
    print("="*50)
    print(f"Unique lags (two-sided): {lags}")
    print(f"Coarray aperture (two-sided span): {aperture_two_sided} lags")
    print(f"Max positive lag: {max_positive_lag}")
    if L > 0:
        print(f"One-sided largest contiguous segment: {segment}  (L = {L})")
    else:
        print("One-sided largest contiguous segment: []  (L = 0)")
    print(f"K_max (floor(L/2)): {data.max_detectable_sources}")
    print(f"Holes (one-sided): {holes.tolist()}  (count = {num_holes})")

if __name__ == "__main__":
    main()


