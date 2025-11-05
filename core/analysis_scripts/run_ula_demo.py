# analysis_scripts/run_ula_demo.py

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add the project root (one level up) to the import path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.radarpy.geometry.ula_processors import ULArrayProcessor


def pretty_physical_positions(grid_pos, d):
    phys = grid_pos * d
    # If d is effectively an integer (1.0, 2.0, ...), print ints
    if abs(d - round(d)) < 1e-12:
        return phys.astype(int).tolist()
    # Otherwise print compact floats
    return [float(f"{p:.6g}") for p in phys]


def main():
    parser = argparse.ArgumentParser(description="Run ULA geometry/coarray analysis.")
    parser.add_argument("--N", type=int, default=4, help="Number of sensors (>=2).")
    parser.add_argument("--d", type=float, default=1.0, help="Physical inter-sensor spacing.")
    parser.add_argument("--save-prefix", type=str, default="", 
                        help="Optional file prefix to save outputs (e.g., results/summaries/ula_N4_d1). Saves CSVs.")
    parser.add_argument("--markdown", action="store_true",
                        help="Print the summary table in Markdown (requires tabulate).")
    args = parser.parse_args()

    print(f"--- Starting ULA Analysis Demo (N={args.N}, d={args.d}) ---")

    # Instantiate and run the full analysis
    ula = ULArrayProcessor(N=args.N, d=args.d)
    data = ula.run_full_analysis()

    # Grid vs. physical positions
    grid_pos = np.asarray(data.sensors_positions, dtype=int)
    phys_pos = pretty_physical_positions(grid_pos, args.d)

    print("\n" + "="*50)
    print("PHYSICAL ARRAY")
    print("="*50)
    print(f"Name: {data.name}")
    print(f"Grid positions: {grid_pos.tolist()}")
    print(f"Physical positions (grid*d): {phys_pos}")
    print(f"Sensor spacing d: {args.d}")

    # Performance summary
    print("\n" + "="*50)
    print("FINAL PERFORMANCE SUMMARY")
    print("="*50)
    if args.markdown and hasattr(data.performance_summary_table, "to_markdown"):
        try:
            # to_markdown needs 'tabulate' installed
            print(data.performance_summary_table.to_markdown(index=False))
        except Exception:
            print(data.performance_summary_table.to_string(index=False))
    else:
        print(data.performance_summary_table.to_string(index=False))

    # Key coarray info
    print("\n" + "="*50)
    print("KEY COARRAY DATA (integer lag grid)")
    print("="*50)
    print(f"Unique lags (two-sided): {data.unique_differences}")
    print(f"One-sided largest contiguous segment: {data.largest_contiguous_segment}")
    print(f"K_max (floor(L/2)): {data.max_detectable_sources}")
    print(f"Holes (one-sided): {data.missing_virtual_positions} (count={data.num_holes})")

    # Optional save
    if args.save_prefix:
        base = args.save_prefix
        os.makedirs(os.path.dirname(base), exist_ok=True)

        # Save performance summary
        perf_path = f"{base}_perf_summary.csv"
        data.performance_summary_table.to_csv(perf_path, index=False)

        # Save weight table
        if data.weight_table is not None:
            wt_path = f"{base}_weight_table.csv"
            data.weight_table.to_csv(wt_path, index=False)

        # Save differences table (can be large; save only if you want it)
        diffs_path = f"{base}_differences_sample.csv"
        data.all_differences_table.head(1000).to_csv(diffs_path, index=False)

        print("\nSaved:")
        print(f"  - {perf_path}")
        print(f"  - {wt_path}")
        print(f"  - {diffs_path}")

if __name__ == "__main__":
    main()


