# analysis_scripts/run_nested_demo.py

import os
import sys

# Add the project root (MIMO_GEOMETRY_ANALYSIS) to the path
# This allows the script to find the 'geometry_processors' module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import the specific processor class
from core.radarpy.geometry.nested_processor import NestedArrayProcessor

def main():
    """Initializes and runs the analysis for a Nested Array (NA)."""
    
    # --- Configuration ---
    # We choose N1=2, N2=2 for N=4 total sensors (standard design)
    N1 = 2
    N2 = 2
    
    print(f"--- Starting Nested Array (N1={N1}, N2={N2}) Analysis Demo ---")
    
    # Instantiate the processor
    na_processor = NestedArrayProcessor(N1=N1, N2=N2, d=1)
    
    # Run the full analysis pipeline
    analysis_results = na_processor.run_full_analysis()
    
    # Display the physical sensor positions
    print("\n" + "="*40)
    print(f"Physical Sensor Positions (N={analysis_results.num_sensors}):")
    print(analysis_results.sensors_positions)
    
    # Display the final summary
    print("\n" + "="*40)
    print("      FINAL PERFORMANCE SUMMARY")
    print("="*40)
    print(analysis_results.performance_summary_table.to_markdown(index=False))
    
    # --- KEY COARRAY DATA (integer lag grid) ---
    import numpy as np

    lags = np.asarray(analysis_results.coarray_positions, dtype=int)                 # two-sided, sorted
    segment = np.asarray(analysis_results.largest_contiguous_segment, dtype=int)     # one-sided (>=0)
    holes = np.asarray(analysis_results.missing_virtual_positions, dtype=int)        # one-sided (>=0)

    L = int(len(segment))
    seg_range = f"[{int(segment[0])}:{int(segment[-1])}]" if L > 0 else "[]"

    print("\n" + "="*50)
    print("KEY COARRAY DATA (integer lag grid)")
    print("="*50)
    print(f"Unique lags (two-sided): {lags.tolist()}")
    print(f"Largest one-sided contiguous segment: {segment.tolist()}  (L = {L}, range = {seg_range})")
    print(f"K_max (floor(L/2)): {analysis_results.max_detectable_sources}")
    print(f"Holes (one-sided): {holes.tolist()}  (count = {int(holes.size)})")

if __name__ == "__main__":
    main()

