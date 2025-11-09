# analysis_scripts/run_z1_demo.py

import os
import sys

# Ensure the project root is in the path for module imports
# This is a robust way to handle the imports from the project root.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import the specific processor class
from core.radarpy.geometry.z1_processor import Z1ArrayProcessor

def main():
    """Initializes and runs the analysis for the Array Z1 (w(1)=0)."""
    
    # --- Configuration ---
    N_sensors = 6
    
    print(f"--- Starting Array Z1 (N={N_sensors}) Analysis Demo ---")
    
    # Instantiate the processor: Array Z1
    z1_processor = Z1ArrayProcessor(N=N_sensors, d=1)
    
    # Run the full analysis pipeline
    analysis_results = z1_processor.run_full_analysis()
    
    # Display the results
    print("\n" + "="*40)
    print(f"Physical Sensor Positions (N={analysis_results.num_sensors}):")
    print(analysis_results.sensors_positions)
    
    # Display the final summary
    print("\n" + "="*40)
    print("      ARRAY Z1 PERFORMANCE SUMMARY")
    print("="*40)
    print(analysis_results.performance_summary_table.to_markdown(index=False))
    
    print("\n--- Key Coarray Data ---")
    print(f"Unique Coarray Positions (Positive Lags): {analysis_results.coarray_positions[analysis_results.coarray_positions >= 0]}")
    print(f"Contiguous Segment Range (L1 to L2): {analysis_results.segment_ranges[0]}")
    print(f"Maximum Detectable Sources (K_max): {analysis_results.max_detectable_sources}")
    print(f"Expected w(1) = 0: Value is {analysis_results.performance_summary_table.iloc[7]['Value']}")

if __name__ == "__main__":
    main()

