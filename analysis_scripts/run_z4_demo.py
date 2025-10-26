# analysis_scripts/run_z4_demo.py

import os
import sys

# Ensure the project root is in the path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import the specific processor class
from geometry_processors.z4_processor import Z4ArrayProcessor

def main():
    """Initializes and runs the analysis for the Array Z4 (w(1)=w(2)=0)."""
    
    # --- Configuration ---
    N_sensors = 5  # Minimum N=5 required
    
    print(f"--- Starting Array Z4 (N={N_sensors}) Analysis Demo ---")
    
    # Instantiate the processor: Array Z4
    z4_processor = Z4ArrayProcessor(N=N_sensors, d=1.0)
    
    # Run the full analysis pipeline
    analysis_results = z4_processor.run_full_analysis()
    
    # Display the results
    print("\n" + "="*40)
    print(f"Physical Sensor Positions (N={analysis_results.num_sensors}):")
    print(analysis_results.sensors_positions)
    
    # Display the final summary
    print("\n" + "="*40)
    print("      ARRAY Z4 PERFORMANCE SUMMARY (Expected w(1)=w(2)=0)")
    print("="*40)
    print(analysis_results.performance_summary_table.to_markdown(index=False))
    
    print("\n--- Key Coarray Data ---")
    print(f"Contiguous Segment L1:L2 Range: {analysis_results.segment_ranges[0]}")
    print(f"Maximum Detectable Sources (K_max): {analysis_results.max_detectable_sources}")
    print(f"Weights w(1) and w(2) (Expected 0, 0): {analysis_results.performance_summary_table.iloc[7:9]['Value'].values}")

if __name__ == "__main__":
    main()