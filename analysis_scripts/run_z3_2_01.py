# analysis_scripts/run_z3_2_demo.py

import os
import sys
# Ensure the project root is in the path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import the specific processor class
from geometry_processors.z3_2_processor import Z3_2ArrayProcessor

def main():
    """Initializes and runs the analysis for the Array Z3(2)."""
    
    # --- Configuration ---
    N_sensors = 6  # Minimum N=5 required
    
    print(f"--- Starting Array Z3(2) (N={N_sensors}) Analysis Demo ---")
    
    # Instantiate the processor: Array Z3(2)
    z3_2_processor = Z3_2ArrayProcessor(N=N_sensors, d=1)
    
    # Run the full analysis pipeline
    analysis_results = z3_2_processor.run_full_analysis()
    
    # Display the results
    print("\n" + "="*40)
    print(f"Physical Sensor Positions (N={analysis_results.num_sensors}):")
    print(analysis_results.sensors_positions)
    
    # Display the final summary
    print("\n" + "="*40)
    print("      ARRAY Z3(2) PERFORMANCE SUMMARY (Expected w(2)=1)")
    print("="*40)
    print(analysis_results.performance_summary_table.to_markdown(index=False))
    
    print("\n--- Key Coarray Data ---")
    print(f"Contiguous Segment L1:L2 Range: {analysis_results.segment_ranges[0]}")
    print(f"Maximum Detectable Sources (K_max): {analysis_results.max_detectable_sources}")
    print(f"Weights w(1), w(2), w(3): {analysis_results.performance_summary_table.iloc[7:10]['Value'].values}")

if __name__ == "__main__":
    main()