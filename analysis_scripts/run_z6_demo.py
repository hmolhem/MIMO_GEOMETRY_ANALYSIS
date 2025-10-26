# analysis_scripts/run_z6_demo.py

import os
import sys

# Ensure the project root is in the path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import the specific processor class
from geometry_processors.z6_processor import Z6ArrayProcessor

def main():
    """Initializes and runs the analysis for the Array Z6 (w(1)=w(2)=0)."""
    
    # --- Configuration ---
    N_sensors = 10  # Minimum N=5 required [cite: 285]
    
    print(f"--- Starting Array Z6 (N={N_sensors}) Analysis Demo ---")
    
    # Instantiate the processor: Array Z6
    z6_processor = Z6ArrayProcessor(N=N_sensors, d=1.0)
    
    # Run the full analysis pipeline
    analysis_results = z6_processor.run_full_analysis()
    
    # Display the results
    print("\n" + "="*40)
    print(f"Physical Sensor Positions (N={analysis_results.num_sensors}):")
    print(analysis_results.sensors_positions)
    
    # Display the final summary
    print("\n" + "="*40)
    print("      ARRAY Z6 PERFORMANCE SUMMARY (Expected w(1)=w(2)=0)")
    print("="*40)
    print(analysis_results.performance_summary_table.to_markdown(index=False))
    
    print("\n--- Key Coarray Data ---")
    print(f"Contiguous Segment L1:L2 Range: {analysis_results.segment_ranges[0]}")
    print(f"Maximum Detectable Sources (K_max): {analysis_results.max_detectable_sources}")
    # Display w(1), w(2), w(3)
    weights = analysis_results.performance_summary_table.iloc[7:10]['Value'].values
    print(f"Weights w(1), w(2), w(3) (Expected 0, 0, 3): {weights}")

if __name__ == "__main__":
    main()