# analysis_scripts/run_nested_demo.py

import os
import sys

# Add the project root (MIMO_GEOMETRY_ANALYSIS) to the path
# This allows the script to find the 'geometry_processors' module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Import the specific processor class
from geometry_processors.nested_processor import NestedArrayProcessor

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
    
    print("\n--- Key Coarray Data ---")
    print(f"Coarray Positions (Unique Lags): {analysis_results.coarray_positions}")
    print(f"Coarray Weights (w(l)): {analysis_results.coarray_weight_distribution}")
    print(f"Max Detectable Sources (K_max): {analysis_results.max_detectable_sources}")
    print(f"Coarray Aperture: {analysis_results.coarray_positions[-1]}")
    

if __name__ == "__main__":
    main()