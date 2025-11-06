# analysis_scripts/run_ula_demo.py

import os
import sys
# Add the project root to the path to enable absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the specific processor class
from core.radarpy.geometry.ula_processors import ULArrayProcessor

def main():
    """Initializes and runs the analysis for a ULA."""
    
    print("--- Starting ULA (N=4) Analysis Demo ---")
    
    # Instantiate the processor: ULA with N=4 sensors, spacing d=1.0
    ula_processor = ULArrayProcessor(M=4, d=1)
    
    # Run the full analysis pipeline
    analysis_results = ula_processor.run_full_analysis()
    
    # Display the final summary
    print("\n" + "="*40)
    print("      FINAL PERFORMANCE SUMMARY")
    print("="*40)
    print(analysis_results.performance_summary_table.to_markdown(index=False))
    
    # (Optional: Code to save plots or summary table to the 'results/' folder would go here)
    print(ula_processor.data.all_differences_table)

if __name__ == "__main__":
    main()

