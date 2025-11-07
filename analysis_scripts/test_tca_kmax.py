"""Quick test TCA K_max extraction."""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_processors.tca_processor import TCAArrayProcessor

processor = TCAArrayProcessor(M=3, N=4)
array_data = processor.run_full_analysis()

K_max = None
if hasattr(array_data, 'performance_summary_table') and array_data.performance_summary_table is not None:
    perf = array_data.performance_summary_table
    print(f"Table columns: {list(perf.columns)}")
    
    if 'Metrics' in perf.columns:
        print("Branch: Metrics column")
        k_row = perf[perf['Metrics'].str.contains('K_max', case=False, na=False)]
        if not k_row.empty:
            K_max = int(k_row['Value'].iloc[0])
    elif 'K_max' in perf.columns:
        print("Branch: K_max direct column")
        K_max = int(perf['K_max'].iloc[0])
        print(f"Extracted K_max={K_max}")
    else:
        print("Branch: First column search")

print(f"Final K_max: {K_max}")
