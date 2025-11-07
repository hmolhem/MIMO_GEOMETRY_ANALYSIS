"""Debug K_max extraction."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_processors.tca_processor import TCAArrayProcessor

p = TCAArrayProcessor(M=3, N=4)
r = p.run_full_analysis()
perf = r.performance_summary_table

print("Columns:", list(perf.columns))
print("Has 'Metrics':", 'Metrics' in perf.columns)
print("Has 'K_max':", 'K_max' in perf.columns)

K_max = None
if 'Metrics' in perf.columns:
    print("Branch: Metrics column")
elif 'K_max' in perf.columns:
    print("Branch: K_max direct column")
    K_max = int(perf['K_max'].iloc[0])
    
print("Final K_max:", K_max)
