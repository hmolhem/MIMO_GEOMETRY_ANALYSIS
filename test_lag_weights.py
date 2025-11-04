import numpy as np
from geometry_processors.z4_processor import Z4ArrayProcessor
from util.coarray import build_virtual_ula_covariance

z = Z4ArrayProcessor(N=7, d=1.0)
z.run_full_analysis(verbose=False)
positions_grid = np.asarray(z.data.sensors_positions, dtype=int)
print(f'Z4 positions (grid): {positions_grid}')

# Count pairs per lag
lag_counts = {}
for i in range(len(positions_grid)):
    for j in range(len(positions_grid)):
        lag = int(positions_grid[i] - positions_grid[j])
        lag_counts[lag] = lag_counts.get(lag, 0) + 1

print(f'\nLag weights (w[l]):')
for lag in sorted([l for l in lag_counts.keys() if l >= 0])[:15]:
    print(f'  w[{lag:2d}] = {lag_counts[lag]}')
