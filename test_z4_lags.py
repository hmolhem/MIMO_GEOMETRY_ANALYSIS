import numpy as np
from geometry_processors.z4_processor import Z4ArrayProcessor

z = Z4ArrayProcessor(N=7, d=1.0)
z.run_full_analysis(verbose=False)
positions_grid = np.asarray(z.data.sensors_positions, dtype=int)
print(f'Z4 positions (grid): {positions_grid}')

# Compute all lags manually
lags = set()
for i in range(len(positions_grid)):
    for j in range(len(positions_grid)):
        lag = int(positions_grid[i] - positions_grid[j])
        lags.add(lag)

sorted_lags = sorted(lags)
print(f'All lags: {sorted_lags}')
print(f'Positive lags: {[l for l in sorted_lags if l >= 0]}')
