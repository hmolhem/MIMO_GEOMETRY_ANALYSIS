import numpy as np
from geometry_processors.z4_processor import Z4ArrayProcessor
from util.coarray import build_virtual_ula_covariance

z = Z4ArrayProcessor(N=7, d=1.0)
z.run_full_analysis(verbose=False)
positions_grid = np.asarray(z.data.sensors_positions, dtype=int)
print(f'Z4 positions (grid): {positions_grid}')

# Simulate a dummy covariance
N = len(positions_grid)
Rxx = np.eye(N, dtype=complex)
Rv, dvirt, (L1, L2), one_side, rmap = build_virtual_ula_covariance(Rxx, positions_grid, 1.0)
print(f'Lv = {Rv.shape[0]}, L1={L1}, L2={L2}')
print(f'one_side lags: {one_side}')
