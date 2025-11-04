import numpy as np
from geometry_processors.z6_processor import Z6ArrayProcessor

z = Z6ArrayProcessor(N=7, d=1.0)
z.run_full_analysis(verbose=False)
positions_grid = np.asarray(z.data.sensors_positions, dtype=int)
print(f'Z6 positions (grid): {positions_grid}')

# Compute all positive lags
lags = set()
for i in range(len(positions_grid)):
    for j in range(len(positions_grid)):
        lag = int(positions_grid[i] - positions_grid[j])
        if lag >= 0:
            lags.add(lag)

sorted_lags = sorted(lags)
print(f'Positive lags: {sorted_lags}')

# Find ALL contiguous segments
segments = []
current_start = sorted_lags[0]
current_length = 1

for i in range(1, len(sorted_lags)):
    if sorted_lags[i] == sorted_lags[i-1] + 1:
        current_length += 1
    else:
        segments.append((current_start, current_length))
        current_start = sorted_lags[i]
        current_length = 1

segments.append((current_start, current_length))

print(f'\nAll contiguous segments:')
for start, length in segments:
    seg = list(range(start, start + length))
    print(f'  Start={start}, Length={length}: {seg}')

# Find longest contiguous
best_start, best_length = max(segments, key=lambda x: x[1])
print(f'\nLongest contiguous: start={best_start}, length={best_length}')
print(f'Segment: {list(range(best_start, best_start + best_length))}')
