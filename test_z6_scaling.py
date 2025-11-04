from geometry_processors.z6_processor import Z6ArrayProcessor

for N in [5, 7, 9, 11, 13, 15]:
    z = Z6ArrayProcessor(N=N, d=1.0)
    z.run_full_analysis(verbose=False)
    Mv = len(z.data.largest_contiguous_segment) if z.data.largest_contiguous_segment is not None else 0
    K_max = Mv // 2
    positions = list(z.sensors_grid)
    print(f'N={N:2d}: Mv={Mv:2d}, K_max={K_max}, positions={positions}')
