import pandas as pd
import numpy as np

# Load data
df_off = pd.read_csv('results/bench/alss_off_hard.csv')
df_on = pd.read_csv('results/bench/alss_on_hard.csv')

print('\n=== ALSS PERFORMANCE (Delta=13°, Hard Case) ===\n')
print('Configuration: Z5(N=7), M=64 snapshots, K=2 sources, 200 trials per SNR\n')

# Per-SNR comparison
for snr in [0, 5, 10]:
    off_rmse = df_off[df_off['SNR_dB'] == snr]['rmse_deg'].mean()
    on_rmse = df_on[df_on['SNR_dB'] == snr]['rmse_deg'].mean()
    improvement = (on_rmse - off_rmse) / off_rmse * 100
    
    off_resolve = df_off[df_off['SNR_dB'] == snr]['resolved'].mean() * 100
    on_resolve = df_on[df_on['SNR_dB'] == snr]['resolved'].mean() * 100
    
    print(f'SNR = {snr:2d} dB:')
    print(f'  RMSE:    Baseline={off_rmse:.3f}°, ALSS={on_rmse:.3f}° ({improvement:+.1f}%)')
    print(f'  Resolve: Baseline={off_resolve:.1f}%, ALSS={on_resolve:.1f}%')
    print()

# Overall
off_overall = df_off['rmse_deg'].mean()
on_overall = df_on['rmse_deg'].mean()
overall_imp = (on_overall - off_overall) / off_overall * 100

print(f'Overall Average:')
print(f'  Baseline RMSE: {off_overall:.3f}°')
print(f'  ALSS RMSE:     {on_overall:.3f}° ({overall_imp:+.1f}%)')
print()
