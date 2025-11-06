#!/usr/bin/env python3
"""Quick check: show sample DOA estimates from baseline."""
import pandas as pd

df = pd.read_csv('results/bench/alss_off_sweep.csv')
snr5 = df[df['SNR_dB'] == 5.0]

print("\n" + "="*70)
print("SAMPLE DOA ESTIMATES (SNR=5dB, BASELINE)")
print("="*70)
print(f"\nTotal trials: {len(snr5)}")
print(f"Resolved: {snr5['resolved'].sum()} ({snr5['resolved'].sum()/len(snr5)*100:.1f}%)")
print(f"Mean RMSE: {snr5['rmse_deg'].mean():.3f}°\n")

# Show first 20 examples
print("First 20 trials:")
print("-" * 70)
for idx, row in snr5.head(20).iterrows():
    true_doas = row['doas_true']
    est_doas = row['doas_est']
    rmse = row['rmse_deg']
    resolved = '✓' if row['resolved'] else '✗'
    print(f"{resolved} True: {true_doas:20s} → Est: {est_doas:20s} | RMSE: {rmse:6.2f}°")

print("\n" + "="*70)
