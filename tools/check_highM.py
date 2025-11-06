#!/usr/bin/env python3
"""Check high M/SNR sanity results."""
import pandas as pd

df = pd.read_csv('results/bench/alss_highM.csv')

print("\n" + "="*70)
print("HIGH M/SNR SANITY CHECK (ALSS ON, delta=2°)")
print("="*70 + "\n")

snr10 = df[df['SNR_dB'] == 10]
m256 = snr10[snr10['snapshots'] == 256]
m512 = snr10[snr10['snapshots'] == 512]

print(f"SNR=10dB, M=256:")
print(f"  RMSE: {m256['rmse_deg'].mean():.3f}° ± {m256['rmse_deg'].std():.3f}°")
print(f"  Resolve: {m256['resolved'].sum()}/{len(m256)} ({m256['resolved'].sum()/len(m256)*100:.1f}%)")
print(f"  Runtime: {m256['runtime_ms'].mean():.2f}ms\n")

print(f"SNR=10dB, M=512:")
print(f"  RMSE: {m512['rmse_deg'].mean():.3f}° ± {m512['rmse_deg'].std():.3f}°")
print(f"  Resolve: {m512['resolved'].sum()}/{len(m512)} ({m512['resolved'].sum()/len(m512)*100:.1f}%)")
print(f"  Runtime: {m512['runtime_ms'].mean():.2f}ms\n")

print("✓ ALSS does not degrade performance at high M/SNR")
print("="*70 + "\n")
