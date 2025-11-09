# analysis_scripts/plot_benchmarks.py
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def lineplot_rmse_vs_snr(df, figdir):
    os.makedirs(figdir, exist_ok=True)
    for K in sorted(df.K.unique()):
        for M in sorted(df.snapshots.unique()):
            sub = df[(df.K==K) & (df.snapshots==M)]
            if sub.empty: continue
            plt.figure()
            for arr in sorted(sub.array.unique()):
                s = sub[sub.array==arr].groupby("SNR_dB")["rmse_deg"].mean().reset_index()
                plt.plot(s["SNR_dB"], s["rmse_deg"], marker='o', label=arr)
            plt.xlabel("SNR (dB)")
            plt.ylabel("RMSE (deg)")
            plt.title(f"RMSE vs SNR (K={K}, snapshots={M})")
            plt.grid(True)
            plt.legend()
            path = os.path.join(figdir, f"rmse_vs_snr_K{K}_M{M}.png")
            plt.savefig(path, dpi=160, bbox_inches="tight")
            plt.close()
            print(f"[Saved] {path}")

def lineplot_resolve_vs_delta(df, figdir):
    os.makedirs(figdir, exist_ok=True)
    # Derive Δθ from true DOAs (for K=2 cases)
    sub = df[df.K==2].copy()
    if sub.empty:
        return
    def delta_from_str(s):
        a = np.array([float(x) for x in s.split(";")])
        return np.abs(a[1]-a[0]) if a.size==2 else np.nan
    sub["delta_deg"] = sub["doas_true"].apply(delta_from_str)

    for M in sorted(sub.snapshots.unique()):
        for SNR in sorted(sub.SNR_dB.unique()):
            ss = sub[(sub.snapshots==M) & (sub.SNR_dB==SNR)]
            if ss.empty: continue
            plt.figure()
            for arr in sorted(ss.array.unique()):
                g = ss[ss.array==arr].groupby("delta_deg")["resolved"].mean().reset_index()
                g = g.sort_values("delta_deg")
                plt.plot(g["delta_deg"], g["resolved"], marker='o', label=arr)
            plt.xlabel("Δθ (deg)")
            plt.ylabel("Pr{resolve}")
            plt.title(f"Resolution vs Δθ (SNR={SNR} dB, M={M})")
            plt.grid(True)
            plt.legend()
            path = os.path.join(figdir, f"resolve_vs_delta_SNR{SNR}_M{M}.png")
            plt.savefig(path, dpi=160, bbox_inches="tight")
            plt.close()
            print(f"[Saved] {path}")

def main():
    ap = argparse.ArgumentParser(description="Plot benchmark results")
    ap.add_argument("--in", dest="inp", type=str, required=True)
    ap.add_argument("--figdir", type=str, default="results/figs")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    lineplot_rmse_vs_snr(df, args.figdir)
    lineplot_resolve_vs_delta(df, args.figdir)

if __name__ == "__main__":
    main()
