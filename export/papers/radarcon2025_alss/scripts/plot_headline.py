# scripts/plot_headline.py
"""
Publication-ready plotting for MIMO array DOA benchmark results.

Generates:
- Fig 1: RMSE vs snapshots at SNR=10 dB, Δθ=2°
- Fig 2: Resolve rate vs SNR at M=256, Δθ=2°
- Fig 3: Heatmap RMSE for Z5 SpatialMUSIC (SNR×M) with CRB overlay
- Table: Headline summary at SNR=10, M=256, Δθ=2° (CSV for LaTeX)
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Input/output paths
IN = Path("results/bench/headline.csv")
CRB = Path("results/bench/crb_overlay.csv")
OUT_DIR = Path("results/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(IN)
crb = pd.read_csv(CRB) if CRB.exists() else pd.DataFrame()

def agg(df, by):
    """Aggregate benchmark results by grouping keys."""
    return df.groupby(by).agg(
        AvgRMSE=("rmse_deg", "mean"),
        Resolve=("resolved", "mean"),
        Mv=("Mv", "first")
    ).reset_index().assign(Resolve=lambda t: 100 * t.Resolve)

# ---------- Fig 1: RMSE vs M at SNR=10, Δ=2 ----------
print("[Fig 1] RMSE vs Snapshots at SNR=10 dB, Δθ=2°")
f1 = df.query("SNR_dB==10 & delta_deg==2")
g1 = agg(f1, ["array", "alg", "snapshots"])
plt.figure(figsize=(8, 6))
for (arr, alg), sub in g1.groupby(["array", "alg"]):
    label = f"{arr} {alg}"
    plt.plot(sub["snapshots"], sub["AvgRMSE"], marker="o", label=label, linewidth=2)
plt.xlabel("Snapshots (M)", fontsize=12)
plt.ylabel("RMSE (deg)", fontsize=12)
plt.title("RMSE vs Snapshots at SNR=10 dB, Δθ=2°", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc="best")
plt.tight_layout()
plt.savefig(OUT_DIR / "rmse_vs_M_SNR10_delta2.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'rmse_vs_M_SNR10_delta2.png'}")

# ---------- Fig 2: Resolve vs SNR at M=256, Δ=2 ----------
print("[Fig 2] Resolution vs SNR at M=256, Δθ=2°")
f2 = df.query("snapshots==256 & delta_deg==2")
g2 = agg(f2, ["array", "alg", "SNR_dB"])
plt.figure(figsize=(8, 6))
for (arr, alg), sub in g2.groupby(["array", "alg"]):
    label = f"{arr} {alg}"
    plt.plot(sub["SNR_dB"], sub["Resolve"], marker="s", label=label, linewidth=2)
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Resolved (%)", fontsize=12)
plt.title("Resolution vs SNR at M=256, Δθ=2°", fontsize=14, fontweight="bold")
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc="best")
plt.tight_layout()
plt.savefig(OUT_DIR / "resolve_vs_SNR_M256_delta2.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR / 'resolve_vs_SNR_M256_delta2.png'}")

# ---------- Fig 3: Heatmap Z5 SpatialMUSIC (SNR×M) ----------
print("[Fig 3] Heatmap Z5 SpatialMUSIC RMSE (SNR × M)")
f3 = df.query("array.str.startswith('Z5') & alg=='SpatialMUSIC'", engine="python")
if not f3.empty:
    piv = agg(f3, ["SNR_dB", "snapshots"]).pivot(index="SNR_dB", columns="snapshots", values="AvgRMSE")
    plt.figure(figsize=(10, 6))
    im = plt.imshow(piv.values, aspect="auto", origin="lower", cmap="hot",
                    extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
    plt.colorbar(im, label="RMSE (deg)")
    plt.xlabel("Snapshots (M)", fontsize=12)
    plt.ylabel("SNR (dB)", fontsize=12)
    plt.title("Z5 SpatialMUSIC RMSE (deg)", fontsize=14, fontweight="bold")
    
    # Overlay CRB contours (if available)
    if not crb.empty:
        c3 = crb.query("array.str.startswith('Z5') & alg=='SpatialMUSIC'", engine="python")
        if not c3.empty:
            # Average CRB per (SNR, M) to contour
            c3m = c3.groupby(["SNR_dB", "snapshots"]).agg(crb=("crb_deg", "mean")).reset_index()
            # Build grid matching pivot
            Z = np.full_like(piv.values, np.nan, dtype=float)
            for _, r in c3m.iterrows():
                i = np.where(piv.index.values == r["SNR_dB"])[0]
                j = np.where(piv.columns.values == r["snapshots"])[0]
                if len(i) == 1 and len(j) == 1:
                    Z[i[0], j[0]] = r["crb"]
            try:
                CS = plt.contour(piv.columns, piv.index, Z, colors="cyan", linewidths=1.5, levels=5)
                plt.clabel(CS, inline=True, fmt=lambda v: f"CRB={v:.03f}°", fontsize=8, colors="white")
            except Exception as e:
                print(f"  [WARN] CRB contour failed: {e}")
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "heatmap_Z5_spatial.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR / 'heatmap_Z5_spatial.png'}")
else:
    print("  [SKIP] No Z5 SpatialMUSIC data found")

# ---------- Headline Table (SNR=10, M=256, Δ=2) ----------
print("[Table] Headline summary at SNR=10 dB, M=256, Δθ=2°")
hl = df.query("SNR_dB==10 & snapshots==256 & delta_deg==2").copy()
hlg = agg(hl, ["array", "alg"])

# Attach CRB if present (match by array, alg, SNR, M, Δ)
if not crb.empty:
    m = crb.query("SNR_dB==10 & snapshots==256 & delta_deg==2")[["array", "alg", "crb_deg"]]
    hlt = hlg.merge(m, on=["array", "alg"], how="left")
else:
    hlt = hlg.copy()
    hlt["crb_deg"] = np.nan

# Compute RMSE/CRB ratio
hlt["RMSE/CRB"] = np.where(hlt["crb_deg"].gt(0), hlt["AvgRMSE"] / hlt["crb_deg"], np.nan)
hlt = hlt[["array", "alg", "AvgRMSE", "Resolve", "Mv", "crb_deg", "RMSE/CRB"]]

# Format for readability
hlt["AvgRMSE"] = hlt["AvgRMSE"].round(4)
hlt["Resolve"] = hlt["Resolve"].round(1)
hlt["crb_deg"] = hlt["crb_deg"].round(5)
hlt["RMSE/CRB"] = hlt["RMSE/CRB"].round(2)

hlt.to_csv(OUT_DIR / "headline_table_SNR10_M256_delta2.csv", index=False)
print(f"  Saved: {OUT_DIR / 'headline_table_SNR10_M256_delta2.csv'}")
print("\nTable Preview:")
print(hlt.to_string(index=False))

print(f"\n[Done] All figures/tables saved to {OUT_DIR}")
