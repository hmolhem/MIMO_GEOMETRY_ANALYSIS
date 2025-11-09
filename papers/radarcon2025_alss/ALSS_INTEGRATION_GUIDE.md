# ALSS Paper Integration Guide

**Generated:** November 4, 2025  
**Purpose:** Quick-reference guide for integrating ALSS into RadarCon 2025 paper

---

## 📊 What You Have Now

### **3 Ready-to-Use Files**
1. **`alss_figures.tex`** - LaTeX/TikZ figure code (3 layouts: separate, combined, table)
2. **`alss_paper_section.tex`** - Complete method + results sections (drop-in ready)
3. **`plot_alss_figures.py`** - Python script to generate PDF/PNG figures

### **6 Benchmark CSV Files**
- `results/bench/alss_off_hard.csv` (baseline, Δ=13°)
- `results/bench/alss_on_hard.csv` (ALSS, Δ=13°)
- `results/bench/alss_off_sweep.csv` (baseline, Δ=2°)
- `results/bench/alss_on_sweep.csv` (ALSS, Δ=2°)
- `results/bench/alss_highM.csv` (M=256/512 sanity check)
- `results/bench/alss_ar1_tau07_coreL2.csv` (AR1 sensitivity)

---

## 🎯 The Key Numbers (Use These Exact Values)

### **Main Result (Moderate Difficulty, Δ=13°, M=64)**
```
SNR = 5 dB:  15.82° → 14.73°  (6.9% improvement)
SNR = 10 dB: 13.25° → 12.41°  (6.3% improvement)
SNR = 0 dB:  25.30° → 25.47°  (negligible, -0.7%)
```

### **Easy Case (Δ=2°, M=64)**
```
Baseline RMSE: ~0.91° across all SNR
ALSS RMSE:     ~0.91° (no change, harmless)
```

### **High M/SNR Sanity**
```
M=256, SNR=10dB: RMSE = 0.905° (both baseline and ALSS)
M=512, SNR=10dB: RMSE = 0.905° (no degradation)
```

---

## 📝 Two-Sentence Summary (Drop into Abstract/Intro)

> At finite snapshots and low SNR, coarray lag estimates at large |ℓ| are variance-dominated (small w[ℓ]), corrupting the virtual Toeplitz covariance used by CoarrayMUSIC. We introduce Adaptive Lag-Selective Shrinkage (ALSS)—a per-lag shrinker proportional to 1/(M·w[ℓ]) that protects core lags, enforces Hermitian symmetry, and reduces RMSE by 6.9% at moderate SNR without degrading high-performance regimes.

---

## 🖼️ Figures for Paper

### **Option 1: Two Separate Figures**
Use `\begin{figure}...\end{figure}` blocks from `alss_figures.tex`:
- **Figure N:** RMSE vs SNR (Δ=13°) - Shows 6.9% improvement
- **Figure N+1:** Resolve Rate vs SNR (Δ=2°) - Shows harmless behavior

### **Option 2: Combined Side-by-Side**
Use `\begin{figure*}...\end{figure*}` block (requires two-column format):
- Single figure with (a) and (b) subfigures
- More compact, better for short papers

### **Option 3: Data Table**
Use `\begin{table}...\end{table}` if you prefer tabular format over plots

### **To Generate PNG/PDF Versions**
```powershell
# Run the plotting script (requires matplotlib in mimo-geom-dev)
.\mimo-geom-dev\Scripts\Activate.ps1
python tools\plot_alss_figures.py

# Outputs to: papers/radarcon2025_alss/figures/
# - alss_rmse_vs_snr.pdf
# - alss_resolve_vs_snr.pdf  
# - alss_combined.pdf
```

---

## 📚 Paper Structure Recommendation

### **Section: Method (4.X Coarray Regularization)**
Copy from `alss_paper_section.tex` → "METHOD SECTION"
- Explains the 1/(M·w[ℓ]) shrinkage rule
- Describes zero-prior vs AR(1) modes
- Defines core-lag protection (L_core)

### **Section: Results (5.X ALSS Ablation)**
Copy from `alss_paper_section.tex` → "RESULTS SECTION"
- Reports 6.9% improvement at SNR=5dB
- Shows harmless behavior in easy case
- Validates safety at high M/SNR

### **Takeaway Line (use verbatim)**
> "ALSS is a safe, low-cost regularizer that yields modest accuracy gains (6--7%) when the baseline is struggling but not failing."

---

## 🔧 Recommended Parameter Defaults

### **Default Configuration (for paper)**
```bash
--alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3
```

### **Aggressive (very low SNR, optional)**
```bash
--alss on --alss-mode ar1 --alss-tau 1.5 --alss-coreL 2
```

**What Each Parameter Does:**
- `--alss-mode zero`: Shrink toward zero (conservative)
- `--alss-tau 1.0`: Moderate shrinkage strength
- `--alss-coreL 3`: Protect lags 0, 1, 2, 3 from shrinkage

---

## 🧪 Reproduction Commands (for Appendix/Supplementary)

```powershell
# Baseline (ALSS OFF)
python core\analysis_scripts\run_benchmarks.py `
  --arrays Z5 --N 7 --algs CoarrayMUSIC `
  --lambda_factor 2.0 --snr 0,5,10 --snapshots 64 `
  --k 2 --delta 13 --trials 200 `
  --alss off `
  --out results/bench/alss_off_hard.csv

# ALSS ON
python core\analysis_scripts\run_benchmarks.py `
  --arrays Z5 --N 7 --algs CoarrayMUSIC `
  --lambda_factor 2.0 --snr 0,5,10 --snapshots 64 `
  --k 2 --delta 13 --trials 200 `
  --alss on --alss-mode zero --alss-tau 1.0 --alss-coreL 3 `
  --out results/bench/alss_on_hard.csv

# Analysis
python tools\analyze_alss_results.py `
  results/bench/alss_off_hard.csv `
  results/bench/alss_on_hard.csv
```

---

## ✅ Integration Checklist

- [ ] Add `\usepackage{pgfplots}` and `\usepackage{subcaption}` to preamble
- [ ] Copy method section from `alss_paper_section.tex`
- [ ] Copy results section from `alss_paper_section.tex`
- [ ] Insert figures (choose Option 1, 2, or 3 above)
- [ ] Update figure references (`\ref{fig:alss_hard}`, etc.)
- [ ] Add two-sentence summary to abstract
- [ ] Cite key result: "6.9% improvement at SNR=5dB"
- [ ] Include reproduction commands in appendix (optional)

---

## 📦 Git Commit (When Ready)

```powershell
git add papers/radarcon2025_alss/
git add results/bench/*.csv
git add tools/analyze_alss_results.py
git add tools/check_baseline_samples.py
git add tools/check_highM.py
git add tools/plot_alss_figures.py

git commit -m "feat(alss): complete ablation study + paper-ready figures

Key results (Z5, N=7, M=64, 1200 trials):
- Moderate SNR (5-10dB): 6-7% RMSE reduction (15.82°→14.73°)
- Easy scenarios: No degradation (ALSS harmless)
- High M/SNR: No degradation (safe)

Deliverables:
- LaTeX figures (TikZ + table formats)
- Complete method + results sections
- Publication-quality plots (PDF/PNG)
- Reproduction scripts + analysis tools"
```

---

## 🎓 What to Claim in the Paper

**✅ SAFE CLAIMS:**
- "ALSS reduces RMSE by 6.9% at SNR=5dB (15.82°→14.73°)"
- "Provides 6--7% improvement at moderate SNR (5--10 dB)"
- "Does not degrade performance when baseline is already good"
- "Safe to enable by default (<1ms overhead)"
- "Most effective at moderate SNR with finite snapshots (M<128)"

**❌ AVOID THESE CLAIMS:**
- "ALSS rescues CoarrayMUSIC" (too strong, only 6% gain on average)
- "Dramatic improvement" (no, modest is correct)
- "Works at all SNR levels" (no, negligible at 0dB)
- "Solves the finite-snapshot problem" (no, mitigates it slightly)

**🎯 ACCURATE FRAMING:**
> "ALSS is a safe, low-cost regularizer that yields modest but consistent accuracy gains in moderate-difficulty regimes where the baseline is struggling but not failing."

---

## 📞 Contact for Questions

See `ALSS_ABLATION_RESULTS.md` for comprehensive experimental details, parameter sensitivity analysis, and additional context.

**Files:**
- `papers/radarcon2025_alss/alss_figures.tex` - LaTeX figure code
- `papers/radarcon2025_alss/alss_paper_section.tex` - Method + results text
- `papers/radarcon2025_alss/ALSS_ABLATION_RESULTS.md` - Full ablation report
- `papers/radarcon2025_alss/ALSS_INTEGRATION_GUIDE.md` - This file
