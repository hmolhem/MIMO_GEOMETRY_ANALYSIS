# Paper 1 Conceptual Figure Plan

This document defines the proposed conceptual and paper-facing figures for the first clean IEEE-style paper:

```text
papers/radarcon2025_alss/ALSS_SCENARIO3_Z5_TRIAL1000_IEEE.tex
```

The purpose of this figure plan is to improve the explanatory quality of the paper without overclaiming the role of ALSS.

---

## 1. Scientific Caution

ALSS does not physically modify the antenna radiation pattern or the array geometry.

ALSS is a post-processing statistical regularization method applied after sample covariance estimation and coarray lag averaging, but before virtual covariance reconstruction and Coarray MUSIC processing.

Therefore, the paper should avoid captions or claims such as:

```text
ALSS improves the antenna radiation pattern.
ALSS corrects the physical antenna pattern.
ALSS changes the array factor.
```

More accurate wording:

```text
Mutual coupling distorts the effective array response.
ALSS improves the DOA-estimation behavior through coarray-domain regularization.
ALSS sharpens or stabilizes the MUSIC pseudospectrum under the tested conditions.
ALSS reduces finite-snapshot lag-estimation variance before virtual covariance reconstruction.
```

---

## 2. Recommended Figure Categories

The paper can benefit from three types of figures:

```text
1. Geometry-level figures
2. Array-response / mutual-coupling figures
3. Estimation-domain figures
```

The most important distinction is:

```text
Mutual coupling affects the effective array response.
ALSS affects the estimated covariance/coarray/MUSIC spectrum.
```

---

## 3. Proposed Figure Set

## Figure 1 — Z5 Sparse Array Geometry

### Purpose

Show the physical sensor layout of the Z5 sparse array used in the validated Scenario 3 trial-1000 experiment.

### Suggested content

```text
x-axis: sensor position in half-wavelength units
markers: active sensor locations
optional annotations: aperture, missing small spacings, w(1)=0 or w(1)=w(2)=0 if applicable
```

### Claim supported

```text
The tested geometry is sparse and weight-constrained, making it suitable for studying mutual-coupling-aware coarray processing.
```

### Status

```text
Recommended for Paper 1.
```

---

## Figure 2 — Z5 Coarray Weights

### Purpose

Show the difference-coarray weight distribution for Z5.

### Suggested content

```text
x-axis: coarray lag
y-axis: coarray weight w(l)
highlight low-weight or missing small lags
```

### Claim supported

```text
Z5 reduces small-lag sensor-pair contributions, but the resulting uneven coarray weights create finite-snapshot variance imbalance.
```

### Status

```text
Strongly recommended for Paper 1.
```

### Why this figure matters

This is probably the most important conceptual figure for explaining why ALSS is needed.

It connects:

```text
geometry design
coarray weights
finite-snapshot variance
ALSS motivation
```

---

## Figure 3 — Mutual Coupling Effect on Effective Array Response

### Purpose

Show how mutual coupling changes the effective array response.

### Suggested content

Compare:

```text
Ideal array response
Coupled array response
```

for a representative source direction or steering-vector scan.

### Important caution

This figure should not include ALSS as an antenna-pattern correction method.

Correct title:

```text
Effect of Mutual Coupling on Z5 Effective Array Response
```

Avoid title:

```text
Effect of ALSS on Antenna Pattern
```

### Claim supported

```text
Mutual coupling distorts the effective sensing response before DOA estimation.
```

### Status

```text
Optional but useful.
```

---

## Figure 4 — MUSIC Pseudospectrum Comparison

### Purpose

Show the estimation-domain effect of ALSS.

### Suggested content

For a representative Scenario 3 condition, compare:

```text
No coupling, no ALSS
Mutual coupling, no ALSS
Mutual coupling, with ALSS
```

### Preferred condition

A representative condition should be selected from the archived trial-1000 result, preferably one where ALSS clearly improves performance but does not look cherry-picked.

Candidate condition:

```text
Array: Z5
Coupling: c1=0.3
SNR: moderate-to-high
Snapshots: 64 or 128
ALSS: ar1/tau=0.25/coreL=3
```

### Claim supported

```text
ALSS improves the MUSIC pseudospectrum behavior under mutual coupling by regularizing noisy coarray lag estimates.
```

### Status

```text
Strongly recommended for Paper 1 if reproducible from code.
```

---

## Figure 5 — Lag-Domain ALSS Effect

### Purpose

Show where ALSS acts in the processing chain.

### Suggested content

Compare:

```text
raw coarray lag magnitudes
ALSS-regularized coarray lag magnitudes
lag weights w(l)
```

Possible combined layout:

```text
top: coarray weight w(l)
middle: raw lag estimate |r_hat(l)|
bottom: ALSS lag estimate |r_ALSS(l)|
```

### Claim supported

```text
ALSS primarily modifies lower-confidence non-core lags while preserving protected core lags.
```

### Status

```text
Recommended if implementation-level access to lag estimates is available.
```

---

## Figure 6 — Trial-1000 Improvement Versus SNR

### Existing archived figure

```text
results/figures/scenario3_trial1000/scenario3_improvement_vs_snr.png
```

### Purpose

Show the main quantitative result.

### Claim supported

```text
ALSS improves Z5 Scenario 3 RMSE in most tested SNR conditions, with stronger gains under mutual coupling.
```

### Status

```text
Already archived.
Primary paper figure.
```

---

## Figure 7 — RMSE Versus SNR Under Mutual Coupling

### Existing archived figure

```text
results/figures/scenario3_trial1000/scenario3_rmse_vs_snr_c1_0p3.png
```

### Purpose

Show direct baseline versus ALSS RMSE under the most important practical case.

### Claim supported

```text
Under mutual coupling, ALSS reduces RMSE relative to baseline Coarray MUSIC for most tested conditions.
```

### Status

```text
Already archived.
Primary paper figure.
```

---

## Figure 8 — Improvement Versus Snapshot Count

### Existing archived figure

```text
results/figures/scenario3_trial1000/scenario3_improvement_vs_snapshots.png
```

### Purpose

Show finite-snapshot behavior.

### Claim supported

```text
ALSS remains beneficial across the tested snapshot counts, especially under mutual coupling.
```

### Status

```text
Already archived.
Primary paper figure.
```

---

## 4. Recommended Paper-1 Figure Set

For a compact IEEE-style paper, the recommended first-draft figure set is:

```text
Figure 1: Z5 sensor geometry
Figure 2: Z5 coarray weights
Figure 3: MUSIC pseudospectrum comparison
Figure 4: Trial-1000 improvement versus SNR
Figure 5: RMSE versus SNR under mutual coupling
Figure 6: Improvement versus snapshots
```

If page length is tight, use:

```text
Figure 1: Z5 coarray weights
Figure 2: MUSIC pseudospectrum comparison
Figure 3: Trial-1000 improvement versus SNR
Figure 4: RMSE versus SNR under mutual coupling
```

---

## 5. Figures to Avoid in Paper 1

Avoid figures that imply ALSS modifies the physical antenna pattern.

Avoid:

```text
ALSS antenna pattern correction
ALSS radiation pattern improvement
ALSS beam pattern repair
```

Use instead:

```text
MUSIC pseudospectrum improvement
coarray lag regularization
virtual covariance stabilization
RMSE reduction
```

---

## 6. Suggested Next Implementation Steps

Recommended next branch:

```text
paper1-conceptual-figures
```

Recommended generation order:

```text
1. Generate Z5 geometry plot.
2. Generate Z5 coarray weight plot.
3. Generate one representative MUSIC pseudospectrum comparison.
4. Generate one lag-domain before/after ALSS plot if code access is clean.
5. Add figures to results/figures/paper1_conceptual/.
6. Add README.md documenting generation source and commands.
7. Integrate only the strongest figures into the paper draft.
```

---

## 7. Paper Wording Recommendation

Recommended wording for the paper:

```text
The beampattern-style array-response plot is used only to illustrate the effect of mutual coupling on the effective sensing response. ALSS does not modify the physical array response; instead, it regularizes the estimated coarray correlations used to form the virtual covariance matrix for Coarray MUSIC.
```

Recommended caption style:

```text
Effect of mutual coupling on the effective Z5 array response. This plot illustrates sensing-response distortion due to coupling and is separate from the ALSS post-processing stage.
```

For ALSS:

```text
Representative MUSIC pseudospectrum showing the effect of ALSS under mutual coupling. ALSS regularizes coarray lag estimates before virtual covariance reconstruction, improving peak stability in this example.
```

---

## 8. Current Status

```text
Status: planning document
Paper: Paper 1 / Z5 Scenario 3 trial-1000
Next step: generate conceptual figures in a separate branch
```