# MIMO_GEOMETRY_ANALYSIS Clean Rebuild Plan

## 1. Purpose

This document defines the clean rebuild strategy for the `MIMO_GEOMETRY_ANALYSIS` repository.

The goal is to transform the current repository into a clean, paper-ready, reproducible research project for:

- sparse-array geometry analysis
- coarray processing
- MUSIC-based DOA estimation
- mutual-coupling robustness
- Adaptive Lag-Selective Shrinkage (ALSS)
- future FPGA/HLS acceleration

This rebuild does not delete previous work blindly. The goal is to separate valuable research material from legacy, temporary, generated, or exploratory files.

---

## 2. Current Problem

The repository contains valuable work, but the structure has become mixed.

Current issues include:

- multiple paper drafts
- generated LaTeX auxiliary files
- exploratory Python scripts
- old figures and experiment outputs
- unclear relationship between scripts, figures, tables, and paper claims
- CI/linting applied to too many legacy files
- no complete paper-to-code map
- unclear distinction between maintained code and old experiments

This makes the repository harder to use for:

- academic review
- paper reproduction
- GitHub portfolio presentation
- resume support
- future FPGA/Embedded-AI development

---

## 3. Rebuild Strategy

The rebuild will be done incrementally on the `clean-rebuild` branch.

The existing repository remains the source of record. We will not start by deleting everything.

Instead, we will:

1. document the current state
2. identify the active paper version
3. identify the active experiment scripts
4. map paper claims to code and results
5. isolate legacy files
6. clean the README
7. define a reproducible experiment workflow
8. prepare the repository for future FPGA/HLS extension

If the clean rebuild becomes stable, it can later become the basis for a new polished public repository.

---

## 4. Target Repository Structure

The target long-term structure is:

```text
MIMO_GEOMETRY_ANALYSIS/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── docs/
│   ├── rebuild_plan.md
│   ├── paper_to_code_map.md
│   ├── reproduction_guide.md
│   ├── experiment_summary.md
│   └── Technical_Proposal_Roadmap.tex/pdf
├── src/
│   └── mimo_geometry/
│       ├── arrays/
│       ├── coarray/
│       ├── doa/
│       ├── mutual_coupling/
│       └── alss/
├── tests/
├── experiments/
│   ├── run_snr_sweep.py
│   ├── run_snapshot_sweep.py
│   ├── run_coupling_sweep.py
│   └── reproduce_paper_results.py
├── papers/
│   └── radarcon2025_alss/
├── results/
│   ├── tables/
│   └── figures/
└── archive/
    └── legacy_scripts/
