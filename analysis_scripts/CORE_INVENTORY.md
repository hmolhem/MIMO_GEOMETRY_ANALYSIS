# Inventory and Recommended Canonical Mapping for core/analysis_scripts

This file lists the primary scripts currently present in `core/analysis_scripts/` along with duplicates found at the repository root `analysis_scripts/`. For each script I recommend a canonical home (core, demos, tools, or package) and the migration action.

---

## Benchmarks & Scenario Runners (CANONICAL: `core/analysis_scripts/` → package `mimo_geom_analysis.runners`)

- run_benchmarks.py
  - Duplicates: `analysis_scripts/run_benchmarks.py` (root)
  - Action: Keep canonical logic in `core/analysis_scripts/run_benchmarks.py`. Move callable logic into `src/mimo_geom_analysis/runners.py` and make `core` script a thin CLI wrapper. Replace root duplicate with a deprecation wrapper.

- run_paper_experiments.py
  - Action: Canonical in core. Extract to package `mimo_geom_analysis.papers.run_paper_experiments` and provide wrapper.

- run_paper_experiments.py

- run_scenario1_baseline.py
- run_scenario2_coupling_impact.py
- run_scenario3_alss_regularization.py
- run_scenario4_array_comparison.py
- run_scenario5_coupling_models.py
  - Duplicates: some `analysis_scripts/` counterparts exist (search results show multiple copies)
  - Action: Keep canonical in core and extract common code to `src/mimo_geom_analysis/scenarios`.

- run_cross_scenario_analysis.py
  - Action: canonical in core; implement as high-level orchestrator in package.

## Demos & Interactive (CANONICAL: `analysis_scripts/demos/` wrappers → package `mimo_geom_analysis.demos`)

- run_ula_demo.py (core + root)
- run_nested_demo.py (core + root)
- run_z1_demo.py, run_z3_1_demo.py, run_z3_2_demo.py, run_z4_demo.py, run_z5_demo.py, run_z6_demo.py (core + root)
- graphical_demo.py (core + root)
- methods_demo.py (core + root)
  - Action: Consolidate user-facing demos under `analysis_scripts/demos/` as thin wrappers. Move implementation into `src/mimo_geom_analysis/demos` for importable functions. Leave root wrappers that call canonical demo paths (deprecation messages).

## Plotting & Postprocessing Tools (CANONICAL: `analysis_scripts/tools/` or `src/mimo_geom_analysis.plot`)

- plot_benchmarks.py (core + root)
- plot_paper_benchmarks.py (root)
- visualize_mcm_effects.py
- plot_coarray utilities (internal functions)
  - Action: Move reusable plotting functions to `src/mimo_geom_analysis/plot.py` and keep small CLI wrappers in `analysis_scripts/tools/` or `analysis_scripts/core/`.

## Utilities / One-offs (CANONICAL: `src/mimo_geom_analysis.tools`)

- generate_paper_data.py, generate_paper_data_final.py
- extract_wcsa_pdf.py (root)
- validate_music.py, debug_music.py, debug_kmax.py
- test_* scripts (ad-hoc)
  - Action: Consolidate long-lived utilities into `src/mimo_geom_analysis.tools`. Archive or document one-off scripts in `analysis_scripts/legacy/` if they are not needed.

## Duplicates and underscore variants

Files with trailing underscores (e.g., `run_z4_demo_.py`) are often experimental or legacy copies.
- Action: Treat the version in `core/analysis_scripts/` as authoritative when both exist. Keep the underscore variant for inspection and remove or archive after validation.

## Tests

- There is a `test/` folder with `test.py`. Action: Add a `tests/` directory with pytest-compatible unit tests focusing on:
  - Processor smoke tests (instantiate processors and run `.run_full_analysis()`)
  - Runner smoke tests (`--help`, and one minimal run with `--trials 1`)

## Packaging / CLI

Suggested console scripts (pyproject `project.scripts`):
- `mimo-bench = mimo_geom_analysis.runners:main`
- `mimo-demo-ula = mimo_geom_analysis.demos:run_ula`
- `mimo-graphical-demo = mimo_geom_analysis.demos:graphical_main`

These allow `pip install .` then run `mimo-bench` from the command line.

## Migration priority (safe, incremental)
1. Inventory + mapping (this document)
2. Add package modules for `runners`, `scenarios`, `demos`, and `plot` under `src/mimo_geom_analysis` and add tests
3. Replace heavy logic in `core` scripts with thin wrappers that call package functions
4. Replace root-level duplicate scripts with thin backward-compatible wrappers (print deprecation) and update docs
5. Add CI to run smoke tests and pytest
6. Remove archived duplicates after at least one release cycle

---

If you approve this mapping, I'll proceed by moving one canonical script (recommended: `run_benchmarks.py`) into `src/mimo_geom_analysis/runners.py`, refactor its imports to package-based imports, add a minimal test, and update the wrapper to call the package function. This will serve as a template for the rest of the migration.
