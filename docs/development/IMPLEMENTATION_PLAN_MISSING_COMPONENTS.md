# Implementation Plan: Missing Paper Components

**Date**: November 7, 2025  
**Goal**: Add missing arrays and experiments from Kulkarni & Vaidyanathan (2024) paper  
**Strategy**: Modular additions - NO changes to existing working code

---

## Phase 1: Missing Array Implementations (NEW FILES ONLY)

### Priority: HIGH (needed for paper replication)

All new array processors will be added as **separate files** in `geometry_processors/`:

1. **SNA3 (Super Nested Array, 3rd order)** ✓ Planned
   - File: `geometry_processors/sna_processor.py`
   - Reference: Liu & Vaidyanathan [9]
   - Construction: Expands nested array to reduce w(2), w(3)
   - Parameters: N1, N2 (like nested array)

2. **ANAII-2 (Augmented Nested Array variant II-2)** ✓ Planned
   - File: `geometry_processors/ana_processor.py`
   - Reference: Liu et al. [11]
   - Construction: Redistributes dense ULA sensors around sparse subarray
   - Parameters: N1, N2

3. **DNA/DDNA (Dilated/Displaced Dilated Nested)** ✓ Planned
   - File: `geometry_processors/dna_processor.py`
   - Reference: Shaalan et al. [12]
   - Construction: Dilates nested array by factor, adds displacement
   - Parameters: N1, N2, dilation_factor

4. **MISC (Maximum Inter-element Spacing Constraint)** ✓ Planned
   - File: `geometry_processors/misc_processor.py`
   - Reference: Zheng et al. [13]
   - Construction: Pattern from [3] with max spacing constraint
   - Parameters: N (total sensors)

5. **TCA (Thinned Coprime Array)** ✓ Planned
   - File: `geometry_processors/tca_processor.py`
   - Reference: Raza et al. [14]
   - Construction: Removes sensors from coprime array to reduce coupling
   - Parameters: M, N (coprime integers)

6. **ePCA (Enhanced Padded Coprime Array)** ✓ Planned
   - File: `geometry_processors/epca_processor.py`
   - Reference: Zheng et al. [15]
   - Construction: Adds sensors to coprime array to fill holes
   - Parameters: M, N (coprime)

7. **CADiS (Compressed & Displaced Coprime)** ✓ Planned
   - File: `geometry_processors/cadis_processor.py`
   - Reference: Qin et al. [7]
   - Construction: Compresses and displaces coprime subarray
   - Parameters: M, N, p (compression factor)

8. **cMRA (Constrained Minimum Redundancy Array)** ✓ Planned
   - File: `geometry_processors/cmra_processor.py`
   - Reference: Ishiguro [3], tabulated for N≤20
   - Construction: Lookup table for known optimal arrays
   - Parameters: N (only works for N≤20)

---

## Phase 2: Missing Experiments (NEW SCRIPTS ONLY)

All new experiments will be in `core/analysis_scripts/paper_validation/`:

1. **Cross-Array Comparison (Fig. 1, 3, 4)** ✓ Planned
   - File: `core/analysis_scripts/paper_validation/run_fig1_coarray_weights.py`
   - Purpose: Plot w(0) to w(40) for all 16 arrays at N=16
   - Output: `results/paper_validation/fig1_coarray_weights.png`

2. **MUSIC Spectra for D=22 sources (Fig. 2)** ✓ Planned
   - File: `core/analysis_scripts/paper_validation/run_fig2_music_spectra.py`
   - Purpose: Show peak identification for each array
   - Output: `results/paper_validation/fig2_music_spectra.png`

3. **MSE vs SNR/Snapshots/Coupling (Fig. 3, 4)** ✓ Planned
   - File: `core/analysis_scripts/paper_validation/run_fig3_mse_D6.py`
   - File: `core/analysis_scripts/paper_validation/run_fig4_mse_D20.py`
   - Purpose: Compare arrays at D=6 and D=20 sources
   - Output: 6 subplots (3 per figure)

4. **Aperture-Constrained Study (Table V, Fig. 5, 6)** ✓ Planned
   - File: `core/analysis_scripts/paper_validation/run_aperture_constraint.py`
   - Purpose: Fix A≤100, vary N for each array, test D=6 and D=20
   - Output: `results/paper_validation/table_v_aperture.csv` + plots

5. **Source Number Sweep (Fig. 7)** ✓ Planned
   - File: `core/analysis_scripts/paper_validation/run_fig7_source_sweep.py`
   - Purpose: Vary D from 6 to 80 with N=25 sensors
   - Output: `results/paper_validation/fig7_source_sweep.png`

---

## Phase 3: Integration & Validation

1. **Update `__init__.py`** ✓ Planned
   - Add imports for new array processors
   - Keep existing imports unchanged

2. **Create Test Suite** ✓ Planned
   - File: `core/tests/test_new_arrays.py`
   - Verify each new array matches paper properties
   - Check w(1), w(2), aperture, DOFs

3. **Documentation** ✓ Planned
   - File: `PAPER_VALIDATION_GUIDE.md`
   - Explain how to run each validation script
   - Map scripts to paper figures/tables

---

## Safety Guarantees

### ✅ What Will NOT Be Changed:

1. **Existing array processors** - No modifications to:
   - `ula_processors.py`, `nested_processor.py`
   - `z1_processor.py` through `z6_processor.py`
   - `bases_classes.py` (framework)

2. **Existing experiments** - No modifications to:
   - `run_scenario1_baseline.py`
   - `run_scenario2_coupling_impact.py`
   - `run_scenario3_alss_regularization.py`
   - `run_scenario4_array_comparison.py`

3. **Existing results** - All preserved in:
   - `results/scenario1_baseline/`
   - `results/scenario2_coupling/`
   - `results/scenario3_alss_corrected/`
   - `results/scenario4_arrays/`

### ✅ What Will Be Added (New Files Only):

```
geometry_processors/
├── sna_processor.py          [NEW]
├── ana_processor.py          [NEW]
├── dna_processor.py          [NEW]
├── misc_processor.py         [NEW]
├── tca_processor.py          [NEW]
├── epca_processor.py         [NEW]
├── cadis_processor.py        [NEW]
├── cmra_processor.py         [NEW]
└── (existing files unchanged)

core/analysis_scripts/paper_validation/    [NEW DIRECTORY]
├── run_fig1_coarray_weights.py
├── run_fig2_music_spectra.py
├── run_fig3_mse_D6.py
├── run_fig4_mse_D20.py
├── run_aperture_constraint.py
├── run_fig7_source_sweep.py
└── README.md

results/paper_validation/     [NEW DIRECTORY]
├── fig1_coarray_weights.png
├── fig2_music_spectra.png
├── fig3_mse_D6.png
├── fig4_mse_D20.png
├── fig5_aperture_D6.png
├── fig6_aperture_D20.png
├── fig7_source_sweep.png
├── table_v_aperture.csv
└── (CSV data files)

core/tests/
├── test_new_arrays.py        [NEW]
└── (existing tests unchanged)
```

---

## Implementation Order (Recommended)

### Week 1: Foundation Arrays
1. ✅ SNA3 (simplest - similar to nested)
2. ✅ ANAII-2 (similar to nested)
3. ✅ Test both with `test_new_arrays.py`

### Week 2: Coprime Variants
4. ✅ TCA (thinned coprime)
5. ✅ ePCA (padded coprime)
6. ✅ CADiS (compressed coprime)
7. ✅ Test all three

### Week 3: Special Arrays
8. ✅ DNA/DDNA (dilated nested)
9. ✅ MISC (pattern-based)
10. ✅ cMRA (lookup table, N≤20 only)

### Week 4: Paper Validation Experiments
11. ✅ Figure 1 (coarray weights)
12. ✅ Figures 3-4 (MSE comparisons)
13. ✅ Figures 5-6 (aperture constraint)
14. ✅ Figure 7 (source sweep)

---

## Testing Strategy

For each new array:

```python
def test_new_array():
    """Template for testing new arrays"""
    # 1. Create processor
    proc = NewArrayProcessor(params)
    results = proc.run_full_analysis()
    
    # 2. Verify basic properties
    assert results.num_sensors == expected_N
    assert results.coarray_aperture > 0
    
    # 3. Verify weight constraints (if applicable)
    wt = {int(r["Lag"]): int(r["Weight"]) 
          for _, r in results.weight_table.iterrows()}
    assert wt.get(1, 0) >= 0  # Check w(1)
    
    # 4. Compare with paper Table IV (if N=16)
    if expected_N == 16:
        assert results.coarray_aperture == paper_value_A
        assert results.max_detectable_sources == paper_value_Dm
    
    print(f"✅ {proc.name} validated!")
```

---

## References from Paper

Kulkarni & Vaidyanathan (2024) references we'll implement:

- [3] M. Ishiguro, "Minimum redundancy linear arrays for a large number of antennas"
- [7] S. Qin et al., "Generalized coprime array configurations (CADiS)"
- [9] C.-L. Liu & P.P. Vaidyanathan, "Super nested arrays" (SNA)
- [11] J. Liu et al., "Augmented nested arrays" (ANA)
- [12] A.M. Shaalan et al., "Dilated nested arrays" (DNA/DDNA)
- [13] Z. Zheng et al., "MISC array"
- [14] A. Raza et al., "Thinned coprime array" (TCA)
- [15] W. Zheng et al., "Padded coprime arrays" (ePCA)

---

## Expected Outcomes

After completion, you will have:

1. ✅ **All 16 arrays from paper** (your 9 + new 8)
2. ✅ **All paper figures reproducible** (Fig. 1-7)
3. ✅ **Table V aperture constraint data**
4. ✅ **Validation suite** ensuring correctness
5. ✅ **Your ALSS research** (untouched, still works)
6. ✅ **Publication-ready comparison** with paper

**Estimated Total Time**: 4 weeks (working ~10 hours/week)

**Estimated Lines of Code**: 
- 8 new array processors: ~200 lines each = 1,600 lines
- 6 validation scripts: ~300 lines each = 1,800 lines
- Test suite: ~400 lines
- **Total**: ~3,800 new lines (all additions, zero modifications)

---

## How to Use This Plan

**Option A: Implement yourself**
- Follow implementation order above
- Use existing processors as templates
- Run tests after each addition

**Option B: I implement for you**
- I'll create one array/script at a time
- You review and test each before moving to next
- Safe, incremental progress

**Option C: Hybrid**
- I create templates/skeletons
- You fill in array-specific construction logic
- We validate together

---

**Ready to start?** I recommend beginning with **SNA3** (Super Nested Array) since it's:
- Simplest (very similar to your existing nested processor)
- Well-documented in paper
- Good test of the framework

Would you like me to implement SNA3 as the first example?
