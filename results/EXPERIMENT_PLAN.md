# New Experimental Design Plan

**Date:** November 6, 2025  
**Status:** Environment Prepared - Ready for New Experiments

---

## Environment Status

✅ **Cleanup Complete:**
- 97 old CSV datasets moved to `garbage/old_experiments_20251106_191349`
- 23 old PNG figures moved to garbage folder
- Results directory clean and ready
- All old data safely backed up

✅ **Software Ready:**
- Comprehensive MIMO geometry analysis framework operational
- MCM (Mutual Coupling Matrix) feature integrated
- Automated benchmark runners configured
- Virtual environment (mimo-geom-dev) activated

---

## Experimental Capabilities Available

### 1. **Array Geometries**
- ULA (Uniform Linear Array)
- Nested Arrays
- Z1-Z6 Specialized Arrays (optimized coarray)

### 2. **DOA Estimation Algorithms**
- Spatial MUSIC (conventional)
- Coarray MUSIC (virtual array)
- ALSS (Adaptive Lag-Selective Shrinkage)

### 3. **Mutual Coupling Scenarios**
- **Ideal (Baseline):** `coupling_matrix=None`
- **Exponential Model:** Distance-based coupling decay
- **Toeplitz Model:** Symmetric coupling for ULA
- **Measured Data:** Load coupling from CSV/NPY files

### 4. **Experimental Parameters**
- **Array Size (N):** Number of physical sensors
- **Snapshots (M):** Number of time samples
- **SNR:** Signal-to-noise ratio sweep
- **DOA Separation (Δθ):** Angular spacing between sources
- **Coupling Strength:** c₁ parameter (0.1 to 0.5)
- **Decay Factor:** α parameter (0.3 to 1.0)

---

## Proposed Experimental Scenarios

### **Scenario 1: Baseline Performance Characterization**

**Objective:** Establish baseline DOA estimation performance without coupling

**Configuration:**
```python
arrays = ["ULA", "Nested", "Z5"]
N = 7, 10, 15
SNR = [0, 5, 10, 15, 20] dB
DOA_separation = [2, 5, 10, 15, 20, 30, 45] degrees
snapshots = [64, 128, 256]
coupling_matrix = None  # Ideal array
```

**Expected Outputs:**
- RMSE vs. SNR curves
- Resolution probability vs. Δθ
- Comparative array performance tables

---

### **Scenario 2: Mutual Coupling Impact Study**

**Objective:** Quantify performance degradation due to mutual coupling

**Configuration:**
```python
array = "Z5"
N = 7
SNR = [0, 5, 10, 15, 20] dB
DOA_separation = [5, 10, 15, 20] degrees
snapshots = 256
coupling_models = ["exponential", "toeplitz"]
coupling_strength = [0.1, 0.2, 0.3, 0.4, 0.5]
```

**Expected Outputs:**
- RMSE degradation vs. coupling strength
- Condition number analysis
- DOF loss characterization

---

### **Scenario 3: ALSS Regularization Effectiveness**

**Objective:** Evaluate ALSS performance improvement under coupling

**Configuration:**
```python
array = "Z5"
algorithms = ["CoarrayMUSIC", "CoarrayMUSIC+ALSS"]
coupling_matrix = generate_mcm(N, positions, model="exponential", c1=0.3)
SNR = [-5, 0, 5, 10, 15, 20] dB
DOA_separation = [2, 5, 10, 13, 15, 20] degrees  # Focus on hard cases
snapshots = [64, 128, 256, 512]
```

**Expected Outputs:**
- RMSE improvement percentage
- Resolution improvement at low SNR
- Hard case (Δθ ≈ 13°) performance

---

### **Scenario 4: Array Comparison Under Realistic Conditions**

**Objective:** Compare different array types with mutual coupling

**Configuration:**
```python
arrays = ["ULA", "Nested", "Z5", "Z6"]
N_normalized = True  # Same aperture or same number of sensors
coupling_matrix = generate_mcm(N, positions, model="exponential", c1=0.3)
SNR = [5, 10, 15] dB
DOA_separation = [5, 10, 15, 20] degrees
snapshots = 256
```

**Expected Outputs:**
- Comparative RMSE plots
- DOF efficiency under coupling
- Robustness ranking

---

### **Scenario 5: Coupling Model Comparison**

**Objective:** Evaluate sensitivity to coupling model choice

**Configuration:**
```python
array = "Z5"
N = 7
coupling_models = {
    "exponential": {"c1": 0.3, "alpha": 0.5},
    "toeplitz": {"coupling_coeffs": [1.0, 0.3, 0.15, 0.08, 0.04, 0.02, 0.01]},
    "measured": {"matrix_file": "measured_coupling.csv"}  # If available
}
SNR = 10 dB
DOA_separation = 10 degrees
snapshots = 256
trials = 500
```

**Expected Outputs:**
- RMSE comparison across models
- Bias analysis
- Model sensitivity study

---

## Execution Workflow

### **Step 1: Configure Experiment**

```python
# Edit experiment configuration
nano configs/experiment_config.yaml
```

### **Step 2: Run Benchmarks**

```powershell
# Activate virtual environment
.\activate_venv.bat

# Run specific scenario
python core/analysis_scripts/run_scenario1_baseline.py

# Or use automated runner
.\run_benchmarks_with_venv.ps1 -Arrays "Z5" -N 7 -Trials 500 -WithCoupling
```

### **Step 3: Analyze Results**

```powershell
# Generate plots
python core/analysis_scripts/plot_results.py --scenario 1 --save-pdf

# Export tables
python core/analysis_scripts/export_tables.py --format latex --scenario 1
```

### **Step 4: Archive Results**

```powershell
# Save results with timestamp
$date = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item -Path "results\bench\*" -Destination "results\archived\scenario1_$date" -Recurse
```

---

## Output Organization

```
results/
├── scenario1_baseline/
│   ├── csv/                  # Raw data
│   ├── plots/                # PNG/PDF figures
│   └── summary.md            # Results summary
│
├── scenario2_coupling_impact/
│   ├── csv/
│   ├── plots/
│   └── summary.md
│
├── scenario3_alss_effectiveness/
│   └── ...
│
├── scenario4_array_comparison/
│   └── ...
│
└── scenario5_model_comparison/
    └── ...
```

---

## Quick Start Commands

```powershell
# Scenario 1: Baseline (No Coupling)
python core/analysis_scripts/run_scenario1.py --arrays ULA,Nested,Z5 --trials 500

# Scenario 2: With Coupling
python core/analysis_scripts/run_scenario2.py --coupling-strength 0.1,0.2,0.3,0.4,0.5

# Scenario 3: ALSS Comparison
python core/analysis_scripts/run_scenario3.py --algorithms CoarrayMUSIC,ALSS

# Scenario 4: Array Comparison
python core/analysis_scripts/run_scenario4.py --with-coupling --trials 500

# Scenario 5: Model Comparison
python core/analysis_scripts/run_scenario5.py --models exponential,toeplitz
```

---

## Expected Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Scenario 1 | 2-3 hours | Baseline performance database |
| Scenario 2 | 3-4 hours | Coupling impact study |
| Scenario 3 | 2-3 hours | ALSS effectiveness report |
| Scenario 4 | 3-4 hours | Comparative analysis |
| Scenario 5 | 2-3 hours | Model sensitivity study |
| **Total** | **12-17 hours** | **Complete experimental dataset** |

---

## Notes

- **Random Seeds:** Use consistent seeds for reproducibility
- **Parallel Execution:** Can run multiple scenarios simultaneously
- **Data Backup:** Automated backup to `results/archived/`
- **Publication Plots:** 300 DPI, LaTeX-ready formatting
- **Statistical Significance:** Minimum 100 trials, 500 recommended

---

## Next Steps

1. ✅ ~~Environment prepared~~
2. ⏳ Define specific research questions
3. ⏳ Customize experiment configurations
4. ⏳ Run experiments and collect data
5. ⏳ Analyze results and generate figures
6. ⏳ Prepare publication materials

---

**Environment Ready!** 🎯  
All tools, documentation, and infrastructure in place for comprehensive experimental study.
