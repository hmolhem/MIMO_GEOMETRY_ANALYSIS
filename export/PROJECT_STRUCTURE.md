# MIMO Array Geometry Analysis - Project Structure

## Overview
This project analyzes MIMO radar array geometries through difference coarray analysis pipeline. It implements 8+ different array types with comprehensive analysis and visualization capabilities.

## Directory Structure

```
MIMO_GEOMETRY_ANALYSIS/
│
├── 📁 .github/
│   └── copilot-instructions.md         # AI coding agent instructions
│
├── 📁 analysis_scripts/                # Demo scripts for different array types
│   ├── graphical_demo.py              # Interactive graphical analysis tool
│   ├── methods_demo.py                # Method testing and validation
│   ├── run_nested_demo.py             # Nested Array demo
│   ├── run_ula_demo.py                # Uniform Linear Array demo
│   ├── run_z1_demo.py                 # Z1 Array demo
│   ├── run_z3_1_demo.py               # Z3(1) Array demo
│   ├── run_z3_2_demo.py               # Z3(2) Array demo
│   ├── run_z4_demo.py                 # Z4 Array demo
│   ├── run_z5_demo.py                 # Z5 Array demo
│   └── run_z6_demo.py                 # Z6 Array demo
│
├── 📁 data/                           # Data files (if any)
│
├── 📁 geometry_processors/            # Core analysis framework
│   ├── __init__.py                    # Package initialization
│   ├── bases_classes.py               # Abstract base classes (ArraySpec, BaseArrayProcessor)
│   ├── ula_processors.py              # Uniform Linear Array processor
│   ├── nested_processor.py            # Nested Array processor
│   ├── z1_processor.py                # Z1 Array processor (2-Sparse ULA + 1 Sensor)
│   ├── z3_1_processor.py              # Z3(1) Array processor (4-Sparse ULA + 3 Sensors Same Side)
│   ├── z3_2_processor.py              # Z3(2) Array processor (4-Sparse ULA + 3 Sensors Variant)
│   ├── z4_processor.py                # Z4 Array processor (w(1)=w(2)=0 Array)
│   ├── z5_processor.py                # Z5 Array processor (Advanced w(1)=w(2)=0 Array)
│   └── z6_processor.py                # Z6 Array processor (Ultimate Weight Constraints)
│
├── 📁 mimo-geom-dev/                  # Python virtual environment (Python 3.13.0)
│   ├── pyvenv.cfg                     # Virtual environment configuration
│   ├── 📁 Include/                    # Python headers
│   ├── 📁 Lib/                        # Python packages (numpy, pandas, matplotlib, etc.)
│   ├── 📁 Scripts/                    # Environment activation scripts
│   └── 📁 share/                      # Shared resources
│
├── 📁 miniScript/                     # Small utility scripts
│   └── testplt.py                     # Matplotlib testing script
│
├── 📁 notebooks/                      # Jupyter notebooks (if any)
│
├── 📁 results/                        # Analysis outputs
│   ├── method_test_log.txt            # Automated method testing log
│   ├── 📁 plots/                      # Generated visualization files
│   │   ├── Array_Z1_(N=5)_analysis.png
│   │   ├── Array_Z3(1)_(N=5)_analysis.png
│   │   ├── Array_Z3(2)_(N=5)_analysis.png
│   │   ├── Array_Z4_(N=5)_analysis.png
│   │   ├── Array_Z5_(N=5)_analysis.png
│   │   ├── Array_Z6_(N=5)_analysis.png
│   │   ├── Nested_Array_N=5_analysis.png
│   │   └── ULA_M5_analysis.png
│   └── 📁 summaries/                  # CSV/Excel performance summaries
│
├── README.md                          # Project documentation (empty)
└── requirements.txt                   # Python dependencies
```

## Key Components

### 🔧 Core Framework (`geometry_processors/`)
- **`bases_classes.py`**: Abstract framework with `ArraySpec` (47 attributes) and `BaseArrayProcessor` (7 abstract methods)
- **Array Processors**: 8+ concrete implementations for different MIMO array geometries

### 🎯 Analysis Scripts (`analysis_scripts/`)
- **`graphical_demo.py`**: Interactive tool for comprehensive analysis with both graphical plots and detailed text output
- **`methods_demo.py`**: Automated testing of all abstract method implementations
- **Individual demos**: Standalone scripts for each array type

### 📊 Results (`results/`)
- **`plots/`**: High-resolution PNG files with 6-panel analysis visualizations
- **`summaries/`**: Performance comparison tables
- **`method_test_log.txt`**: Automated testing results

## Array Types Implemented

1. **ULA** - Uniform Linear Array
2. **Nested** - Nested Array (contiguous coarray)
3. **Z1** - 2-Sparse ULA + 1 Sensor (w(1)=0)
4. **Z3(1)** - 4-Sparse ULA + 3 Sensors Same Side (w(1)=0, w(2)=2)
5. **Z3(2)** - 4-Sparse ULA + 3 Sensors Variant (w(1)=0, w(2)=1)
6. **Z4** - w(1)=w(2)=0 Array
7. **Z5** - Advanced w(1)=w(2)=0 Array
8. **Z6** - Ultimate Weight Constraints Array

## Dependencies

```
numpy>=1.21.0      # Core array operations
pandas>=1.3.0      # Performance summary tables
matplotlib>=3.5.0  # Visualization
```

## Quick Start

1. **Activate virtual environment**:
   ```powershell
   .\mimo-geom-dev\Scripts\Activate.ps1
   ```

2. **Run interactive analysis**:
   ```powershell
   python .\analysis_scripts\graphical_demo.py
   ```

3. **Test all methods**:
   ```powershell
   python .\analysis_scripts\methods_demo.py
   ```

## Analysis Pipeline

Each array processor follows a standardized 7-step analysis:
1. **Physical Array Specification** - Define sensor positions and spacing
2. **Difference Coarray Computation** - Calculate all N² pairwise differences
3. **Coarray Analysis** - Identify unique positions, virtual-only elements, holes
4. **Weight Distribution** - Count frequency of each lag
5. **Contiguous Segment Analysis** - Find maximum contiguous segments
6. **Holes Analysis** - Identify missing positions
7. **Performance Summary** - Generate metrics table for comparison

## Features

- ✅ **8+ Array Types** with standardized analysis
- ✅ **Interactive Graphical Analysis** with comprehensive visualizations
- ✅ **Detailed Text Output** mirroring all graphical information
- ✅ **Automated Testing** of all method implementations
- ✅ **High-Resolution Plots** saved as PNG files
- ✅ **Performance Comparison** tables and metrics
- ✅ **Virtual Environment** with all dependencies pre-installed