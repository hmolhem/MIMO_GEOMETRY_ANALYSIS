# MIMO Array Geometry Analysis Framework

## Architecture Overview

This project analyzes MIMO radar array geometries through a **difference coarray analysis pipeline**. The core concept: compute pairwise differences between sensor positions to derive virtual sensor arrays with enhanced degrees of freedom.

**Key Components:**
- `geometry_processors/bases_classes.py` - Abstract framework with `ArraySpec` data container (47 attributes) and `BaseArrayProcessor` (8 abstract methods)
- `geometry_processors/*.py` - 8+ concrete array implementations (ULA, Nested, Z1-Z6 specialized arrays)  
- `analysis_scripts/` - Standalone demos with CLI parsing (`argparse`) and standardized Python path setup
- `results/` - Auto-generated outputs (plots/, summaries/, method_test_log.txt)
- `mimo-geom-dev/` - Local Python virtual environment (Python 3.13.0) managed via pyenv

## Core Analysis Pipeline

All array processors follow a standardized 7-step analysis sequence defined in `BaseArrayProcessor.run_full_analysis()`:

1. **Physical Array Specification** - Define sensor positions and spacing
2. **Difference Coarray Computation** - Calculate all N² pairwise differences (n_i - n_j)
3. **Coarray Analysis** - Identify unique positions, virtual-only elements, holes, and segments
4. **Weight Distribution** - Count frequency of each lag (difference value)
5. **Contiguous Segment Analysis** - Find maximum contiguous virtual array segments
6. **Holes Analysis** - Identify missing positions in the ideal contiguous range
7. **Performance Summary** - Generate metrics table for comparison

## Data Structure Patterns

### ArraySpec Data Container
The `ArraySpec` class serves as a structured data container with **47 pre-defined attributes** organized by analysis phase. Key naming conventions:

```python
# Physical arrays: sensor_positions, num_sensors, sensor_spacing
# Coarray data: unique_differences, coarray_positions, virtual_only_positions  
# Analysis results: weight_table, contiguous_segments, performance_summary_table
```

### Result Access Pattern
All processors return `ArraySpec` objects. Access results via the `.data` attribute:

```python
processor = ULArrayProcessor(M=4, d=1)
results = processor.run_full_analysis()
print(results.performance_summary_table.to_markdown(index=False))
```

## Critical Dependencies & Environment

**Environment Setup:** Project uses local virtual environment `mimo-geom-dev/` with Python 3.13.0. Key dependencies:
```python
numpy>=1.21.0  # Core array operations
pandas>=1.3.0  # Performance summary tables
matplotlib>=3.5.0  # Visualization (minimal usage)
```

**Import Patterns:** All concrete processors already include required imports:
```python
import numpy as np
import pandas as pd
from .bases_classes import BaseArrayProcessor
```

**Path Setup Pattern:** All analysis scripts use standardized relative import setup:
```python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geometry_processors.{processor_name} import {ProcessorClass}
```

## Radar Domain Specifics

### Virtual Array Concept
- **Physical sensors** create **virtual sensors** through difference coarray computation
- Virtual aperture size determines **maximum detectable sources** (K_max = floor(L/2))
- **Weight distribution** affects estimation accuracy - higher weights at small lags preferred

### Performance Metrics
Key metrics in `performance_summary_table`:
- **Coarray Aperture** - Span of virtual array (max_pos - min_pos)
- **Contiguous Segment Length (L)** - Longest hole-free virtual array segment
- **Maximum Detectable Sources (K_max)** - Theoretical source estimation limit
- **Weight at Lag 1/2/3** - Frequency of small-lag differences (quality indicators)

## Development Workflows

### Environment Setup (Windows + pyenv)
```powershell
# Activate virtual environment (Windows PowerShell)
.\mimo-geom-dev\Scripts\Activate.ps1

# Or use batch file
.\mimo-geom-dev\Scripts\activate.bat
```

**Critical:** Project uses pyenv-managed Python 3.13.0 in local virtual environment `mimo-geom-dev/`. All processors expect this environment with numpy>=1.21.0, pandas>=1.3.0, matplotlib>=3.5.0.

### Dual File Pattern
Most processors have dual implementations (e.g., `z4_processor.py` + `z4_processor_.py`). The underscore version typically represents:
- Alternative implementations or experimental versions
- Different parameter handling or optimization approaches
- Both should implement identical abstract methods from `BaseArrayProcessor`

### Available Array Types (8 Implementations)
- **ULA** (`ULArrayProcessor`) - `ULArrayProcessor(M=4, d=1)`
- **Nested** (`NestedArrayProcessor`) - `NestedArrayProcessor(N1=2, N2=3, d=1)`
- **Z1-Z6 Specialized Arrays** (`Z1ArrayProcessor`, `Z3_1ArrayProcessor`, etc.)

### CLI Demo Pattern
All demo scripts follow standardized argparse pattern:
```python
parser.add_argument("--N", type=int, default=7, help="Number of sensors")
parser.add_argument("--d", type=float, default=1.0, help="Physical spacing multiplier")
parser.add_argument("--markdown", action="store_true", help="Print summary in Markdown")
parser.add_argument("--save-csv", action="store_true", help="Save summary CSV")
parser.add_argument("--save-json", action="store_true", help="Save JSON sidecar")
```

### Adding New Array Types
1. Inherit from `BaseArrayProcessor` in `geometry_processors/`
2. Implement all 8 abstract methods following this pattern:
   ```python
   class NewArrayProcessor(BaseArrayProcessor):
       def __init__(self, custom_params):
           positions = # compute sensor positions
           super().__init__(name="CustomArray", array_type="Custom", sensor_positions=positions)
   ```
3. Focus on `compute_all_differences()` - core algorithm varies by geometry
4. Create demo script in `analysis_scripts/` with standardized path setup

### Advanced Processor Features (Z4 Example)
Z4 processors demonstrate domain-specific validation patterns:
```python
# Built-in invariant assertions for specialized arrays
if args.do_asserts:
    wt = {int(r["Lag"]): int(r["Weight"]) for _, r in data.weight_table.iterrows()}
    assert wt.get(1, 0) == 0 and wt.get(2, 0) == 0, "Z4 requires w(1)=w(2)=0"
    assert int(seg[0]) == 3, f"L1 must be 3; got {int(seg[0])}"
```
- Use canonical layouts for specific N values (e.g., N=7: `[0, 5, 8, 11, 14, 17, 21]`)
- Implement domain-specific validation assertions
- Support multiple hole analysis modes (`--holes one` vs `--holes both`)

### Running Analysis & Testing

**CLI Execution Pattern:**
```powershell
# Standard demo with output options
python analysis_scripts/run_z4_demo.py --N 7 --d 1.0 --markdown --save-csv --save-json

# With domain-specific assertions (Z4 arrays)
python analysis_scripts/run_z4_demo_.py --N 7 --assert --show-weights --holes both
```

**Programmatic Pattern:**
```python
# Standard pattern for all array types
processor = ProcessorClass(params)
results = processor.run_full_analysis()  # Returns ArraySpec with .data attribute
print(results.performance_summary_table.to_markdown(index=False))
```

**Method Testing:** Use `analysis_scripts/methods_demo.py` for comprehensive method validation - generates `results/method_test_log.txt` with detailed test results for all abstract method implementations.

### Output Management
- `results/plots/` - For matplotlib visualizations (currently minimal)
- `results/summaries/` - For CSV/Excel exports of performance tables  
- `results/method_test_log.txt` - Automated testing log from methods_demo.py
- `plot_coarray()` uses console ASCII visualization - extend for matplotlib as needed

## Extension Points

- **Nested Arrays, Coprime Arrays** - Follow ULA processor pattern in new files
- **Comparative Analysis** - Batch multiple array types using the standardized pipeline  
- **Visualization Enhancement** - Replace console plots with matplotlib in `plot_coarray()`
- **Optimization Studies** - Vary parameters (M, d) and collect performance metrics

Focus on the **difference coarray computation** as the mathematical core - all other analysis flows from this step.