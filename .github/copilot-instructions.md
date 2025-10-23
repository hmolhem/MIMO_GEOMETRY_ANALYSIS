# MIMO Array Geometry Analysis Framework

## Architecture Overview

This project analyzes MIMO radar array geometries through a **difference coarray analysis pipeline**. The core concept: compute pairwise differences between sensor positions to derive virtual sensor arrays with enhanced degrees of freedom.

**Key Components:**
- `geometry_processors/bases_classes.py` - Abstract framework defining the 7-step analysis pipeline
- `geometry_processors/ula_processors.py` - Concrete ULA (Uniform Linear Array) implementation  
- `analysis_scripts/` - Entry points for running specific array configurations
- `results/plots/` & `results/summaries/` - Output directories for visualizations and data

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
processor = ULArrayProcessor(M=4, d=1.0)
results = processor.run_full_analysis()
print(results.performance_summary_table.to_markdown(index=False))
```

## Critical Dependencies

**Missing Import Declarations:** The codebase uses NumPy (`np.`) and Pandas (`pd.`) extensively but lacks explicit imports in `ula_processors.py`. Always add:

```python
import numpy as np
import pandas as pd
```

**Path Setup Pattern:** Analysis scripts use relative imports via sys.path manipulation:

```python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

### Adding New Array Types
1. Inherit from `BaseArrayProcessor` in `geometry_processors/`
2. Implement all 7 abstract methods (see ULA example)
3. Focus on `compute_all_differences()` - core algorithm varies by geometry
4. Create analysis script in `analysis_scripts/` with proper path setup

### Running Analysis
```python
# Standard pattern for all array types
processor = ArrayProcessor(params)
results = processor.run_full_analysis() 
# Results automatically saved to results.data.performance_summary_table
```

### Output Management
- Use `results/plots/` for matplotlib visualizations
- Use `results/summaries/` for CSV/Excel exports of performance tables
- The `plot_coarray()` method currently shows console visualization - extend for matplotlib

## Extension Points

- **Nested Arrays, Coprime Arrays** - Follow ULA processor pattern in new files
- **Comparative Analysis** - Batch multiple array types using the standardized pipeline  
- **Visualization Enhancement** - Replace console plots with matplotlib in `plot_coarray()`
- **Optimization Studies** - Vary parameters (M, d) and collect performance metrics

Focus on the **difference coarray computation** as the mathematical core - all other analysis flows from this step.