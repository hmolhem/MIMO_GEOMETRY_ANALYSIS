# MIMO Geometry Analysis

A Python package for analyzing and visualizing MIMO (Multiple-Input Multiple-Output) antenna array geometries.

## Features

- **Multiple Array Configurations**
  - Uniform Linear Array (ULA)
  - Uniform Rectangular Array (URA)
  - Uniform Circular Array (UCA)
  - Custom array configurations

- **Geometry Analysis**
  - Element spacing statistics
  - Array aperture calculation
  - Pairwise distance and angle computations
  - Spatial correlation analysis
  - Channel capacity estimation

- **Visualization Tools**
  - 2D and 3D array plots
  - Array factor (radiation pattern) visualization
  - Spatial correlation heatmaps
  - Channel capacity vs SNR plots
  - Comprehensive summary plots

- **Array Transformations**
  - Translation
  - Rotation around any axis

## Installation

```bash
pip install -r requirements.txt
```

Or install in development mode:

```bash
pip install -e .
```

## Quick Start

### Creating and Analyzing Arrays

```python
from mimo_geometry import AntennaArray, GeometryAnalyzer, ArrayVisualizer

# Create a Uniform Linear Array
ula = AntennaArray.create_ula(num_elements=8, spacing=0.5)

# Analyze the array geometry
analyzer = GeometryAnalyzer(ula)
analysis = analyzer.analyze_array()
print(f"Array aperture: {analysis['aperture']:.3f} wavelengths")
print(f"Capacity @ 10dB SNR: {analysis['capacity_snr_10dB']:.2f} bits/s/Hz")

# Visualize the array
visualizer = ArrayVisualizer(ula)
fig, ax = visualizer.plot_array_2d()
fig.savefig('ula_plot.png')
```

### Creating Custom Arrays

```python
import numpy as np

# Define custom antenna positions
positions = np.array([
    [0, 0, 0],
    [0.5, 0, 0],
    [1.0, 0, 0],
    [0, 0.5, 0],
    [0, 1.0, 0]
])

custom_array = AntennaArray(positions, array_type="L-shaped")
```

### Array Types

#### Uniform Linear Array (ULA)

```python
ula = AntennaArray.create_ula(
    num_elements=8,  # Number of antenna elements
    spacing=0.5      # Element spacing in wavelengths
)
```

#### Uniform Rectangular Array (URA)

```python
ura = AntennaArray.create_ura(
    rows=4,          # Number of rows
    cols=4,          # Number of columns
    spacing=(0.5, 0.5)  # (x_spacing, y_spacing) in wavelengths
)
```

#### Uniform Circular Array (UCA)

```python
uca = AntennaArray.create_uca(
    num_elements=8,  # Number of elements
    radius=1.0       # Radius in wavelengths
)
```

## Examples

The `examples/` directory contains several demonstration scripts:

### Basic Example

```bash
cd examples
python basic_example.py
```

This creates and analyzes ULA, URA, and UCA configurations, generating:
- 2D array visualizations
- 3D array visualizations
- Array factor (radiation) patterns

### Advanced Example

```bash
cd examples
python advanced_example.py
```

This demonstrates:
- Custom array configurations
- Detailed geometry analysis
- Spatial correlation analysis
- Channel capacity analysis
- Array transformations (translation, rotation)

## Analysis Capabilities

### Geometry Analysis

```python
analyzer = GeometryAnalyzer(array)

# Get spacing statistics
stats = analyzer.compute_element_spacing_stats()
print(f"Mean spacing: {stats['mean_spacing']:.3f} wavelengths")

# Calculate array aperture
aperture = analyzer.compute_array_aperture()

# Compute spatial correlation
correlation = analyzer.compute_spatial_correlation(angle_spread=10.0)

# Estimate channel capacity
capacity = analyzer.estimate_channel_capacity(snr_db=10.0, angle_spread=10.0)
```

### Array Factor (Radiation Pattern)

```python
import numpy as np

azimuth_angles = np.linspace(-180, 180, 360)
array_factor = analyzer.compute_array_factor(
    azimuth_angles=azimuth_angles,
    elevation=0.0,
    wavelength=1.0
)
```

### Visualization

```python
visualizer = ArrayVisualizer(array)

# 2D plot
fig, ax = visualizer.plot_array_2d(plane='xy')

# 3D plot
fig, ax = visualizer.plot_array_3d()

# Array factor pattern
fig, axes = visualizer.plot_array_factor()

# Spatial correlation heatmap
fig, ax = visualizer.plot_spatial_correlation(angle_spread=10.0)

# Capacity vs SNR curve
fig, ax = visualizer.plot_capacity_vs_snr(snr_range=(-10, 30))

# Comprehensive summary plot
fig = visualizer.create_summary_plot()
```

## Running Tests

```bash
python -m unittest discover tests
```

Or run tests with verbose output:

```bash
python -m unittest tests/test_mimo_geometry.py -v
```

## Requirements

- Python >= 3.7
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- scipy >= 1.7.0

## Package Structure

```
mimo_geometry/
├── __init__.py          # Package initialization
├── antenna_array.py     # AntennaArray class definition
├── geometry_analyzer.py # GeometryAnalyzer class
└── visualizer.py        # ArrayVisualizer class

examples/
├── basic_example.py     # Basic usage examples
└── advanced_example.py  # Advanced features demo

tests/
└── test_mimo_geometry.py # Unit tests
```

## API Reference

### AntennaArray

Main class for representing antenna arrays.

**Class Methods:**
- `create_ula(num_elements, spacing)` - Create ULA
- `create_ura(rows, cols, spacing)` - Create URA
- `create_uca(num_elements, radius)` - Create UCA

**Properties:**
- `num_elements` - Number of antenna elements
- `center` - Geometric center of array
- `positions` - Array of 3D positions

**Methods:**
- `get_distances()` - Pairwise distance matrix
- `get_angles(reference_idx)` - Angles from reference antenna
- `translate(offset)` - Translate array
- `rotate(angle, axis)` - Rotate array

### GeometryAnalyzer

Analyzes geometric properties of arrays.

**Methods:**
- `compute_array_aperture()` - Calculate effective aperture
- `compute_element_spacing_stats()` - Spacing statistics
- `compute_spatial_correlation(angle_spread, wavelength)` - Spatial correlation matrix
- `estimate_channel_capacity(snr_db, angle_spread)` - Channel capacity
- `compute_array_factor(azimuth_angles, elevation, wavelength)` - Array factor
- `analyze_array()` - Comprehensive analysis

### ArrayVisualizer

Visualization tools for arrays.

**Methods:**
- `plot_array_2d(plane, show_labels, figsize)` - 2D projection
- `plot_array_3d(show_labels, figsize)` - 3D visualization
- `plot_array_factor(elevation, wavelength, figsize)` - Radiation pattern
- `plot_spatial_correlation(angle_spread, figsize)` - Correlation heatmap
- `plot_capacity_vs_snr(snr_range, angle_spread, figsize)` - Capacity curve
- `create_summary_plot(figsize)` - Comprehensive summary

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

MIMO Analysis Team

## Acknowledgments

This package provides tools for analyzing MIMO antenna geometries commonly used in wireless communication systems, including 5G, WiFi, and radar applications.