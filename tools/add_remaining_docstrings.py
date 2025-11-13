"""
Add remaining docstrings to geometry processors.

Author: Hossein Molhem
Date: November 12, 2025
"""

import re
from pathlib import Path


def make_docstring(desc, has_n=False, has_d=False, returns='None', raises='None'):
    """Create a formatted docstring."""
    doc_lines = [
        '        """',
        f'        {desc}',
        '        ',
        '        Author: Hossein Molhem',
        '        ',
        '        Args:'
    ]
    
    if has_n:
        doc_lines.append('            N (int): Total number of sensors')
    if has_d:
        doc_lines.append('            d (float): Physical spacing multiplier (default: 1.0)')
    if not has_n and not has_d:
        doc_lines.append('            None')
    
    doc_lines.extend([
        '            ',
        '        Returns:',
        f'            {returns}',
        '            ',
        '        Raises:',
        f'            {raises}',
        '        """'
    ])
    
    return '\n'.join(doc_lines)


# Method definitions: (method_name, description, has_N_param, has_d_param, returns, raises)
METHODS = {
    'geometry_processors/z3_1_processor.py': [
        ('__init__', 'Initialize Z3(1) array processor.', True, True, 'None', 'ValueError: If N < 5'),
        ('compute_array_spacing', 'Set the physical sensor spacing.', False, False, 'None (updates self.data.sensor_spacing)', 'None'),
        ('compute_all_differences', 'Compute all pairwise differences between sensor positions.', False, False, 'None (updates self.data with difference computations)', 'None'),
        ('analyze_coarray', 'Analyze the difference coarray to identify unique positions and virtual sensors.', False, False, 'None (updates self.data with coarray analysis)', 'None'),
        ('plot_coarray', 'Plot the coarray visualization.', False, False, 'None', 'None'),
        ('compute_weight_distribution', 'Compute the weight distribution (frequency count) for each lag.', False, False, 'None (updates self.data with weight distribution)', 'None'),
        ('analyze_contiguous_segments', 'Identify and analyze contiguous segments in the one-sided coarray.', False, False, 'None (updates self.data with segment analysis)', 'None'),
        ('analyze_holes', 'Identify missing positions (holes) in the one-sided coarray.', False, False, 'None (updates self.data.missing_virtual_positions)', 'None'),
        ('generate_performance_summary', 'Generate comprehensive performance summary table for the array.', False, False, 'None (updates self.data.performance_summary_table)', 'None'),
    ],
    'geometry_processors/z3_2_processor.py': [
        ('__init__', 'Initialize Z3(2) array processor.', True, True, 'None', 'ValueError: If N < 5'),
        ('compute_array_spacing', 'Set the physical sensor spacing.', False, False, 'None (updates self.data.sensor_spacing)', 'None'),
        ('compute_all_differences', 'Compute all pairwise differences between sensor positions.', False, False, 'None (updates self.data with difference computations)', 'None'),
        ('analyze_coarray', 'Analyze the difference coarray to identify unique positions and virtual sensors.', False, False, 'None (updates self.data with coarray analysis)', 'None'),
        ('plot_coarray', 'Plot the coarray visualization.', False, False, 'None', 'None'),
        ('compute_weight_distribution', 'Compute the weight distribution (frequency count) for each lag.', False, False, 'None (updates self.data with weight distribution)', 'None'),
        ('analyze_contiguous_segments', 'Identify and analyze contiguous segments in the one-sided coarray.', False, False, 'None (updates self.data with segment analysis)', 'None'),
        ('generate_performance_summary', 'Generate comprehensive performance summary table for the array.', False, False, 'None (updates self.data.performance_summary_table)', 'None'),
    ],
    'geometry_processors/z4_processor.py': [
        ('__init__', 'Initialize Z4 array processor with w(1)=w(2)=0 constraint.', True, True, 'None', 'ValueError: If N < 5 or invalid sensor generation'),
        ('__repr__', 'Return string representation of the Z4 array processor.', False, False, 'str: String representation including N and spacing parameters', 'None'),
        ('compute_array_spacing', 'Set the physical sensor spacing.', False, False, 'None (updates self.data.sensor_spacing)', 'None'),
        ('compute_all_differences', 'Compute all pairwise differences between sensor positions.', False, False, 'None (updates self.data with difference computations)', 'None'),
        ('analyze_coarray', 'Analyze the difference coarray to identify unique positions and virtual sensors.', False, False, 'None (updates self.data with coarray analysis)', 'None'),
        ('compute_weight_distribution', 'Compute the weight distribution (frequency count) for each lag.', False, False, 'None (updates self.data with weight distribution)', 'None'),
        ('analyze_contiguous_segments', 'Identify and analyze contiguous segments in the one-sided coarray.', False, False, 'None (updates self.data with segment analysis)', 'None'),
        ('analyze_holes', 'Identify missing positions (holes) in the one-sided coarray.', False, False, 'None (updates self.data.missing_virtual_positions)', 'None'),
        ('generate_performance_summary', 'Generate comprehensive performance summary table for the array.', False, False, 'None (updates self.data.performance_summary_table)', 'None'),
        ('plot_coarray', 'Plot the coarray visualization.', False, False, 'None', 'None'),
        ('_weight_dict', 'Return dictionary mapping lag values to their weights.', False, False, 'Dict[int, int]: Mapping of lag to weight count', 'None'),
    ],
    'geometry_processors/z5_processor.py': [
        ('__init__', 'Initialize Z5 array processor with optimization.', True, True, 'None', 'ValueError: If N < 5'),
        ('__repr__', 'Return string representation of the Z5 array processor.', False, False, 'str: String representation including N and spacing parameters', 'None'),
        ('_build_z5_positions', 'Build Z5 sensor positions using greedy optimization.', False, False, 'np.ndarray: Integer grid positions for sensors', 'None'),
        ('_preserves_constraints', 'Check if position list maintains w(1)=0 constraint.', False, False, 'bool: True if constraints are satisfied', 'None'),
        ('_largest_contig_segment_nonneg', 'Find largest contiguous segment in non-negative lags.', False, False, 'int: Length of largest contiguous segment', 'None'),
        ('_score_L_and_holes', 'Compute score based on segment length and hole count.', False, False, 'float: Optimization score value', 'None'),
        ('_improve_once_by_perturb', 'Attempt single perturbation to improve array quality.', False, False, 'Tuple[bool, np.ndarray]: Success flag and potentially improved positions', 'None'),
        ('analyze_geometry', 'Analyze the physical array geometry.', False, False, 'None (updates self.data with geometry analysis)', 'None'),
        ('analyze_coarray', 'Analyze the difference coarray to identify unique positions and virtual sensors.', False, False, 'None (updates self.data with coarray analysis)', 'None'),
        ('_largest_contiguous_segment', 'Find the largest contiguous segment in the coarray.', False, False, 'Tuple: Segment array and its length', 'None'),
        ('compute_weight_distribution', 'Compute the weight distribution (frequency count) for each lag.', False, False, 'None (updates self.data with weight distribution)', 'None'),
        ('generate_performance_summary', 'Generate comprehensive performance summary table for the array.', False, False, 'None (updates self.data.performance_summary_table)', 'None'),
    ],
    'geometry_processors/z6_processor.py': [
        ('__init__', 'Initialize Z6 array processor.', True, True, 'None', 'ValueError: If N < 5'),
        ('__repr__', 'Return string representation of the Z6 array processor.', False, False, 'str: String representation including N and spacing parameters', 'None'),
        ('analyze_coarray', 'Analyze the difference coarray to identify unique positions and virtual sensors.', False, False, 'None (updates self.data with coarray analysis)', 'None'),
        ('_largest_contiguous_run', 'Find the largest contiguous run of positions.', False, False, 'Tuple: Start index, end index, and length of largest run', 'None'),
        ('generate_performance_summary', 'Generate comprehensive performance summary table for the array.', False, False, 'None (updates self.data.performance_summary_table)', 'None'),
        ('get_two_sided_holes', 'Get holes considering both positive and negative lags.', False, False, 'np.ndarray: Array of hole positions', 'None'),
        ('get_one_sided_holes', 'Get holes considering only non-negative lags.', False, False, 'np.ndarray: Array of hole positions', 'None'),
    ],
}


def add_docstrings_to_file(filepath, methods):
    """Add docstrings to specified methods in a file."""
    fpath = Path(filepath)
    if not fpath.exists():
        print(f'⚠ File not found: {filepath}')
        return 0
    
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    added_count = 0
    for method_name, desc, has_n, has_d, returns, raises in methods:
        # Pattern to find method definition without existing docstring
        pattern = rf'(    def {re.escape(method_name)}\([^)]*\):)\n(?!        """)'
        
        # Check if method exists
        if not re.search(pattern, content):
            # Try to find if it already has a docstring
            pattern_with_doc = rf'    def {re.escape(method_name)}\([^)]*\):\n        """'
            if re.search(pattern_with_doc, content):
                print(f'  ℹ Skipping {method_name} - already has docstring')
                continue
            else:
                print(f'  ⚠ Method not found: {method_name}')
                continue
        
        docstring = make_docstring(desc, has_n, has_d, returns, raises)
        replacement = r'\1\n' + docstring + '\n'
        content, n = re.subn(pattern, replacement, content, count=1)
        
        if n > 0:
            print(f'  ✓ Added docstring to {method_name}')
            added_count += 1
    
    if added_count > 0:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'  💾 Updated {filepath} with {added_count} docstrings')
    
    return added_count


def main():
    """Main entry point."""
    print("=" * 80)
    print("ADDING REMAINING DOCSTRINGS")
    print("Author: Hossein Molhem")
    print("=" * 80)
    print()
    
    total_added = 0
    for filepath, methods in METHODS.items():
        print(f"Processing: {filepath}")
        count = add_docstrings_to_file(filepath, methods)
        total_added += count
        print()
    
    print("=" * 80)
    print(f"COMPLETE: Added {total_added} docstrings")
    print("=" * 80)


if __name__ == '__main__':
    main()
