"""
Batch add docstrings to remaining geometry processors.

Author: Hossein Molhem
Date: November 12, 2025
"""

import re
from pathlib import Path

# Standard docstring templates for common methods
DOCSTRINGS = {
    '__init__': '''"""
        Initialize {class_name} array processor.
        
        Author: Hossein Molhem
        
        Args:
            N (int): Total number of sensors
            d (float): Physical spacing multiplier (default: 1.0)
            
        Returns:
            None
            
        Raises:
            ValueError: If N does not meet minimum requirements
        """''',
    
    'compute_array_spacing': '''"""
        Set the physical sensor spacing.
        
        Author: Hossein Molhem
        
        Args:
            None
            
        Returns:
            None (updates self.data.sensor_spacing)
            
        Raises:
            None
        """''',
    
    'compute_all_differences': '''"""
        Compute all pairwise differences between sensor positions.
        
        Author: Hossein Molhem
        
        Args:
            None
            
        Returns:
            None (updates self.data with difference computations)
            
        Raises:
            None
        """''',
    
    'analyze_coarray': '''"""
        Analyze the difference coarray to identify unique positions and virtual sensors.
        
        Author: Hossein Molhem
        
        Args:
            None
            
        Returns:
            None (updates self.data with coarray analysis)
            
        Raises:
            None
        """''',
    
    'plot_coarray': '''"""
        Plot the coarray visualization.
        
        Author: Hossein Molhem
        
        Args:
            None
            
        Returns:
            None
            
        Raises:
            None
        """''',
    
    'compute_weight_distribution': '''"""
        Compute the weight distribution (frequency count) for each lag.
        
        Author: Hossein Molhem
        
        Args:
            None
            
        Returns:
            None (updates self.data with weight distribution)
            
        Raises:
            None
        """''',
    
    'analyze_contiguous_segments': '''"""
        Identify and analyze contiguous segments in the coarray.
        
        Author: Hossein Molhem
        
        Args:
            None
            
        Returns:
            None (updates self.data with segment analysis)
            
        Raises:
            None
        """''',
    
    'analyze_holes': '''"""
        Identify missing positions (holes) in the coarray.
        
        Author: Hossein Molhem
        
        Args:
            None
            
        Returns:
            None (updates self.data.missing_virtual_positions)
            
        Raises:
            None
        """''',
    
    'generate_performance_summary': '''"""
        Generate comprehensive performance summary table for the array.
        
        Author: Hossein Molhem
        
        Args:
            None
            
        Returns:
            None (updates self.data.performance_summary_table)
            
        Raises:
            None
        """''',
    
    '__repr__': '''"""
        Return string representation of the array processor.
        
        Author: Hossein Molhem
        
        Args:
            None
            
        Returns:
            str: String representation including processor parameters
            
        Raises:
            None
        """''',
}

# Files and methods to update
FILES_TO_UPDATE = {
    'geometry_processors/z3_1_processor.py': [
        '__init__', 'compute_array_spacing', 'compute_all_differences',
        'analyze_coarray', 'plot_coarray', 'compute_weight_distribution',
        'analyze_contiguous_segments', 'analyze_holes', 'generate_performance_summary'
    ],
    'geometry_processors/z3_2_processor.py': [
        '__init__', 'compute_array_spacing', 'compute_all_differences',
        'analyze_coarray', 'plot_coarray', 'compute_weight_distribution',
        'analyze_contiguous_segments', 'generate_performance_summary'
    ],
    'geometry_processors/z4_processor.py': [
        '__init__', '__repr__', 'compute_array_spacing', 'compute_all_differences',
        'analyze_coarray', 'compute_weight_distribution', 'analyze_contiguous_segments',
        'analyze_holes', 'generate_performance_summary', 'plot_coarray', '_weight_dict'
    ],
    'geometry_processors/z5_processor.py': [
        '__init__', '__repr__', '_build_z5_positions', '_preserves_constraints',
        '_largest_contig_segment_nonneg', '_score_L_and_holes', '_improve_once_by_perturb',
        'analyze_geometry', 'analyze_coarray', '_largest_contiguous_segment',
        'compute_weight_distribution', 'generate_performance_summary'
    ],
    'geometry_processors/z6_processor.py': [
        '__init__', '__repr__', 'analyze_coarray', '_largest_contiguous_run',
        'generate_performance_summary', 'get_two_sided_holes', 'get_one_sided_holes'
    ],
}


def add_docstring_after_def(content: str, method_name: str, docstring: str) -> str:
    """
    Add docstring immediately after a method definition.
    
    Author: Hossein Molhem
    
    Args:
        content (str): File content
        method_name (str): Name of the method
        docstring (str): Docstring to add
        
    Returns:
        str: Modified content with docstring added
        
    Raises:
        None
    """
    # Pattern to find method definition
    pattern = rf'(\s+def {re.escape(method_name)}\([^)]*\):)'
    
    # Find the method
    match = re.search(pattern, content)
    if not match:
        print(f"  ⚠ Warning: Could not find method '{method_name}'")
        return content
    
    # Check if docstring already exists
    pos = match.end()
    after_def = content[pos:pos+500]
    if '"""' in after_def[:50] or "'''" in after_def[:50]:
        print(f"  ℹ Skipping '{method_name}' - already has docstring")
        return content
    
    # Insert docstring after the def line
    new_content = content[:pos] + '\n' + docstring + content[pos:]
    print(f"  ✓ Added docstring to '{method_name}'")
    return new_content


def process_file(filepath: Path, methods: list):
    """
    Process a file and add docstrings to specified methods.
    
    Author: Hossein Molhem
    
    Args:
        filepath (Path): Path to the file
        methods (list): List of method names to add docstrings to
        
    Returns:
            None
            
    Raises:
        None
    """
    print(f"\nProcessing: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    for method in methods:
        if method in DOCSTRINGS:
            docstring = DOCSTRINGS[method]
            content = add_docstring_after_def(content, method, docstring)
        else:
            print(f"  ⚠ No template for method '{method}'")
    
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  💾 File updated successfully")
    else:
        print(f"  ℹ No changes made")


def main():
    """
    Main entry point for batch docstring addition.
    
    Author: Hossein Molhem
    
    Args:
        None
        
    Returns:
        None
        
    Raises:
        None
    """
    root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("BATCH DOCSTRING ADDITION")
    print("Author: Hossein Molhem")
    print("=" * 80)
    
    for rel_path, methods in FILES_TO_UPDATE.items():
        filepath = root / rel_path
        if filepath.exists():
            process_file(filepath, methods)
        else:
            print(f"\n⚠ File not found: {filepath}")
    
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
