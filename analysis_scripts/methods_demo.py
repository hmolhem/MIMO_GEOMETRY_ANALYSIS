# analysis_scripts/methods_demo.py

import os
import sys
import inspect

# Add the project root to the path to enable absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import available processors
from geometry_processors.bases_classes import BaseArrayProcessor
from geometry_processors.ula_processors import ULArrayProcessor
from geometry_processors.nested_processor import NestedArrayProcessor
from geometry_processors.z1_processor import Z1ArrayProcessor
from geometry_processors.z3_1_processor import Z3_1ArrayProcessor
from geometry_processors.z3_2_processor import Z3_2ArrayProcessor
from geometry_processors.z4_processor import Z4ArrayProcessor
from geometry_processors.z5_processor import Z5ArrayProcessor
from geometry_processors.z6_processor import Z6ArrayProcessor

def explore_methods(processor_class, processor_name):
    """
    Explores and demonstrates all methods of a given processor class.
    """
    print(f"\n{'='*60}")
    print(f"EXPLORING METHODS: {processor_name}")
    print(f"{'='*60}")
    
    # Get all methods (excluding private ones)
    all_methods = [method for method in dir(processor_class) if not method.startswith('_')]
    print(f"Available Methods ({len(all_methods)}): {all_methods}")
    
    # Get abstract methods from base class
    if hasattr(processor_class, '__abstractmethods__'):
        abstract_methods = list(processor_class.__abstractmethods__)
        print(f"Abstract Methods Implemented: {abstract_methods}")
    
    print(f"\n--- METHOD DEMONSTRATIONS ---")
    
    # Create processor instance for demonstration
    if processor_class == ULArrayProcessor:
        processor = ULArrayProcessor(M=4, d=1)
    elif processor_class == NestedArrayProcessor:
        processor = NestedArrayProcessor(N1=2, N2=2, d=1)
    else:
        print("Unknown processor type")
        return
    
    # Demonstrate each method individually
    methods_to_demo = [
        'compute_array_spacing',
        'compute_all_differences', 
        'analyze_coarray',
        'compute_weight_distribution',
        'analyze_contiguous_segments',
        'analyze_holes',
        'generate_performance_summary',
        'plot_coarray'
    ]
    
    for method_name in methods_to_demo:
        if hasattr(processor, method_name):
            print(f"\n--- Executing: {method_name}() ---")
            try:
                method = getattr(processor, method_name)
                method()
                
                # Show relevant data after method execution
                if method_name == 'compute_array_spacing':
                    print(f"Sensor spacing: {processor.data.sensor_spacing}")
                    
                elif method_name == 'compute_all_differences':
                    print(f"Total computations: {processor.data.total_diff_computations}")
                    print(f"Sample differences: {processor.data.all_differences_with_duplicates[:10]}...")
                    
                elif method_name == 'analyze_coarray':
                    print(f"Unique positions: {len(processor.data.unique_differences)}")
                    print(f"Virtual-only positions: {len(processor.data.virtual_only_positions)}")
                    
                elif method_name == 'compute_weight_distribution':
                    print(f"Weight table shape: {processor.data.weight_table.shape}")
                    
                elif method_name == 'analyze_contiguous_segments':
                    print(f"Max detectable sources: {processor.data.max_detectable_sources}")
                    print(f"Segment lengths: {processor.data.segment_lengths}")
                    
                elif method_name == 'analyze_holes':
                    print(f"Number of holes: {processor.data.num_holes}")
                    
                elif method_name == 'generate_performance_summary':
                    print("Performance Summary Generated:")
                    print(processor.data.performance_summary_table.to_string(index=False))
                    
            except Exception as e:
                print(f"Error executing {method_name}: {e}")
    
    print(f"\n--- DATA ATTRIBUTES POPULATED ---")
    data_attrs = [attr for attr in dir(processor.data) if not attr.startswith('_')]
    populated_attrs = []
    for attr in data_attrs:
        value = getattr(processor.data, attr)
        if value is not None:
            populated_attrs.append(f"{attr}: {type(value).__name__}")
    
    print(f"Populated attributes ({len(populated_attrs)}):")
    for attr in populated_attrs:
        print(f"  - {attr}")

def main():
    """
    Main function to explore methods of selected processors.
    """
    print("MIMO ARRAY GEOMETRY ANALYSIS - METHOD EXPLORER")
    print("=" * 60)
    
    # Available processors
    processors = {
        '1': (ULArrayProcessor, 'Uniform Linear Array (ULA)'),
        '2': (NestedArrayProcessor, 'Nested Array (NA)'),
        '3': (Z1ArrayProcessor, 'Array z₁ - 2-Sparse ULA + 1 Sensor'),
        '4': (Z3_1ArrayProcessor, 'Array z₃⁽¹⁾ - 4-Sparse ULA + 3 Sensors Same Side'),
        '5': (Z3_2ArrayProcessor, 'Array z₃⁽²⁾ - 4-Sparse ULA + 3 Sensors Variant'),
        '6': (Z4ArrayProcessor, 'Array z₄ - w(1)=w(2)=0 Array'),
        '7': (Z5ArrayProcessor, 'Array z₅ - Advanced w(1)=w(2)=0 Array'),
        '8': (Z6ArrayProcessor, 'Array z₆ - Ultimate Weight Constraints')
        }
    
    exit_options = ['0', 'exit', 'quit', 'q']
    
    print("Available Processors:")
    for key, (cls, name) in processors.items():
        print(f"  {key}. {name}")
    print("  0. Exit")
    
    # Get user selection
    print("\nSelect processors to explore (e.g., '1,2') or '0' to exit:")
    selection = input("Your choice: ").strip().lower()
    
    # Handle exit
    if selection in exit_options:
        print("Exiting METHOD EXPLORER. Goodbye!")
        return
    
    # Parse selection (only support comma-separated numbers, no 'all')
    selected_keys = [key.strip() for key in selection.split(',') if key.strip() in processors]
    
    # Check if no valid selections
    if not selected_keys:
        print("No valid processors selected. Exiting.")
        return
    
    # Create log file
    log_filename = "results/method_test_log.txt"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_file.write("MIMO ARRAY GEOMETRY ANALYSIS - METHOD TEST LOG\n")
        log_file.write("=" * 60 + "\n")
        log_file.write(f"Test Date: {os.path.getctime('.')}\n\n")
        
        # Get abstract methods from base class
        abstract_methods = list(BaseArrayProcessor.__abstractmethods__)
        log_file.write(f"Abstract Methods to Test: {abstract_methods}\n\n")
        
        # Explore selected p~rocessors
        for key in selected_keys:
            if key in processors:
                processor_class, processor_name = processors[key]
                
                log_file.write(f"\n{'='*60}\n")
                log_file.write(f"TESTING: {processor_name}\n")
                log_file.write(f"{'='*60}\n")
                
                # Test each abstract method
                try:
                    # Create processor instance with M=5
                    if processor_class == ULArrayProcessor:
                        processor = ULArrayProcessor(M=5, d=1)
                    elif processor_class == NestedArrayProcessor:
                        processor = NestedArrayProcessor(N1=2, N2=3, d=1)  # Total N=5
                    else:
                        log_file.write(f"ERROR: Unknown processor type {processor_class}\n")
                        continue
                    
                    log_file.write(f"Processor created successfully: {processor.data.name} (M=5)\n")
                    log_file.write(f"Sensor positions: {processor.data.sensors_positions}\n")
                    
                    # Test each abstract method individually
                    for method_name in abstract_methods:
                        if hasattr(processor, method_name):
                            try:
                                method = getattr(processor, method_name)
                                method()
                                log_file.write(f"[OK] {method_name}(): SUCCESS\n")
                                
                                # Log specific results for each method
                                if method_name == 'compute_array_spacing':
                                    log_file.write(f"  - Sensor spacing: {processor.data.sensor_spacing}\n")
                                elif method_name == 'compute_all_differences':
                                    log_file.write(f"  - Total computations: {processor.data.total_diff_computations}\n")
                                    log_file.write(f"  - Unique differences: {len(processor.data.all_differences_with_duplicates) if processor.data.all_differences_with_duplicates is not None else 0}\n")
                                elif method_name == 'analyze_coarray':
                                    log_file.write(f"  - Unique positions: {processor.data.num_unique_positions}\n")
                                    log_file.write(f"  - Virtual-only elements: {len(processor.data.virtual_only_positions) if processor.data.virtual_only_positions is not None else 0}\n")
                                elif method_name == 'compute_weight_distribution':
                                    log_file.write(f"  - Weight table shape: {processor.data.weight_table.shape if processor.data.weight_table is not None else 'None'}\n")
                                elif method_name == 'analyze_contiguous_segments':
                                    log_file.write(f"  - Max detectable sources: {processor.data.max_detectable_sources}\n")
                                    log_file.write(f"  - Segment lengths: {processor.data.segment_lengths}\n")
                                elif method_name == 'analyze_holes':
                                    log_file.write(f"  - Number of holes: {processor.data.num_holes}\n")
                                elif method_name == 'generate_performance_summary':
                                    log_file.write(f"  - Performance table created: {processor.data.performance_summary_table is not None}\n")
                                elif method_name == 'plot_coarray':
                                    log_file.write(f"  - Plotting completed\n")
                                    
                            except Exception as e:
                                log_file.write(f"[FAIL] {method_name}(): FAILED - {str(e)}\n")
                        else:
                            log_file.write(f"[MISSING] {method_name}(): NOT IMPLEMENTED\n")
                    
                    # Test full pipeline with fresh instance
                    log_file.write(f"\n--- Testing Full Pipeline ---\n")
                    try:
                        if processor_class == ULArrayProcessor:
                            test_processor = ULArrayProcessor(M=5, d=1)
                        elif processor_class == NestedArrayProcessor:
                            test_processor = NestedArrayProcessor(N1=2, N2=3, d=1)
                        
                        results = test_processor.run_full_analysis()
                        log_file.write(f"[OK] run_full_analysis(): SUCCESS\n")
                        log_file.write(f"  - Final processor name: {results.name}\n")
                        log_file.write(f"  - Total sensors: {results.num_sensors}\n")
                        log_file.write(f"  - Performance table: {results.performance_summary_table is not None}\n")
                        if results.performance_summary_table is not None:
                            log_file.write(f"  - Performance metrics count: {len(results.performance_summary_table)}\n")
                    except Exception as e:
                        log_file.write(f"[FAIL] run_full_analysis(): FAILED - {str(e)}\n")
                
                except Exception as e:
                    log_file.write(f"ERROR: Failed to create processor - {str(e)}\n")
        
        log_file.write(f"\n{'='*60}\n")
        log_file.write("METHOD TEST COMPLETE\n")
        log_file.write(f"{'='*60}\n")
    
    print(f"Method test results logged to: {log_filename}")
    print(f"\n{'='*60}")
    print("METHOD EXPLORATION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()