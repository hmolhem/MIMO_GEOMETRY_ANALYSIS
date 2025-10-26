# analysis_scripts/graphical_demo.py

import os
import sys
import numpy as np

# Handle path length issues with matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib loaded successfully - plots will be saved as files")
except ImportError as e:
    print(f"Matplotlib import error: {e}")
    print("Falling back to text-based analysis...")
    MATPLOTLIB_AVAILABLE = False

# Add the project root to the path to enable absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import available processors
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

def print_text_analysis(processor, processor_name):
    """
    Text-based analysis when matplotlib is not available.
    """
    print(f"\n{'='*60}")
    print(f"TEXT-BASED ANALYSIS: {processor_name}")
    print(f"{'='*60}")
    
    # Physical Array
    print(f"\n1. Physical Array Geometry:")
    print(f"   Sensor positions: {processor.data.sensors_positions}")
    print(f"   Number of sensors: {processor.data.num_sensors}")
    print(f"   Sensor spacing: {processor.data.sensor_spacing}")
    
    # Coarray Analysis
    if processor.data.coarray_positions is not None:
        print(f"\n2. Coarray Analysis:")
        print(f"   Total coarray elements: {len(processor.data.coarray_positions)}")
        print(f"   Coarray positions: {processor.data.coarray_positions}")
        print(f"   Virtual-only elements: {len(processor.data.virtual_only_positions)}")
    
    # Weight Distribution
    if processor.data.weight_table is not None:
        print(f"\n3. Weight Distribution:")
        print(processor.data.weight_table.to_string(index=False))
    
    # Contiguous Segments
    print(f"\n4. Contiguous Segments:")
    print(f"   Largest segment length: {len(processor.data.largest_contiguous_segment)}")
    print(f"   Max detectable sources: {processor.data.max_detectable_sources}")
    
    # Holes Analysis
    print(f"\n5. Holes Analysis:")
    print(f"   Number of holes: {processor.data.num_holes}")
    if processor.data.num_holes > 0:
        print(f"   Missing positions: {processor.data.missing_virtual_positions}")
    
    # Performance Summary
    if processor.data.performance_summary_table is not None:
        print(f"\n6. Performance Summary:")
        print(processor.data.performance_summary_table.to_string(index=False))

def plot_array_analysis(processor, processor_name):
    """
    Creates comprehensive graphical visualization of array analysis results.
    """
    print(f"\nGenerating analysis for: {processor_name}")
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Showing text-based analysis:")
        print_text_analysis(processor, processor_name)
        return None
    
    # Create figure with subplots
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{processor_name} - Array Analysis Results (M=5)', fontsize=16, fontweight='bold')
    except Exception as e:
        print(f"Error creating plots: {e}")
        print("Falling back to text-based analysis:")
        print_text_analysis(processor, processor_name)
        return None
    
    # Plot 1: Physical Array Geometry
    ax1 = axes[0, 0]
    positions = np.array(processor.data.sensors_positions)
    ax1.scatter(positions, np.zeros_like(positions), s=100, c='red', marker='s', label='Physical Sensors')
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Physical Array Geometry')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Force integer x-axis ticks
    ax1.set_xticks(positions.astype(int))
    ax1.set_xticklabels([str(int(pos)) for pos in positions])
    
    # Add position labels
    for i, pos in enumerate(positions):
        ax1.annotate(f'S{i+1}', (pos, 0), xytext=(0, 10), textcoords='offset points', ha='center')
    
    # Plot 2: Difference Coarray Positions
    ax2 = axes[0, 1]
    if processor.data.coarray_positions is not None:
        coarray_pos = processor.data.coarray_positions
        physical_set = set(processor.data.physical_positions)
        
        # Separate physical and virtual positions
        physical_coarray = [pos for pos in coarray_pos if pos in physical_set]
        virtual_coarray = [pos for pos in coarray_pos if pos not in physical_set]
        
        if physical_coarray:
            ax2.scatter(physical_coarray, np.zeros_like(physical_coarray), 
                       s=80, c='red', marker='s', label='Physical', alpha=0.8)
        if virtual_coarray:
            ax2.scatter(virtual_coarray, np.zeros_like(virtual_coarray), 
                       s=60, c='blue', marker='o', label='Virtual', alpha=0.7)
        
        ax2.set_xlabel('Lag Position')
        ax2.set_title(f'Coarray Positions ({len(coarray_pos)} elements)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Force integer x-axis ticks
        ax2.set_xticks(coarray_pos.astype(int))
        ax2.set_xticklabels([str(int(pos)) for pos in coarray_pos])
    
    # Plot 3: Weight Distribution
    ax3 = axes[0, 2]
    if processor.data.weight_table is not None:
        weights_df = processor.data.weight_table
        lags = weights_df.iloc[:, 0].values  # First column (lags)
        weights = weights_df.iloc[:, 1].values  # Second column (weights)
        
        bars = ax3.bar(lags, weights, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('Lag')
        ax3.set_ylabel('Weight (Count)')
        ax3.set_title('Weight Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Force integer x-axis ticks
        ax3.set_xticks(lags.astype(int))
        ax3.set_xticklabels([str(int(lag)) for lag in lags])
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(weight)}', ha='center', va='bottom')
    
    # Plot 4: Contiguous Segments Analysis
    ax4 = axes[1, 0]
    if processor.data.largest_contiguous_segment is not None:
        segment = processor.data.largest_contiguous_segment
        ax4.plot(segment, np.ones_like(segment), 'go-', linewidth=3, markersize=8, 
                label=f'Contiguous Segment (L={len(segment)})')
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Segment')
        ax4.set_title(f'Contiguous Segments (K_max={processor.data.max_detectable_sources})')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0.5, 1.5)
        
        # Force integer x-axis ticks
        ax4.set_xticks(segment.astype(int))
        ax4.set_xticklabels([str(int(pos)) for pos in segment])
    
    # Plot 5: Holes Analysis
    ax5 = axes[1, 1]
    if processor.data.coarray_positions is not None:
        coarray = processor.data.coarray_positions
        min_pos, max_pos = int(coarray.min()), int(coarray.max())
        ideal_range = np.arange(min_pos, max_pos + 1)
        
        # Mark existing and missing positions
        existing = [pos for pos in ideal_range if pos in coarray]
        missing = [pos for pos in ideal_range if pos not in coarray]
        
        if existing:
            ax5.scatter(existing, np.ones_like(existing), s=80, c='green', 
                       marker='s', label=f'Existing ({len(existing)})', alpha=0.8)
        if missing:
            ax5.scatter(missing, np.ones_like(missing), s=80, c='red', 
                       marker='x', label=f'Holes ({len(missing)})', alpha=0.8)
        
        ax5.set_xlabel('Position')
        ax5.set_ylabel('Status')
        ax5.set_title(f'Holes Analysis ({processor.data.num_holes} holes)')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        ax5.set_ylim(0.5, 1.5)
        
        # Force integer x-axis ticks
        ax5.set_xticks(ideal_range)
        ax5.set_xticklabels([str(int(pos)) for pos in ideal_range])
    
    # Plot 6: Performance Summary Table
    ax6 = axes[1, 2]
    ax6.axis('off')  # Hide axes for table
    
    if processor.data.performance_summary_table is not None:
        perf_table = processor.data.performance_summary_table
        
        # Create table data
        table_data = []
        for _, row in perf_table.iterrows():
            table_data.append([row['Metrics'], str(row['Value'])])
        
        # Create table
        table = ax6.table(cellText=table_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.7, 0.3])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style the table
        table[(0, 0)].set_facecolor('#4CAF50')
        table[(0, 1)].set_facecolor('#4CAF50')
        for i in range(1, len(table_data) + 1):
            table[(i, 0)].set_facecolor('#E8F5E8')
            table[(i, 1)].set_facecolor('#F0F8F0')
        
        ax6.set_title('Performance Summary', fontweight='bold')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save plots
    plots_dir = "results/plots"
    os.makedirs(plots_dir, exist_ok=True)
    filename = f"{plots_dir}/{processor.data.name.replace(' ', '_')}_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filename}")
    
    # Close the plot to free memory (no display needed with Agg backend)
    plt.close()
    print("Plot generated and saved successfully!")
    
    return filename

def print_detailed_analysis(processor, processor_name):
    """
    Display comprehensive text-based analysis that mirrors the graphical output.
    """
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS RESULTS: {processor_name}")
    print(f"{'='*80}")
    
    # 1. Physical Array Geometry
    print(f"\n1. PHYSICAL ARRAY GEOMETRY:")
    print(f"   {'─'*40}")
    positions = np.array(processor.data.sensors_positions)
    print(f"   Sensor positions: {positions}")
    print(f"   Number of sensors: {processor.data.num_sensors}")
    print(f"   Sensor spacing: {processor.data.sensor_spacing}")
    print(f"   Array span: {positions.max() - positions.min()}")
    
    # 2. Coarray Analysis
    print(f"\n2. COARRAY ANALYSIS:")
    print(f"   {'─'*40}")
    if processor.data.coarray_positions is not None:
        coarray_pos = processor.data.coarray_positions
        physical_set = set(processor.data.physical_positions)
        virtual_only = processor.data.virtual_only_positions
        
        print(f"   Total coarray elements: {len(coarray_pos)}")
        print(f"   Coarray positions: {coarray_pos}")
        print(f"   Physical positions: {processor.data.physical_positions}")
        print(f"   Virtual-only positions: {virtual_only}")
        print(f"   Virtual-only count: {len(virtual_only)}")
        print(f"   Coarray aperture: {coarray_pos.max() - coarray_pos.min()}")
    
    # 3. Weight Distribution
    print(f"\n3. WEIGHT DISTRIBUTION:")
    print(f"   {'─'*40}")
    if processor.data.weight_table is not None:
        weights_df = processor.data.weight_table
        print("   Lag | Weight | Sensor Pairs")
        print("   ----+--------+-------------")
        for _, row in weights_df.iterrows():
            lag = int(row.iloc[0])
            weight = int(row.iloc[1])
            pairs = row.iloc[2] if len(row) > 2 else "N/A"
            print(f"   {lag:3d} | {weight:6d} | {pairs}")
    
    # 4. Contiguous Segments
    print(f"\n4. CONTIGUOUS SEGMENTS ANALYSIS:")
    print(f"   {'─'*40}")
    if processor.data.largest_contiguous_segment is not None:
        segment = processor.data.largest_contiguous_segment
        print(f"   Largest contiguous segment: {segment}")
        print(f"   Segment length (L): {len(segment)}")
        print(f"   Max detectable sources (K_max): {processor.data.max_detectable_sources}")
        if processor.data.all_contiguous_segments is not None:
            print(f"   Total number of segments: {len(processor.data.all_contiguous_segments)}")
    
    # 5. Holes Analysis
    print(f"\n5. HOLES ANALYSIS:")
    print(f"   {'─'*40}")
    print(f"   Number of holes: {processor.data.num_holes}")
    if processor.data.num_holes > 0 and processor.data.missing_virtual_positions is not None:
        print(f"   Missing positions: {processor.data.missing_virtual_positions}")
    else:
        print(f"   Missing positions: None (Perfect contiguous array)")
    
    # 6. Performance Summary
    print(f"\n6. PERFORMANCE SUMMARY:")
    print(f"   {'─'*40}")
    if processor.data.performance_summary_table is not None:
        perf_table = processor.data.performance_summary_table
        for _, row in perf_table.iterrows():
            metric = row['Metrics']
            value = row['Value']
            print(f"   {metric:<35}: {value}")
    
    print(f"\n{'='*80}")

def main():
    """
    Main function for graphical array analysis demonstration.
    """
    print("MIMO ARRAY GEOMETRY ANALYSIS - GRAPHICAL DEMO")
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
    print("\nSelect processor for graphical analysis:")
    selection = input("Your choice: ").strip().lower()
    
    # Handle exit
    if selection in exit_options:
        print("Exiting GRAPHICAL DEMO. Goodbye!")
        return
    
    # Process selection
    if selection in processors:
        processor_class, processor_name = processors[selection]
        
        print(f"\nRunning analysis for: {processor_name}")
        
        try:
            # Create processor with appropriate parameters
            if processor_class == ULArrayProcessor:
                processor = ULArrayProcessor(M=5, d=1)
            elif processor_class == NestedArrayProcessor:
                processor = NestedArrayProcessor(N1=2, N2=3, d=1)
            elif processor_class == Z1ArrayProcessor:
                processor = Z1ArrayProcessor(N=5, d=1)
            elif processor_class == Z3_1ArrayProcessor:
                processor = Z3_1ArrayProcessor(N=5, d=1)
            elif processor_class == Z3_2ArrayProcessor:
                processor = Z3_2ArrayProcessor(N=5, d=1)
            elif processor_class == Z4ArrayProcessor:
                processor = Z4ArrayProcessor(N=5, d=1)
            elif processor_class == Z5ArrayProcessor:
                processor = Z5ArrayProcessor(N=5, d=1)
            elif processor_class == Z6ArrayProcessor:
                processor = Z6ArrayProcessor(N=5, d=1)
            else:
                raise ValueError(f"Unknown processor class: {processor_class}")
            
            # Run full analysis
            print("Executing full analysis pipeline...")
            results = processor.run_full_analysis()
            
            # Display comprehensive text-based analysis
            print_detailed_analysis(processor, processor_name)
            
            # Generate graphical visualization
            plot_filename = plot_array_analysis(processor, processor_name)
            
            print(f"\n{'='*60}")
            print(f"ANALYSIS COMPLETE!")
            print(f"{'='*60}")
            print(f"Processor: {results.name}")
            print(f"Total sensors: {results.num_sensors}")
            print(f"Max detectable sources: {results.max_detectable_sources}")
            print(f"Graphical analysis saved to: {plot_filename}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
    else:
        print("Invalid selection. Please try again.")

if __name__ == "__main__":
    main()