"""
Comprehensive Array Comparison Script
=====================================

Compare all implemented MIMO array geometries side-by-side to analyze:
- Coarray aperture vs. sensor count
- Degrees of freedom (K_max) efficiency
- Weight distribution characteristics
- Holes and contiguous segment properties

This script generates comparison tables and visualizations for:
- Week 1 (w(1)=0): MISC, CADiS, cMRA
- Week 2 (Nested): SNA3, ANAII-2, DNA, DDNA

Usage:
------
python analysis_scripts/compare_all_arrays.py --N 7 --save-csv --markdown

Author: Hossein (RadarPy Project)
Date: November 7, 2025
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_processors.misc_processor import MISCArrayProcessor
from geometry_processors.cadis_processor import CADiSArrayProcessor
from geometry_processors.cmra_processor import cMRAArrayProcessor
from geometry_processors.sna_processor import SNA3ArrayProcessor
from geometry_processors.anaii2_processor import ANAII2ArrayProcessor
from geometry_processors.dna_processor import DNAArrayProcessor
from geometry_processors.ddna_processor import DDNAArrayProcessor
from geometry_processors.nested_processor import NestedArrayProcessor


def compare_all_arrays(N: int, d: float = 1.0, verbose: bool = False):
    """
    Compare all implemented array types with approximately N sensors.
    
    Parameters
    ----------
    N : int
        Target number of sensors (arrays will be configured to match as closely as possible)
    d : float
        Base spacing multiplier
    verbose : bool
        If True, print detailed analysis for each array
    
    Returns
    -------
    pd.DataFrame
        Comparison table with key metrics for all arrays
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ARRAY COMPARISON (Target N ≈ {N})")
    print(f"{'='*80}\n")
    
    # Week 1: w(1)=0 Arrays
    print("Week 1 - Arrays with w(1)=0 (Special Coarray Properties):")
    print("-" * 80)
    
    # 1. MISC Array (requires N ≥ 4)
    if N >= 4:
        try:
            misc = MISCArrayProcessor(N=N, d=d)
            misc_data = misc.run_full_analysis()
            
            if verbose:
                print(f"\nMISC Array (N={misc.total_sensors}):")
                print(f"  Positions: {misc_data.sensors_positions}")
                print(f"  Aperture: {misc_data.coarray_aperture}, L: {misc_data.segment_length}, K_max: {misc_data.segment_length // 2}")
            
            wt = misc_data.weight_table
            w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
            w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
            
            results.append({
                'Array': 'MISC',
                'Category': 'w(1)=0',
                'N': misc.total_sensors,
                'Aperture': misc_data.coarray_aperture,
                'Unique_Lags': len(misc_data.unique_differences),
                'L': misc_data.segment_length,
                'K_max': misc_data.segment_length // 2,
                'Holes': len(misc_data.holes_in_segment) if hasattr(misc_data, 'holes_in_segment') else 0,
                'w(1)': w1,
                'w(2)': w2,
                'DOF_Efficiency': f"{(misc_data.segment_length // 2) / misc.total_sensors:.2f}",
                'Notes': 'Minimizes small-lag coupling'
            })
            print(f"✓ MISC: N={misc.total_sensors}, A={misc_data.coarray_aperture}, L={misc_data.segment_length}, K_max={misc_data.segment_length // 2}")
        except Exception as e:
            print(f"✗ MISC: Error - {e}")
    
    # 2. CADiS Array (requires N ≥ 3)
    if N >= 3:
        try:
            cadis = CADiSArrayProcessor(N=N, d=d)
            cadis_data = cadis.run_full_analysis()
            
            if verbose:
                print(f"\nCADiS Array (N={cadis.total_sensors}):")
                print(f"  Positions: {cadis_data.sensors_positions}")
                print(f"  Aperture: {cadis_data.coarray_aperture}, L: {cadis_data.segment_length}")
            
            wt = cadis_data.weight_table
            w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
            w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
            
            results.append({
                'Array': 'CADiS',
                'Category': 'w(1)=0',
                'N': cadis.total_sensors,
                'Aperture': cadis_data.coarray_aperture,
                'Unique_Lags': len(cadis_data.unique_differences),
                'L': cadis_data.segment_length,
                'K_max': cadis_data.segment_length // 2,
                'Holes': len(cadis_data.holes_in_segment) if hasattr(cadis_data, 'holes_in_segment') else 0,
                'w(1)': w1,
                'w(2)': w2,
                'DOF_Efficiency': f"{(cadis_data.segment_length // 2) / cadis.total_sensors:.2f}",
                'Notes': 'Difference coarray-based'
            })
            print(f"✓ CADiS: N={cadis.total_sensors}, A={cadis_data.coarray_aperture}, L={cadis_data.segment_length}, K_max={cadis_data.segment_length // 2}")
        except Exception as e:
            print(f"✗ CADiS: Error - {e}")
    
    # 3. cMRA Array (requires N ≥ 3)
    if N >= 3:
        try:
            cmra = cMRAArrayProcessor(N=N, d=d)
            cmra_data = cmra.run_full_analysis()
            
            if verbose:
                print(f"\ncMRA Array (N={cmra.total_sensors}):")
                print(f"  Positions: {cmra_data.sensors_positions}")
                print(f"  Aperture: {cmra_data.coarray_aperture}, L: {cmra_data.segment_length}")
            
            wt = cmra_data.weight_table
            w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
            w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
            
            results.append({
                'Array': 'cMRA',
                'Category': 'w(1)=0',
                'N': cmra.total_sensors,
                'Aperture': cmra_data.coarray_aperture,
                'Unique_Lags': len(cmra_data.unique_differences),
                'L': cmra_data.segment_length,
                'K_max': cmra_data.segment_length // 2,
                'Holes': len(cmra_data.holes_in_segment) if hasattr(cmra_data, 'holes_in_segment') else 0,
                'w(1)': w1,
                'w(2)': w2,
                'DOF_Efficiency': f"{(cmra_data.segment_length // 2) / cmra.total_sensors:.2f}",
                'Notes': 'Coprime-like with MRA'
            })
            print(f"✓ cMRA: N={cmra.total_sensors}, A={cmra_data.coarray_aperture}, L={cmra_data.segment_length}, K_max={cmra_data.segment_length // 2}")
        except Exception as e:
            print(f"✗ cMRA: Error - {e}")
    
    # Week 2: Nested Arrays
    print(f"\n{'='*80}")
    print("Week 2 - Nested Array Variants:")
    print("-" * 80)
    
    # Determine N1, N2 split for nested arrays (approximately N/2 each)
    N1 = max(2, N // 2)
    N2 = max(2, N - N1)
    
    # 4. Standard Nested Array (baseline)
    try:
        nested = NestedArrayProcessor(N1=N1, N2=N2, d=d)
        nested_data = nested.run_full_analysis()
        
        if verbose:
            print(f"\nNested Array (N1={N1}, N2={N2}, N={nested.N1 + nested.N2}):")
            print(f"  Positions: {nested_data.sensors_positions}")
        
        # Get segment length (nested uses different attribute)
        seg_len = getattr(nested_data, 'segment_length', None)
        if seg_len is None:
            largest_seg = getattr(nested_data, 'largest_contiguous_segment', None)
            if largest_seg is not None and len(largest_seg) > 0:
                seg_len = int(largest_seg[-1] - largest_seg[0] + 1) if isinstance(largest_seg, np.ndarray) else 0
            else:
                seg_len = 0
        
        wt = nested_data.coarray_weight_distribution if hasattr(nested_data, 'coarray_weight_distribution') else pd.DataFrame()
        w1 = int(wt[wt.index == 1].values[0]) if (len(wt) > 0 and 1 in wt.index) else 0
        w2 = int(wt[wt.index == 2].values[0]) if (len(wt) > 0 and 2 in wt.index) else 0
        
        results.append({
            'Array': 'Nested',
            'Category': 'Nested (Baseline)',
            'N': nested.N1 + nested.N2,
            'Aperture': getattr(nested_data, 'aperture', 0),
            'Unique_Lags': len(nested_data.unique_differences) if hasattr(nested_data, 'unique_differences') else 0,
            'L': seg_len,
            'K_max': seg_len // 2,
            'Holes': getattr(nested_data, 'num_holes', 0),
            'w(1)': w1,
            'w(2)': w2,
            'DOF_Efficiency': f"{(seg_len // 2) / (nested.N1 + nested.N2):.2f}" if seg_len > 0 else "0.00",
            'Notes': 'Classic O(N²) DOF'
        })
        print(f"✓ Nested: N={nested.N1 + nested.N2}, A={getattr(nested_data, 'aperture', 0)}, L={seg_len}, K_max={seg_len // 2}")
    except Exception as e:
        print(f"✗ Nested: Error - {e}")
    
    # 5. SNA3 Array
    try:
        sna3 = SNA3ArrayProcessor(N=N, d=d)
        sna3_data = sna3.run_full_analysis()
        
        if verbose:
            print(f"\nSNA3 Array (N={sna3.total_sensors}):")
            print(f"  Positions: {sna3_data.sensors_positions}")
        
        wt = sna3_data.weight_table
        w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
        w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
        
        results.append({
            'Array': 'SNA3',
            'Category': 'Nested (Three-subarray)',
            'N': sna3.total_sensors,
            'Aperture': sna3_data.coarray_aperture,
            'Unique_Lags': len(sna3_data.unique_differences),
            'L': sna3_data.segment_length,
            'K_max': sna3_data.segment_length // 2,
            'Holes': len(sna3_data.holes_in_segment) if hasattr(sna3_data, 'holes_in_segment') else 0,
            'w(1)': w1,
            'w(2)': w2,
            'DOF_Efficiency': f"{(sna3_data.segment_length // 2) / sna3.total_sensors:.2f}",
            'Notes': 'Super nested variant'
        })
        print(f"✓ SNA3: N={sna3.total_sensors}, A={sna3_data.coarray_aperture}, L={sna3_data.segment_length}, K_max={sna3_data.segment_length // 2}")
    except Exception as e:
        print(f"✗ SNA3: Error - {e}")
    
    # 6. ANAII-2 Array
    try:
        # ANAII-2 requires specific N values, find closest
        anaii2_configs = {7: (2,2), 10: (3,4), 14: (5,6), 18: (7,8)}
        closest_n = min(anaii2_configs.keys(), key=lambda x: abs(x - N))
        N1_a, N2_a = anaii2_configs[closest_n]
        
        anaii2 = ANAII2ArrayProcessor(N1=N1_a, N2=N2_a, d=d)
        anaii2_data = anaii2.run_full_analysis()
        
        if verbose:
            print(f"\nANAII-2 Array (N1={N1_a}, N2={N2_a}, N={anaii2.total_sensors}):")
            print(f"  Positions: {anaii2_data.sensors_positions}")
        
        wt = anaii2_data.weight_table
        w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
        w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
        
        results.append({
            'Array': 'ANAII-2',
            'Category': 'Nested (Three-subarray)',
            'N': anaii2.total_sensors,
            'Aperture': anaii2_data.coarray_aperture,
            'Unique_Lags': len(anaii2_data.unique_differences),
            'L': anaii2_data.segment_length,
            'K_max': anaii2_data.segment_length // 2,
            'Holes': len(anaii2_data.holes_in_segment) if hasattr(anaii2_data, 'holes_in_segment') else 0,
            'w(1)': w1,
            'w(2)': w2,
            'DOF_Efficiency': f"{(anaii2_data.segment_length // 2) / anaii2.total_sensors:.2f}",
            'Notes': 'Augmented nested II-2'
        })
        print(f"✓ ANAII-2: N={anaii2.total_sensors}, A={anaii2_data.coarray_aperture}, L={anaii2_data.segment_length}, K_max={anaii2_data.segment_length // 2}")
    except Exception as e:
        print(f"✗ ANAII-2: Error - {e}")
    
    # 7. DNA Array (D=1: standard nested, D=2: dilated)
    try:
        dna1 = DNAArrayProcessor(N1=N1, N2=N2, D=1, d=d)
        dna1_data = dna1.run_full_analysis()
        
        wt = dna1_data.weight_table
        w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
        w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
        
        results.append({
            'Array': 'DNA (D=1)',
            'Category': 'Nested (Dilated)',
            'N': dna1.total_sensors,
            'Aperture': dna1_data.coarray_aperture,
            'Unique_Lags': len(dna1_data.unique_differences),
            'L': dna1_data.segment_length,
            'K_max': dna1_data.segment_length // 2,
            'Holes': len(dna1_data.holes_in_segment) if hasattr(dna1_data, 'holes_in_segment') else 0,
            'w(1)': w1,
            'w(2)': w2,
            'DOF_Efficiency': f"{(dna1_data.segment_length // 2) / dna1.total_sensors:.2f}",
            'Notes': 'Standard nested (D=1)'
        })
        print(f"✓ DNA (D=1): N={dna1.total_sensors}, A={dna1_data.coarray_aperture}, L={dna1_data.segment_length}, K_max={dna1_data.segment_length // 2}")
        
        # DNA with D=2
        dna2 = DNAArrayProcessor(N1=N1, N2=N2, D=2, d=d)
        dna2_data = dna2.run_full_analysis()
        
        wt = dna2_data.weight_table
        w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
        w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
        
        results.append({
            'Array': 'DNA (D=2)',
            'Category': 'Nested (Dilated)',
            'N': dna2.total_sensors,
            'Aperture': dna2_data.coarray_aperture,
            'Unique_Lags': len(dna2_data.unique_differences),
            'L': dna2_data.segment_length,
            'K_max': dna2_data.segment_length // 2,
            'Holes': len(dna2_data.holes_in_segment) if hasattr(dna2_data, 'holes_in_segment') else 0,
            'w(1)': w1,
            'w(2)': w2,
            'DOF_Efficiency': f"{(dna2_data.segment_length // 2) / dna2.total_sensors:.2f}",
            'Notes': 'Dilated (reduced coupling)'
        })
        print(f"✓ DNA (D=2): N={dna2.total_sensors}, A={dna2_data.coarray_aperture}, L={dna2_data.segment_length}, K_max={dna2_data.segment_length // 2}")
    except Exception as e:
        print(f"✗ DNA: Error - {e}")
    
    # 8. DDNA Array (various D1, D2 combinations)
    try:
        # DDNA (D1=1, D2=1): standard nested
        ddna11 = DDNAArrayProcessor(N1=N1, N2=N2, D1=1, D2=1, d=d)
        ddna11_data = ddna11.run_full_analysis()
        
        wt = ddna11_data.weight_table
        w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
        w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
        
        results.append({
            'Array': 'DDNA (1,1)',
            'Category': 'Nested (Double-dilated)',
            'N': ddna11.total_sensors,
            'Aperture': ddna11_data.coarray_aperture,
            'Unique_Lags': len(ddna11_data.unique_differences),
            'L': ddna11_data.segment_length,
            'K_max': ddna11_data.segment_length // 2,
            'Holes': len(ddna11_data.holes_in_segment) if hasattr(ddna11_data, 'holes_in_segment') else 0,
            'w(1)': w1,
            'w(2)': w2,
            'DOF_Efficiency': f"{(ddna11_data.segment_length // 2) / ddna11.total_sensors:.2f}",
            'Notes': 'Standard nested (D1=D2=1)'
        })
        print(f"✓ DDNA (1,1): N={ddna11.total_sensors}, A={ddna11_data.coarray_aperture}, L={ddna11_data.segment_length}, K_max={ddna11_data.segment_length // 2}")
        
        # DDNA (D1=1, D2=2): DNA-equivalent
        ddna12 = DDNAArrayProcessor(N1=N1, N2=N2, D1=1, D2=2, d=d)
        ddna12_data = ddna12.run_full_analysis()
        
        wt = ddna12_data.weight_table
        w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
        w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
        
        results.append({
            'Array': 'DDNA (1,2)',
            'Category': 'Nested (Double-dilated)',
            'N': ddna12.total_sensors,
            'Aperture': ddna12_data.coarray_aperture,
            'Unique_Lags': len(ddna12_data.unique_differences),
            'L': ddna12_data.segment_length,
            'K_max': ddna12_data.segment_length // 2,
            'Holes': len(ddna12_data.holes_in_segment) if hasattr(ddna12_data, 'holes_in_segment') else 0,
            'w(1)': w1,
            'w(2)': w2,
            'DOF_Efficiency': f"{(ddna12_data.segment_length // 2) / ddna12.total_sensors:.2f}",
            'Notes': 'DNA-equivalent (D1=1, D2=2)'
        })
        print(f"✓ DDNA (1,2): N={ddna12.total_sensors}, A={ddna12_data.coarray_aperture}, L={ddna12_data.segment_length}, K_max={ddna12_data.segment_length // 2}")
        
        # DDNA (D1=2, D2=2): full double dilation
        ddna22 = DDNAArrayProcessor(N1=N1, N2=N2, D1=2, D2=2, d=d)
        ddna22_data = ddna22.run_full_analysis()
        
        wt = ddna22_data.weight_table
        w1 = int(wt[wt['Lag'] == 1]['Weight'].iloc[0]) if 1 in wt['Lag'].values else 0
        w2 = int(wt[wt['Lag'] == 2]['Weight'].iloc[0]) if 2 in wt['Lag'].values else 0
        
        results.append({
            'Array': 'DDNA (2,2)',
            'Category': 'Nested (Double-dilated)',
            'N': ddna22.total_sensors,
            'Aperture': ddna22_data.coarray_aperture,
            'Unique_Lags': len(ddna22_data.unique_differences),
            'L': ddna22_data.segment_length,
            'K_max': ddna22_data.segment_length // 2,
            'Holes': len(ddna22_data.holes_in_segment) if hasattr(ddna22_data, 'holes_in_segment') else 0,
            'w(1)': w1,
            'w(2)': w2,
            'DOF_Efficiency': f"{(ddna22_data.segment_length // 2) / ddna22.total_sensors:.2f}",
            'Notes': 'Full DDNA (minimal coupling)'
        })
        print(f"✓ DDNA (2,2): N={ddna22.total_sensors}, A={ddna22_data.coarray_aperture}, L={ddna22_data.segment_length}, K_max={ddna22_data.segment_length // 2}")
    except Exception as e:
        print(f"✗ DDNA: Error - {e}")
    
    # Create comparison DataFrame
    df = pd.DataFrame(results)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Compare all implemented MIMO array geometries')
    parser.add_argument('--N', type=int, default=7, help='Target number of sensors (default: 7)')
    parser.add_argument('--d', type=float, default=1.0, help='Base spacing multiplier (default: 1.0)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed analysis for each array')
    parser.add_argument('--markdown', action='store_true', help='Print comparison table in Markdown format')
    parser.add_argument('--save-csv', action='store_true', help='Save comparison table to CSV')
    parser.add_argument('--output-dir', type=str, default='results/comparisons', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run comparison
    comparison_df = compare_all_arrays(N=args.N, d=args.d, verbose=args.verbose)
    
    # Display results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    
    if args.markdown:
        print(comparison_df.to_markdown(index=False))
    else:
        print(comparison_df.to_string(index=False))
    
    # Analysis insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}\n")
    
    # Find best K_max
    best_kmax = comparison_df.loc[comparison_df['K_max'].idxmax()]
    print(f"✓ Best K_max (DOF): {best_kmax['Array']} with K_max={best_kmax['K_max']} from N={best_kmax['N']} sensors")
    
    # Find largest aperture
    best_aperture = comparison_df.loc[comparison_df['Aperture'].idxmax()]
    print(f"✓ Largest Aperture: {best_aperture['Array']} with A={best_aperture['Aperture']} from N={best_aperture['N']} sensors")
    
    # Find best DOF efficiency
    comparison_df['DOF_Efficiency_Val'] = comparison_df['DOF_Efficiency'].astype(float)
    best_efficiency = comparison_df.loc[comparison_df['DOF_Efficiency_Val'].idxmax()]
    print(f"✓ Best DOF Efficiency: {best_efficiency['Array']} with {best_efficiency['DOF_Efficiency']} K_max/N ratio")
    
    # Count arrays with w(1)=0
    w1_zero = comparison_df[comparison_df['w(1)'] == 0]
    print(f"✓ Arrays with w(1)=0: {len(w1_zero)} ({', '.join(w1_zero['Array'].tolist())})")
    
    # Count hole-free arrays
    hole_free = comparison_df[comparison_df['Holes'] == 0]
    print(f"✓ Hole-free arrays: {len(hole_free)}/{len(comparison_df)}")
    
    # Save to CSV if requested
    if args.save_csv:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f'array_comparison_N{args.N}.csv')
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✓ Comparison table saved to: {output_path}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
