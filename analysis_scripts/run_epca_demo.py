"""
ePCA (Extended Prime Coprime Array) Demonstration Script
=========================================================

Demonstrates the ePCA array processor with various prime triplet configurations.

Usage:
    python run_epca_demo.py --p1 2 --p2 3 --p3 5 --N1 3 --N2 3 --N3 3 --d 1.0 --markdown
"""

import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_processors.epca_processor import ePCAArrayProcessor, compare_epca_arrays


def main():
    parser = argparse.ArgumentParser(
        description="ePCA (Extended Prime Coprime Array) Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Small primes (default)
  python run_epca_demo.py --p1 2 --p2 3 --p3 5 --N1 3 --N2 3 --N3 3
  
  # Medium primes  
  python run_epca_demo.py --p1 3 --p2 5 --p3 7 --N1 2 --N2 2 --N3 2
  
  # Compare multiple prime triplets
  python run_epca_demo.py --compare
  
  # Save outputs
  python run_epca_demo.py --markdown --save-csv
        """
    )
    
    parser.add_argument("--p1", type=int, default=2, help="First prime (smallest)")
    parser.add_argument("--p2", type=int, default=3, help="Second prime (middle)")
    parser.add_argument("--p3", type=int, default=5, help="Third prime (largest)")
    parser.add_argument("--N1", type=int, default=3, help="Subarray 1 size")
    parser.add_argument("--N2", type=int, default=3, help="Subarray 2 size")
    parser.add_argument("--N3", type=int, default=3, help="Subarray 3 size")
    parser.add_argument("--d", type=float, default=1.0, help="Base spacing multiplier")
    parser.add_argument("--markdown", action="store_true", help="Print summary as Markdown table")
    parser.add_argument("--save-csv", action="store_true", help="Save summary to CSV")
    parser.add_argument("--save-json", action="store_true", help="Save configuration to JSON")
    parser.add_argument("--compare", action="store_true", help="Compare multiple prime triplets")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ePCA (Extended Prime Coprime Array) Demonstration")
    print("="*70 + "\n")
    
    if args.compare:
        print("Comparing multiple prime triplets...\n")
        prime_triplets = [
            (2, 3, 5),
            (3, 5, 7),
            (5, 7, 11),
            (7, 11, 13)
        ]
        
        comparison = compare_epca_arrays(
            prime_triplets,
            N1=args.N1,
            N2=args.N2,
            N3=args.N3,
            d=args.d
        )
        
        print("Comparison Results:")
        print("-------------------")
        if args.markdown:
            print(comparison.to_markdown(index=False))
        else:
            print(comparison.to_string(index=False))
        
        if args.save_csv:
            filename = f"../results/summaries/epca_comparison_N{args.N1}x{args.N2}x{args.N3}_d{args.d}.csv"
            comparison.to_csv(filename, index=False)
            print(f"\n✓ Saved comparison to {filename}")
    
    else:
        # Single configuration analysis
        print(f"Configuration: p1={args.p1}, p2={args.p2}, p3={args.p3}")
        print(f"Subarray sizes: N1={args.N1}, N2={args.N2}, N3={args.N3}")
        print(f"Base spacing: d={args.d}\n")
        
        # Create processor and run analysis
        epca = ePCAArrayProcessor(
            p1=args.p1,
            p2=args.p2,
            p3=args.p3,
            N1=args.N1,
            N2=args.N2,
            N3=args.N3,
            d=args.d
        )
        
        results = epca.run_full_analysis()
        
        print("\nPerformance Summary:")
        print("--------------------")
        if args.markdown:
            print(results.performance_summary_table.to_markdown(index=False))
        else:
            print(results.performance_summary_table.to_string(index=False))
        
        # Save outputs if requested
        if args.save_csv:
            filename = f"../results/summaries/epca_summary_p{args.p1}_{args.p2}_{args.p3}_N{args.N1}x{args.N2}x{args.N3}_d{args.d}.csv"
            results.performance_summary_table.to_csv(filename, index=False)
            print(f"\n✓ Saved summary to {filename}")
        
        if args.save_json:
            import json
            config = {
                "array_type": "ePCA",
                "primes": {"p1": args.p1, "p2": args.p2, "p3": args.p3},
                "subarray_sizes": {"N1": args.N1, "N2": args.N2, "N3": args.N3},
                "spacing": args.d,
                "coprime": epca.is_coprime,
                "total_sensors": epca.total_sensors,
                "aperture": int(results.aperture),
                "segment_length": int(results.segment_length),
                "K_max": int(results.segment_length // 2),
                "holes": len(results.holes_in_segment),
                "dof_efficiency": float(results.segment_length // 2) / epca.total_sensors
            }
            
            filename = f"../results/summaries/epca_config_p{args.p1}_{args.p2}_{args.p3}_N{args.N1}x{args.N2}x{args.N3}.json"
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✓ Saved configuration to {filename}")
    
    print("\n" + "="*70)
    print("✓ ePCA Analysis Complete")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
