import os
import sys
import glob
import importlib
import inspect
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry_processors.bases_classes import BaseArrayProcessor

def check_all():
    print("--- Checking All Geometry Processors ---")
    
    processors_dir = os.path.join(os.path.dirname(__file__), "..", "geometry_processors")
    files = glob.glob(os.path.join(processors_dir, "*_processor.py"))
    print(f"Found {len(files)} processor files.")
    
    results = []
    
    for f in files:
        module_name = os.path.basename(f).replace(".py", "")
        if module_name == "bases_classes":
            continue
            
        try:
            # Import module
            spec = importlib.util.spec_from_file_location(f"geometry_processors.{module_name}", f)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"geometry_processors.{module_name}"] = mod
            spec.loader.exec_module(mod)
            
            # Find processor class
            processor_cls = None
            for name, obj in inspect.getmembers(mod):
                if inspect.isclass(obj) and issubclass(obj, BaseArrayProcessor) and obj is not BaseArrayProcessor:
                    processor_cls = obj
                    break
            
            if processor_cls is None:
                print(f"Skipping {module_name}: No BaseArrayProcessor subclass found.")
                continue
            
            print(f"Testing {name}...")
                
            # Inspect constructor to find required args
            sig = inspect.signature(processor_cls.__init__)
            params = sig.parameters
            kwargs = {}
            
            # Special handling for N1/N2 processors (Nested, ANAII2, etc.)
            if "N1" in params and "N2" in params:
                kwargs["N1"] = 3
                kwargs["N2"] = 3
                # Don't pass N if it's not in params (ANAII2 doesn't take N)
                pass_N = "N" in params
            else:
                pass_N = True

            for pname, p in params.items():
                if pname in ["self", "N", "d", "N1", "N2"]:
                    continue
                if p.default == inspect.Parameter.empty:
                    # Required argument
                    if "alpha" in pname:
                        kwargs[pname] = 0.5
                    elif "beta" in pname:
                        kwargs[pname] = 0.5
                    elif "gamma" in pname:
                        kwargs[pname] = 0.5
                    elif "M" in pname:
                        kwargs[pname] = 2 # Subarrays
                    else:
                        kwargs[pname] = 1.0 # Fallback
            
            # Instantiate and Analyze
            try:
                if pass_N:
                    proc = processor_cls(N=7, d=1.0, **kwargs)
                else:
                    proc = processor_cls(d=1.0, **kwargs) # Uses N1=3, N2=3
            except Exception as e1:
                try:
                    if pass_N:
                        proc = processor_cls(N=8, d=1.0, **kwargs)
                    else:
                        proc = processor_cls(d=1.0, **kwargs) # Retry same?
                except Exception as e2:
                    print(f"Failed to instantiate {name}: {e1}")
                    continue
            
            proc.run_full_analysis(verbose=False)
            
            # Check properties
            lags = proc.data.coarray_positions
            if lags is None:
                print(f"Skipping {name}: No coarray positions.")
                continue
                
            lags = np.array(lags)
            pos_lags = lags[lags >= 0]
            
            # Check for holes at 1, 2
            has_hole_1 = 1 not in pos_lags
            has_hole_2 = 2 not in pos_lags
            
            # Contiguous segment
            seg = proc.data.largest_contiguous_segment
            L = len(seg) if seg is not None else 0
            
            results.append({
                "Processor": name,
                "N": proc.data.num_sensors, # Fixed attribute access
                "Hole @ 1": "YES" if has_hole_1 else "No",
                "Hole @ 2": "YES" if has_hole_2 else "No",
                "L (Contig)": L,
                "Max Lag": pos_lags.max() if len(pos_lags) > 0 else 0
            })
            
        except Exception as e:
            print(f"Error processing {module_name}: {e}")
            
    # Report
    if not results:
        print("\nNo processors were successfully checked.")
        return

    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("GEOMETRY HEALTH CHECK")
    print("="*60)
    print(df.to_string(index=False))
    
    # Summary for User
    print("\nSummary:")
    valid_count = len(df[(df["Hole @ 1"] == "No") & (df["Hole @ 2"] == "No")])
    print(f"Total Processors Checked: {len(df)}")
    print(f"Valid for Standard MUSIC (No low-lag holes): {valid_count}")

if __name__ == "__main__":
    check_all()
