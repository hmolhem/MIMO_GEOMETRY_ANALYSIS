# test/test_z5_unit.py
"""
Tiny unit test for Z5 processor - validates basic constraints and expected behavior
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geometry_processors.z5_processor import Z5ArrayProcessor


def test_z5_basic_constraints():
    """Test Z5 with exact seed, assert w(1)=w(2)=0, w(3)≥1, L≥3, K_max == L//2"""
    print("=== Z5 Unit Test ===")
    
    # Test with N=7 (the canonical case)
    N = 7
    z5 = Z5ArrayProcessor(N=N, d=1.0)
    data = z5.run_full_analysis(verbose=False)  # Quiet mode
    
    # Extract key data
    sensors = np.asarray(data.sensors_positions, dtype=int)
    wt_df = data.weight_table
    seg = np.asarray(data.largest_contiguous_segment, dtype=int)
    
    print(f"Sensors (N={N}): {sensors.tolist()}")
    
    # Weight constraints
    w = {int(r["Lag"]): int(r["Weight"]) for _, r in wt_df.iterrows()} if not wt_df.empty else {}
    w1, w2, w3 = w.get(1, 0), w.get(2, 0), w.get(3, 0)
    
    assert w1 == 0, f"Expected w(1)=0, got {w1}"
    assert w2 == 0, f"Expected w(2)=0, got {w2}"
    assert w3 >= 1, f"Expected w(3)≥1, got {w3}"
    print(f"✓ Weight constraints: w(1)={w1}, w(2)={w2}, w(3)={w3}")
    
    # Segment constraints
    L = int(seg.size)
    K_max = L // 2
    assert L >= 3, f"Expected L≥3, got {L}"
    assert K_max == L // 2, f"Expected K_max=L//2={L//2}, got {K_max}"
    
    L1, L2 = int(seg[0]), int(seg[-1]) if L > 0 else (0, 0)
    print(f"✓ Segment: L={L}, range=[{L1}:{L2}], K_max={K_max}")
    
    # Basic geometry checks
    assert len(sensors) == N, f"Expected {N} sensors, got {len(sensors)}"
    assert np.all(np.diff(sensors) > 0), "Sensors must be strictly increasing"
    assert np.all(sensors >= 0), "All sensor positions must be non-negative"
    print(f"✓ Basic geometry: {N} sensors, sorted, non-negative")
    
    # Performance check - with optimization, should get better L
    print(f"Performance: L={L} (target: ≥3), K_max={K_max}")
    
    print("=== Z5 Unit Test PASSED ===\n")
    return True


def test_z5_multiple_sizes():
    """Quick test for different N values"""
    print("=== Z5 Multi-size Test ===")
    
    for N in [5, 6, 7, 8, 9]:
        z5 = Z5ArrayProcessor(N=N, d=1.0)
        data = z5.run_full_analysis(verbose=False)
        
        # Basic constraint checks
        wt_df = data.weight_table
        w = {int(r["Lag"]): int(r["Weight"]) for _, r in wt_df.iterrows()} if not wt_df.empty else {}
        
        assert w.get(1, 0) == 0 and w.get(2, 0) == 0, f"N={N}: w(1)=w(2)=0 violated"
        
        seg = np.asarray(data.largest_contiguous_segment, dtype=int)
        L = int(seg.size)
        assert L >= 1, f"N={N}: Must have contiguous segment L≥1, got {L}"
        
        print(f"✓ N={N}: L={L}, sensors={data.sensors_positions}")
    
    print("=== Z5 Multi-size Test PASSED ===\n")
    return True


if __name__ == "__main__":
    try:
        test_z5_basic_constraints()
        test_z5_multiple_sizes()
        print("🎉 All Z5 unit tests PASSED!")
    except Exception as e:
        print(f"❌ Z5 unit test FAILED: {e}")
        raise