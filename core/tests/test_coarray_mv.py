# tests/test_coarray_mv.py
"""Unit tests for Mv (virtual array size) reporting in coarray processing."""
import numpy as np
import sys
from pathlib import Path

# Add project root to path (go up 2 levels from core/tests to project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.radarpy.algorithms.coarray import build_virtual_ula_covariance
from core.radarpy.geometry.z4_processor import Z4ArrayProcessor
from core.radarpy.geometry.z5_processor import Z5ArrayProcessor
from core.radarpy.geometry.ula_processors import ULArrayProcessor

def test_z4_mv_reporting():
    """Test that Z4 coarray correctly reports Mv for N=7."""
    # Z4 with N=7 should have contiguous segment around length 12
    z4 = Z4ArrayProcessor(N=7, d=1.0)
    z4.run_full_analysis(verbose=False)
    positions = np.asarray(z4.data.sensors_positions, dtype=int)
    
    # Create dummy covariance (identity for testing)
    N = len(positions)
    Rxx = np.eye(N, dtype=complex)
    
    # Build virtual covariance
    Rv, dvirt, (L1, L2), one_side, rmap, debug_info = build_virtual_ula_covariance(Rxx, positions, 1.0)
    
    # Verify debug info consistency
    assert "Mv" in debug_info, "debug_info must contain 'Mv'"
    assert "Lv" in debug_info, "debug_info must contain 'Lv'"
    assert debug_info["Mv"] == debug_info["Lv"], "Mv must equal Lv"
    assert debug_info["Mv"] >= 2, "Mv must be >= 2 for valid DOA"
    assert Rv.shape[0] == debug_info["Mv"], "Rv dimension must match Mv"
    assert debug_info["L2"] - debug_info["L1"] + 1 == debug_info["Lv"], "L2-L1+1 must equal Lv"
    
    print(f"✓ Z4(N=7): Mv={debug_info['Mv']}, Lv={debug_info['Lv']}, segment=[{L1}:{L2}]")
    assert debug_info["Mv"] >= 10, f"Z4(N=7) should have Mv>=10, got {debug_info['Mv']}"

def test_z5_mv_reporting():
    """Test that Z5 coarray correctly reports Mv for N=7."""
    z5 = Z5ArrayProcessor(N=7, d=1.0)
    z5.run_full_analysis(verbose=False)
    positions = np.asarray(z5.data.sensors_positions, dtype=int)
    
    N = len(positions)
    Rxx = np.eye(N, dtype=complex)
    
    Rv, dvirt, (L1, L2), one_side, rmap, debug_info = build_virtual_ula_covariance(Rxx, positions, 1.0)
    
    assert debug_info["Mv"] == debug_info["Lv"], "Mv must equal Lv"
    assert debug_info["Mv"] >= 2, "Mv must be >= 2 for valid DOA"
    assert Rv.shape[0] == debug_info["Mv"], "Rv dimension must match Mv"
    
    print(f"✓ Z5(N=7): Mv={debug_info['Mv']}, Lv={debug_info['Lv']}, segment=[{L1}:{L2}]")
    assert debug_info["Mv"] >= 8, f"Z5(N=7) should have Mv>=8, got {debug_info['Mv']}"

def test_ula_mv_reporting():
    """Test that ULA coarray correctly reports Mv for N=7."""
    ula = ULArrayProcessor(N=7, d=1.0)
    ula.run_full_analysis(verbose=False)
    positions = np.asarray(ula.data.sensors_positions, dtype=int)
    
    N = len(positions)
    Rxx = np.eye(N, dtype=complex)
    
    Rv, dvirt, (L1, L2), one_side, rmap, debug_info = build_virtual_ula_covariance(Rxx, positions, 1.0)
    
    assert debug_info["Mv"] == debug_info["Lv"], "Mv must equal Lv"
    assert debug_info["Mv"] >= 2, "Mv must be >= 2 for valid DOA"
    assert Rv.shape[0] == debug_info["Mv"], "Rv dimension must match Mv"
    
    print(f"✓ ULA(N=7): Mv={debug_info['Mv']}, Lv={debug_info['Lv']}, segment=[{L1}:{L2}]")
    assert debug_info["Mv"] == 7, f"ULA(N=7) should have Mv=7 (lags 0..6), got {debug_info['Mv']}"

if __name__ == "__main__":
    print("Testing Mv reporting in coarray processing...\n")
    
    try:
        test_z4_mv_reporting()
        test_z5_mv_reporting()
        test_ula_mv_reporting()
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
