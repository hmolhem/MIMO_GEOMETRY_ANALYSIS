# tests/test_z4.py
import numpy as np
from geometry_processors.z4_processor import Z4ArrayProcessor

def test_z4_n7():
    p = Z4ArrayProcessor(N=7, d=1.0)
    d = p.run_full_analysis()

    w = d.weight_dict
    assert w.get(1,0) == 0
    assert w.get(2,0) == 0
    assert w.get(3,0) > 0

    A = 3*7 - 7  # 14
    seg = d.largest_contiguous_segment
    assert seg[0] == 3 and seg[-1] == A
    assert len(seg) == 12
    assert int(d.max_detectable_sources) == 6

    holes = set(np.asarray(d.missing_virtual_positions_below_A, int).tolist())
    assert holes == {1, 2}
