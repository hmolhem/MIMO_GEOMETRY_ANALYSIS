# util/coarray.py
import numpy as np
from .alss import apply_alss

def build_virtual_ula_covariance(Rxx, positions, d_phys, *, 
                                 alss_enabled=False, alss_mode="zero",
                                 alss_tau=1.0, alss_coreL=3, M=None):
    """
    Build virtual ULA covariance from spatial covariance via lag averaging.
    
    Parameters:
    -----------
    Rxx : ndarray (N, N)
        Spatial sample covariance matrix
    positions : ndarray
        Integer-grid sensor indices
    d_phys : float
        Physical grid spacing (meters)
    
    Returns:
    --------
    Rv : ndarray (Lv, Lv)
        Virtual ULA covariance matrix
    dvirt : float
        Virtual array spacing (meters)
    (L1, L2) : tuple
        One-sided segment bounds
    one_side : ndarray
        Lags used in virtual array
    rmap : dict
        Lag -> list of (i,j) pairs mapping
    debug_info : dict
        Debug information with keys: L1, L2, Lv, Mv, lags_used, dvirt
    """
    positions = np.asarray(positions, dtype=int)
    N = len(positions)
    
    # Compute all difference lags
    lags_all = []
    lag_map = {}  # lag -> list of (i,j) pairs
    
    for i in range(N):
        for j in range(N):
            lag = int(positions[i] - positions[j])
            if lag not in lag_map:
                lag_map[lag] = []
            lag_map[lag].append((i, j))
    
    # Find all unique lags (full difference coarray)
    all_lags = sorted(lag_map.keys())
    
    # Find the longest contiguous segment of positive lags (including 0 if present)
    # For sparse arrays like Z4, this may not include lag 0
    pos_lags = [l for l in all_lags if l >= 0]
    
    if len(pos_lags) == 0:
        raise ValueError("No positive lags found in difference coarray")
    
    # Find longest contiguous segment
    # Priority: longest segment (prefer length), then containing lag 0 (DC), then lower start
    best_start = pos_lags[0]
    best_length = 1
    current_start = pos_lags[0]
    current_length = 1
    
    for i in range(1, len(pos_lags)):
        if pos_lags[i] == pos_lags[i-1] + 1:
            # Continue current segment
            current_length += 1
        else:
            # Gap found - check if current is best
            # Priority: length > contains_zero > lower_start
            current_has_zero = (current_start == 0)
            best_has_zero = (best_start == 0)
            
            if current_length > best_length or \
               (current_length == best_length and current_has_zero and not best_has_zero) or \
               (current_length == best_length and current_has_zero == best_has_zero and current_start < best_start):
                best_start = current_start
                best_length = current_length
            # Start new segment
            current_start = pos_lags[i]
            current_length = 1
    
    # Check final segment
    current_has_zero = (current_start == 0)
    best_has_zero = (best_start == 0)
    
    if current_length > best_length or \
       (current_length == best_length and current_has_zero and not best_has_zero) or \
       (current_length == best_length and current_has_zero == best_has_zero and current_start < best_start):
        best_start = current_start
        best_length = current_length
    
    # Use the longest contiguous segment
    one_side = np.arange(best_start, best_start + best_length)
    Lv = len(one_side)
    
    # Build virtual covariance via UNBIASED lag averaging
    # Key: divide by actual pair count w(l) for each lag, not uniform averaging
    
    # First compute lag-based autocorrelation estimates
    r_lag_dict = {}
    w_lag_dict = {}
    
    for lag in lag_map.keys():
        pairs = lag_map[lag]
        w_lag_dict[lag] = len(pairs)
        r_lag_dict[lag] = np.sum([Rxx[i, j] for i, j in pairs]) / len(pairs)
    
    # --- ALSS (optional) ---
    if alss_enabled:
        if M is None:
            raise ValueError("ALSS enabled but snapshots M not provided.")
        r_lag_dict = apply_alss(
            r_lag=r_lag_dict,
            w_lag=w_lag_dict,
            R_x=Rxx,
            M=int(M),
            mode=alss_mode,
            tau=float(alss_tau),
            coreL=int(alss_coreL),
        )
    
    # Now build Toeplitz matrix from processed lags
    Rv = np.zeros((Lv, Lv), dtype=complex)
    
    for m in range(Lv):
        for n in range(Lv):
            lag = int(one_side[m] - one_side[n])  # actual lag value (not index)
            if lag in r_lag_dict:
                Rv[m, n] = r_lag_dict[lag]
    
    dvirt = d_phys  # virtual spacing same as physical
    L1, L2 = best_start, best_start + best_length - 1
    Mv = Lv  # Virtual array size (number of contiguous lags)
    
    # Build debug info dictionary
    debug_info = {
        "L1": int(L1),
        "L2": int(L2),
        "Lv": int(Lv),
        "Mv": int(Mv),
        "lags_used": list(map(int, one_side)),
        "dvirt": float(dvirt)
    }
    
    return Rv, dvirt, (L1, L2), one_side, lag_map, debug_info
