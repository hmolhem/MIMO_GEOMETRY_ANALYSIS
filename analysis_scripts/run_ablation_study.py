"""
Parameter Ablation Study for ALSS
Validates "no tuning required" claim by testing parameter sensitivity
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

def generate_steering_vector(N, theta, d=0.5):
    """Generate steering vector for ULA"""
    n = np.arange(N)
    return np.exp(-1j * 2 * np.pi * d * n * np.sin(np.radians(theta)))

def apply_mutual_coupling(A, c1=0.3, alpha=0.5):
    """Apply exponential mutual coupling model"""
    N = A.shape[0]
    C = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            C[i, j] = c1 * (alpha ** abs(i - j))
    return C @ A

def coarray_music_z3_2(sensor_positions, theta_true, M, snr_db, 
                       use_mcm=False, use_alss=False, 
                       w_min=3, beta=1.0, mode='adaptive', sigma_L=2.0):
    """
    Coarray MUSIC for Z3_2 array with ALSS options
    
    Parameters:
    -----------
    sensor_positions : array - Z3_2 sensor positions [0, 5, 6, 11, 12, 16]
    theta_true : array - True DOA angles in degrees
    M : int - Number of snapshots
    snr_db : float - SNR in dB
    use_mcm : bool - Apply mutual coupling
    use_alss : bool - Apply ALSS regularization
    w_min : int - Minimum weight threshold for L_0
    beta : float - Regularization strength parameter
    mode : str - 'zero', 'soft', or 'adaptive'
    sigma_L : float - Soft mode parameter
    """
    
    N = len(sensor_positions)
    K = len(theta_true)
    
    # Generate steering matrix
    A = np.zeros((N, K), dtype=complex)
    for k, theta in enumerate(theta_true):
        for i, pos in enumerate(sensor_positions):
            A[i, k] = np.exp(-1j * 2 * np.pi * pos * 0.5 * np.sin(np.radians(theta)))
    
    # Apply mutual coupling if requested
    if use_mcm:
        A = apply_mutual_coupling(A, c1=0.3, alpha=0.5)
    
    # Generate signals
    snr_linear = 10 ** (snr_db / 10)
    signal_power = 1.0
    noise_power = signal_power / snr_linear
    
    S = np.sqrt(signal_power) * (np.random.randn(K, M) + 1j * np.random.randn(K, M)) / np.sqrt(2)
    N_noise = np.sqrt(noise_power) * (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(2)
    X = A @ S + N_noise
    
    # Sample covariance
    R = (X @ X.conj().T) / M
    
    # Compute difference coarray
    differences = []
    weights = {}
    for i in range(N):
        for j in range(N):
            diff = sensor_positions[i] - sensor_positions[j]
            differences.append(diff)
            weights[diff] = weights.get(diff, 0) + 1
    
    # Apply ALSS if requested
    if use_alss:
        # Estimate noise variance
        eigenvalues = np.linalg.eigvalsh(R)
        sigma_n2 = np.min(eigenvalues)
        
        # Compute L_0
        L_0 = max([lag for lag in weights.keys() if weights[lag] >= w_min])
        
        # Compute tau
        tau = beta / (snr_linear * np.sqrt(M))
        
        # Create modified R with ALSS
        R_alss = R.copy()
        for i in range(N):
            for j in range(N):
                lag = sensor_positions[i] - sensor_positions[j]
                if abs(lag) > L_0:
                    # Estimate variance for this lag
                    w_lag = weights.get(lag, 1)
                    V_hat = sigma_n2 / (w_lag * M)
                    
                    # Compute retention factor based on mode
                    if mode == 'zero':
                        alpha_lag = 0.0
                    elif mode == 'soft':
                        alpha_lag = np.exp(-((abs(lag) - L_0)**2) / (2 * sigma_L**2))
                    else:  # adaptive
                        alpha_lag = 1 / (1 + tau * V_hat)
                    
                    R_alss[i, j] *= alpha_lag
        
        R = R_alss
    
    # Construct virtual Toeplitz matrix
    unique_lags = sorted(set(differences))
    virtual_size = len(unique_lags)
    R_v = np.zeros((virtual_size, virtual_size), dtype=complex)
    
    lag_to_idx = {lag: idx for idx, lag in enumerate(unique_lags)}
    
    for m in range(virtual_size):
        for n in range(virtual_size):
            lag_diff = unique_lags[m] - unique_lags[n]
            if lag_diff in weights:
                count = 0
                sum_val = 0
                for i in range(N):
                    for j in range(N):
                        if sensor_positions[i] - sensor_positions[j] == lag_diff:
                            sum_val += R[i, j]
                            count += 1
                if count > 0:
                    R_v[m, n] = sum_val / count
    
    # MUSIC algorithm
    eigenvalues, eigenvectors = np.linalg.eigh(R_v)
    noise_subspace = eigenvectors[:, :-K]
    
    # Spectrum search
    theta_search = np.linspace(-90, 90, 1801)
    spectrum = np.zeros(len(theta_search))
    
    for idx, theta in enumerate(theta_search):
        # Virtual steering vector
        a_v = np.zeros(virtual_size, dtype=complex)
        for i, lag in enumerate(unique_lags):
            a_v[i] = np.exp(-1j * 2 * np.pi * lag * 0.5 * np.sin(np.radians(theta)))
        
        a_v = a_v / np.linalg.norm(a_v)
        spectrum[idx] = 1 / (np.abs(a_v.conj() @ noise_subspace @ noise_subspace.conj().T @ a_v) + 1e-10)
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(spectrum, height=np.max(spectrum) * 0.3)
    
    # Select top K peaks
    if len(peaks) >= K:
        peaks = peaks[np.argsort(spectrum[peaks])[-K:]]
        theta_est = theta_search[sorted(peaks)]
    else:
        # If not enough peaks found, use highest K points
        top_indices = np.argsort(spectrum)[-K:]
        theta_est = theta_search[sorted(top_indices)]
    
    return theta_est

def run_ablation_study():
    """Run complete ablation study"""
    
    print("="*70)
    print("ALSS Parameter Ablation Study")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Z3_2 array configuration
    sensor_positions = np.array([0, 5, 6, 11, 12, 16])
    theta_true = np.array([-30, 0, 30])
    M = 200
    snr_db = 10
    num_trials = 50
    
    results = []
    
    # Baseline (no ALSS)
    print("Running baseline (no ALSS)...")
    for condition in ['no_mcm', 'with_mcm']:
        use_mcm = (condition == 'with_mcm')
        rmse_list = []
        
        for trial in range(num_trials):
            theta_est = coarray_music_z3_2(sensor_positions, theta_true, M, snr_db,
                                          use_mcm=use_mcm, use_alss=False)
            rmse = np.sqrt(np.mean((np.sort(theta_est) - theta_true)**2))
            rmse_list.append(rmse)
        
        results.append({
            'Configuration': 'Baseline (no ALSS)',
            'Condition': condition,
            'RMSE_mean': np.mean(rmse_list),
            'RMSE_std': np.std(rmse_list)
        })
    
    # L_0 ablation (w_min variations)
    print("Running L_0 threshold ablation...")
    for w_min in [2, 3, 4]:
        for condition in ['no_mcm', 'with_mcm']:
            use_mcm = (condition == 'with_mcm')
            rmse_list = []
            
            for trial in range(num_trials):
                theta_est = coarray_music_z3_2(sensor_positions, theta_true, M, snr_db,
                                              use_mcm=use_mcm, use_alss=True, w_min=w_min)
                rmse = np.sqrt(np.mean((np.sort(theta_est) - theta_true)**2))
                rmse_list.append(rmse)
            
            config_name = f"w_min={w_min}" + (" (default)" if w_min == 3 else "")
            results.append({
                'Configuration': config_name,
                'Condition': condition,
                'RMSE_mean': np.mean(rmse_list),
                'RMSE_std': np.std(rmse_list)
            })
    
    # Tau ablation (beta variations)
    print("Running regularization strength ablation...")
    for beta in [0.5, 1.0, 2.0]:
        for condition in ['no_mcm', 'with_mcm']:
            use_mcm = (condition == 'with_mcm')
            rmse_list = []
            
            for trial in range(num_trials):
                theta_est = coarray_music_z3_2(sensor_positions, theta_true, M, snr_db,
                                              use_mcm=use_mcm, use_alss=True, beta=beta)
                rmse = np.sqrt(np.mean((np.sort(theta_est) - theta_true)**2))
                rmse_list.append(rmse)
            
            config_name = f"beta={beta}" + (" (default)" if beta == 1.0 else "")
            results.append({
                'Configuration': config_name,
                'Condition': condition,
                'RMSE_mean': np.mean(rmse_list),
                'RMSE_std': np.std(rmse_list)
            })
    
    # Mode ablation
    print("Running operational mode ablation...")
    for mode in ['zero', 'soft', 'adaptive']:
        for condition in ['no_mcm', 'with_mcm']:
            use_mcm = (condition == 'with_mcm')
            rmse_list = []
            
            for trial in range(num_trials):
                theta_est = coarray_music_z3_2(sensor_positions, theta_true, M, snr_db,
                                              use_mcm=use_mcm, use_alss=True, mode=mode)
                rmse = np.sqrt(np.mean((np.sort(theta_est) - theta_true)**2))
                rmse_list.append(rmse)
            
            mode_name = f"{mode.capitalize()}-Mode"
            if mode == 'adaptive':
                mode_name += " (default)"
            
            results.append({
                'Configuration': mode_name,
                'Condition': condition,
                'RMSE_mean': np.mean(rmse_list),
                'RMSE_std': np.std(rmse_list)
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Pivot for display
    pivot_df = df.pivot(index='Configuration', columns='Condition', values='RMSE_mean')
    
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(pivot_df.to_string())
    print()
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'summaries')
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, 'ablation_study_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Print LaTeX table format
    print("\n" + "="*70)
    print("LATEX TABLE FORMAT")
    print("="*70)
    
    # Order configurations
    config_order = [
        'Baseline (no ALSS)',
        'w_min=2',
        'w_min=3 (default)',
        'w_min=4',
        'beta=0.5',
        'beta=1.0 (default)',
        'beta=2.0',
        'Zero-Mode',
        'Soft-Mode',
        'Adaptive-Mode (default)'
    ]
    
    for config in config_order:
        row_data = df[df['Configuration'] == config]
        if len(row_data) > 0:
            no_mcm = row_data[row_data['Condition'] == 'no_mcm']['RMSE_mean'].values[0]
            with_mcm = row_data[row_data['Condition'] == 'with_mcm']['RMSE_mean'].values[0]
            
            # Format configuration name for LaTeX
            latex_config = config.replace('_', r'\_').replace('=', r'=')
            print(f"{latex_config} & {no_mcm:.2f}° & {with_mcm:.2f}° \\\\")
    
    print()
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df

if __name__ == "__main__":
    df_results = run_ablation_study()
