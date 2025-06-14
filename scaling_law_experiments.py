import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
from scipy.optimize import minimize
from tqdm import tqdm
from itertools import product

# Suppress RuntimeWarnings that can occur with extreme parameter values during optimization
warnings.filterwarnings('ignore', category=RuntimeWarning)

class TissueMomentumLawOfficial:
    """
    Official Implementation of Tissue et al., 2024 Momentum Law based on official code:
    L(s) = L₀ + A × S₁^(-α) - C × S₂
    """
    def __init__(self, lambda_decay=0.999):
        self.lambda_decay = lambda_decay
        self.params = None  # L₀, A, C, α

    def _calculate_s1_s2_official(self, lr_schedule, warmup_steps=2000, max_lr=0.001):
        """
        Official S1 and S2 calculation exactly following the official code.
        
        This matches the official implementation exactly:
        - S1: cumulative sum of learning rates 
        - S2: cumulative sum of momentum values
        """
        # Convert to numpy for consistency with official code
        lr_schedule = np.array(lr_schedule, dtype=np.float64)
        
        # Build full lr schedule including warmup (following official code approach)
        # In official code, warmup is treated as constant max_lr for S1/S2 calculation
        lr_schedule_full = np.concatenate([
            np.full(warmup_steps, max_lr, dtype=np.float64),
            lr_schedule.astype(np.float64)
        ])
        
        # Calculate S1: cumulative sum of all learning rates
        S1_full = np.cumsum(lr_schedule_full)
        
        # Calculate S2: cumulative sum of momentum values
        n = len(lr_schedule_full)
        momentum = np.zeros(n, dtype=np.float64)
        
        # Official momentum calculation: momentum[i] = decay_factor * momentum[i-1] + (lr[i-1] - lr[i])
        for i in range(1, n):
            momentum[i] = self.lambda_decay * momentum[i-1] + (lr_schedule_full[i-1] - lr_schedule_full[i])
        
        S2_full = np.cumsum(momentum)
        
        # Return only the decay part (post-warmup)
        S1_decay = S1_full[warmup_steps:]
        S2_decay = S2_full[warmup_steps:]
        
        return S1_decay, S2_decay

    def predict(self, lr_schedule, warmup_steps=2000, max_lr=0.001):
        """Predict loss using fitted parameters"""
        if self.params is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        S1, S2 = self._calculate_s1_s2_official(lr_schedule, warmup_steps, max_lr)
        L0, A, C, alpha = self.params
        
        # Tissue Law: L(s) = L₀ + A × S₁^(-α) - C × S₂
        pred_loss = L0 + A * np.power(np.maximum(S1, 1e-10), -alpha) - C * S2
        return pred_loss

    def _huber_loss(self, residual, delta=1e-3):
        """Huber loss implementation following official code"""
        return np.where(np.abs(residual) < delta, 
                       0.5 * (residual**2), 
                       delta * np.abs(residual) - 0.5 * (delta**2))

    def fit(self, lr_schedule, loss_curve, warmup_steps=2000, max_lr=0.001):
        """Fit parameters using the official optimization approach"""
        print("Fitting Tissue Momentum Law (Official Implementation)...")
        
        # Calculate S1 and S2
        S1_fit, S2_fit = self._calculate_s1_s2_official(lr_schedule, warmup_steps, max_lr)
        
        def objective(params):
            L0, A, C, alpha = params
            if L0 <= 0 or A <= 0 or C < 0 or alpha <= 0:  # Allow C=0 but penalize negative
                return 1e10
            
            # Predict losses
            pred_loss = L0 + A * np.power(np.maximum(S1_fit, 1e-10), -alpha) - C * S2_fit
            
            # Check for invalid predictions
            if np.any(np.isnan(pred_loss)) or np.any(np.isinf(pred_loss)) or np.any(pred_loss <= 0):
                return 1e10
            
            # Log-space Huber loss (following official code)
            residual = np.log(loss_curve + 1e-8) - np.log(pred_loss + 1e-8)
            loss = np.sum(self._huber_loss(residual, delta=1e-3))
            return loss

        # Grid search initialization with better C range
        L0_init_range = np.linspace(0.5, 3.0, 3)
        A_init_range = np.linspace(1, 30, 3)
        C_init_range = np.linspace(0.0, 2.0, 3)  # Allow C=0, focus on meaningful values
        alpha_init_range = np.linspace(0.3, 1.5, 3)

        best_params = None
        best_loss = np.inf

        initial_params_grid = product(L0_init_range, A_init_range, C_init_range, alpha_init_range)
        
        for initial_param in tqdm(list(initial_params_grid), desc="Grid search fitting"):
            result = minimize(
                objective, 
                initial_param, 
                method='L-BFGS-B', 
                bounds=[(0.1, 10.0), (0.1, 100.0), (0.0, 10.0), (0.1, 3.0)],  # Better bounds
                options={'maxiter': 100000, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            if result.success and result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x

        if best_params is not None:
            self.params = best_params
            L0, A, C, alpha = best_params
            print(f"Tissue Law fitted: L₀={L0:.4f}, A={A:.4f}, C={C:.4f}, α={alpha:.4f}")
            
            # Calculate R²
            pred_loss = self.predict(lr_schedule, warmup_steps, max_lr)
            ss_res = np.sum((loss_curve - pred_loss) ** 2)
            ss_tot = np.sum((loss_curve - np.mean(loss_curve)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            print(f"Tissue Law R² = {r2:.5f}")
            
            return best_params, r2
        else:
            print("Tissue Law fitting failed")
            return None, 0


class LuoMultiPowerLawCorrected:
    """
    Simplified corrected wrapper class for Luo Multi-Power Law 
    """
    def __init__(self, warmup_steps=2160, peak_lr=3e-4):
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.fitted_params = None

    def fit(self, lr_schedule_decay_part, loss_curve_decay_part, peak_lr=None, tissue_params=None):
        """
        Improved fitting with better generalization and regularization
        """
        print("Fitting Corrected Luo Multi-Power Law...")
        
        # Improved Luo formula with better generalization
        def luo_formula(params, lrs):
            L0, A, B, C, alpha, beta, gamma = params
            T = len(lrs)
            
            # Ensure positive parameters with reasonable ranges
            L0 = abs(L0)
            A = abs(A)
            B = abs(B)
            C = max(abs(C), 1e-6)  # Prevent division by zero
            alpha = np.clip(abs(alpha), 0.1, 2.0)
            beta = np.clip(abs(beta), 0.1, 1.0)
            gamma = np.clip(abs(gamma), 0.1, 1.0)
            
            # S1_t cumulative sum with proper scaling
            S1_t = np.cumsum(lrs)
            S_W = np.mean(lrs) * 10  # Adaptive warmup based on LR scale
            
            # Power law term (main contribution)
            power_term = A * (S1_t + S_W) ** (-alpha)
            
            # More stable LD computation focused on major transitions
            LD_t = np.zeros(T)
            
            # Find significant LR changes (>10% decrease)
            lr_changes = []
            for t in range(1, T):
                if lrs[t-1] > lrs[t] and (lrs[t-1] - lrs[t]) / lrs[t-1] > 0.1:
                    lr_changes.append(t)
            
            # Only compute LD for significant changes to avoid overfitting
            for change_idx in lr_changes[:min(10, len(lr_changes))]:  # Limit to top 10 changes
                t = change_idx
                eta_diff = lrs[t-1] - lrs[t]
                eta_k = lrs[t]
                
                if eta_diff > 1e-8 and eta_k > 1e-8:
                    # Compute S_k(t) efficiently
                    window_end = min(t + 50, T)  # Limited window
                    S_k_t = np.sum(lrs[t:window_end])
                    
                    if S_k_t > 1e-8:
                        try:
                            inner_term = C * (eta_k ** (-gamma)) * S_k_t + 1
                            G_term = 1 - (inner_term ** (-beta))
                            
                            # Apply contribution with decay for future steps
                            contribution = B * eta_diff * G_term * 0.1  # Reduced impact
                            for future_t in range(t, T):
                                decay_factor = np.exp(-(future_t - t) * 0.01)
                                LD_t[future_t] += contribution * decay_factor
                        except:
                            continue
            
            predictions = L0 + power_term - LD_t
            return np.maximum(predictions, L0 * 0.1)  # Ensure reasonable minimum
        
        # Multi-objective function with regularization
        def objective(params):
            try:
                pred = luo_formula(params, lr_schedule_decay_part)
                
                # Primary loss: log-space MSE
                log_pred = np.log(pred + 1e-8)
                log_actual = np.log(loss_curve_decay_part + 1e-8)
                primary_loss = np.mean((log_pred - log_actual)**2)
                
                # Regularization terms to prevent overfitting
                L0, A, B, C, alpha, beta, gamma = params
                
                # Parameter magnitude penalty
                param_penalty = 0.01 * (abs(B - 400)**2 + abs(alpha - 0.5)**2 + 
                                       abs(beta - 0.6)**2 + abs(gamma - 0.6)**2)
                
                # Smoothness penalty (prefer smoother predictions)
                smoothness_penalty = 0.001 * np.mean(np.diff(pred)**2)
                
                total_loss = primary_loss + param_penalty + smoothness_penalty
                return total_loss
            except:
                return 1e6
        
        # Conservative initial parameters
        if tissue_params is not None:
            print("Using Tissue-guided initialization with regularization...")
            # tissue_params: [L0, A, C, alpha]
            initial = [tissue_params[0], 
                      min(tissue_params[1], 2.0),  # Limit A to prevent overfitting
                      400.0,  # Keep B at reference value
                      min(tissue_params[2], 1.0),  # Limit C
                      min(tissue_params[3], 1.0),  # Limit alpha
                      0.6, 0.6]  # Conservative beta, gamma
        else:
            initial = [2.8, 0.6, 400.0, 0.3, 0.5, 0.6, 0.6]
        
        # Tighter parameter bounds for better generalization
        bounds = [(2.0, 4.0),    # L0: reasonable range
                  (0.2, 1.5),    # A: prevent extreme values
                  (200, 600),    # B: around reference value
                  (0.1, 1.0),    # C: limited range
                  (0.2, 0.8),    # alpha: stable range
                  (0.4, 0.8),    # beta: stable range
                  (0.4, 0.8)]    # gamma: stable range
        
        print("Starting conservative parameter optimization...")
        
        # Try multiple optimization runs with different starting points
        best_result = None
        best_loss = float('inf')
        
        for run in range(3):
            # Slight perturbation of initial parameters
            perturbed_initial = [p * (1 + 0.1 * (np.random.random() - 0.5)) for p in initial]
            
            result = minimize(objective, perturbed_initial, method='L-BFGS-B', 
                            bounds=bounds, options={'maxiter': 200})
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        
        self.fitted_params = best_result.x
        
        # Calculate metrics
        pred_loss = luo_formula(self.fitted_params, lr_schedule_decay_part)
        ss_res = np.sum((loss_curve_decay_part - pred_loss) ** 2)
        ss_tot = np.sum((loss_curve_decay_part - np.mean(loss_curve_decay_part)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        print(f"Corrected Luo MPL fitted with R² = {r2:.6f}")
        print(f"Parameters: L0={self.fitted_params[0]:.3f}, A={self.fitted_params[1]:.3f}, "
              f"B={self.fitted_params[2]:.1f}, C={self.fitted_params[3]:.3f}, "
              f"α={self.fitted_params[4]:.3f}, β={self.fitted_params[5]:.3f}, γ={self.fitted_params[6]:.3f}")
        
        return self.fitted_params, r2

    def predict(self, lr_schedule_decay_part):
        """Make predictions using the fitted parameters with improved formula"""
        if self.fitted_params is None:
            raise ValueError("Model not fitted yet")
        
        L0, A, B, C, alpha, beta, gamma = self.fitted_params
        T = len(lr_schedule_decay_part)
        
        # Apply same constraints as in fitting
        L0 = abs(L0)
        A = abs(A)
        B = abs(B)
        C = max(abs(C), 1e-6)
        alpha = np.clip(abs(alpha), 0.1, 2.0)
        beta = np.clip(abs(beta), 0.1, 1.0)
        gamma = np.clip(abs(gamma), 0.1, 1.0)
        
        # S1_t cumulative sum with adaptive scaling
        S1_t = np.cumsum(lr_schedule_decay_part)
        S_W = np.mean(lr_schedule_decay_part) * 10
        
        # Power law term
        power_term = A * (S1_t + S_W) ** (-alpha)
        
        # Stable LD computation
        LD_t = np.zeros(T)
        
        # Find significant LR changes
        lr_changes = []
        for t in range(1, T):
            if lr_schedule_decay_part[t-1] > lr_schedule_decay_part[t] and \
               (lr_schedule_decay_part[t-1] - lr_schedule_decay_part[t]) / lr_schedule_decay_part[t-1] > 0.1:
                lr_changes.append(t)
        
        # Apply LD only for significant changes
        for change_idx in lr_changes[:min(10, len(lr_changes))]:
            t = change_idx
            eta_diff = lr_schedule_decay_part[t-1] - lr_schedule_decay_part[t]
            eta_k = lr_schedule_decay_part[t]
            
            if eta_diff > 1e-8 and eta_k > 1e-8:
                window_end = min(t + 50, T)
                S_k_t = np.sum(lr_schedule_decay_part[t:window_end])
                
                if S_k_t > 1e-8:
                    try:
                        inner_term = C * (eta_k ** (-gamma)) * S_k_t + 1
                        G_term = 1 - (inner_term ** (-beta))
                        
                        contribution = B * eta_diff * G_term * 0.1
                        for future_t in range(t, T):
                            decay_factor = np.exp(-(future_t - t) * 0.01)
                            LD_t[future_t] += contribution * decay_factor
                    except:
                        continue
        
        predictions = L0 + power_term - LD_t
        return np.maximum(predictions, L0 * 0.1)


def load_data_from_pickle(filepath='gpt_loss+lrs.pkl'):
    """
    Load data from pickle file with proper warmup handling.
    Note: Data already has warmup removed as stated in the problem.
    """
    print(f"Loading data from {filepath}...")
    with open(filepath, 'rb') as f:
        raw_data = pickle.load(f)
    
    datasets = {}
    for key, df_or_dict in raw_data.items():
        if isinstance(df_or_dict, pd.DataFrame):
            df = df_or_dict
            # Use exact column names as discovered: ['step', 'Metrics/loss', 'lr']
            datasets[key] = {
                'steps': df['step'].values,
                'losses': df['Metrics/loss'].values,  # Fixed: use correct column name 
                'lrs': df['lr'].values                # Fixed: use correct column name
            }
        elif isinstance(df_or_dict, dict) and all(k in df_or_dict for k in ['steps', 'losses', 'lrs']):
             datasets[key] = df_or_dict
        else:
            print(f"Warning: Skipping unrecognized data format for key '{key}'")

        if key in datasets:
             print(f"Loaded {key}: {len(datasets[key]['steps'])} steps (post-warmup).")
             print(f"  Loss range: {np.min(datasets[key]['losses']):.4f} - {np.max(datasets[key]['losses']):.4f}")
             print(f"  LR range: {np.min(datasets[key]['lrs']):.2e} - {np.max(datasets[key]['lrs']):.2e}")
    return datasets


def evaluate_model_predictions(true_loss, pred_loss):
    """Evaluate model predictions"""
    # Convert to numpy if needed (removed torch dependencies)
    if hasattr(true_loss, 'cpu'):
        true_loss = true_loss.cpu().numpy()
    if hasattr(pred_loss, 'cpu'):
        pred_loss = pred_loss.cpu().numpy()
        
    mse = np.mean((true_loss - pred_loss)**2)
    log_mse = np.mean((np.log(true_loss + 1e-8) - np.log(pred_loss + 1e-8))**2)
    ss_res = np.sum((true_loss - pred_loss)**2)
    ss_tot = np.sum((true_loss - np.mean(true_loss))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return {'mse': mse, 'log_mse': log_mse, 'r2': r2}


def subsample_data(data_dict, sample_ratio=0.1, min_points=100):
    """
    Subsample data points from a dictionary containing steps, losses, lrs.
    Uses logarithmic spacing to capture more points early in training.
    """
    num_total_points = len(data_dict['steps'])
    num_samples = max(min_points, int(num_total_points * sample_ratio))
    if num_samples >= num_total_points:
        return data_dict['steps'], data_dict['losses'], data_dict['lrs']
    
    # Use logarithmic spacing to capture early training dynamics better
    log_indices = np.logspace(0, np.log10(num_total_points), num_samples)
    indices = np.unique(np.round(log_indices - 1).astype(int))
    indices = indices[indices < num_total_points]  # Safety check
    
    return data_dict['steps'][indices], data_dict['losses'][indices], data_dict['lrs'][indices]


def run_experiments():
    """
    Main experiment function that reproduces the results from Tissue et al. and Luo et al. papers.
    Clean implementation using only scipy optimization.
    """
    print("="*60)
    print("Scaling Law Experiments - Clean Implementation")
    print("="*60)
    
    datasets = load_data_from_pickle()
    
    # Select fitting schedule (cosine as suggested)
    fit_schedule_key = 'M:100M_gpt_D:20B_scheduler:cosine_rope'
    if fit_schedule_key not in datasets:
        print(f"Warning: Cosine schedule not found. Available schedules:")
        for k in datasets.keys():
            print(f"  - {k}")
        fit_schedule_key = list(datasets.keys())[0]
    
    print(f"\n📊 Using '{fit_schedule_key.split(':')[-1].replace('_rope','')}' for fitting scaling laws.")
    fit_data = datasets[fit_schedule_key]

    # Since data is already post-warmup, we don't need to account for warmup steps
    # The "warmup_steps" parameter is now conceptual for S1/S2 calculation
    conceptual_warmup_steps = 2000  # From problem statement
    
    # For subsampling: use more points since this is the main evaluation
    sample_ratio = 0.3  # Use 30% of points for fitting
    fit_steps, fit_loss, fit_lr = subsample_data(fit_data, sample_ratio)
    print(f"Fitting using {len(fit_steps)} subsampled points from {len(fit_data['steps'])} total points.")
    
    # Peak LR is the first LR in the decay schedule (highest LR post-warmup)
    peak_lr_conceptual = fit_lr[0]
    print(f"Peak LR (post-warmup): {peak_lr_conceptual:.2e}")

    # --- Tissue Momentum Law ---
    print("\n" + "="*50)
    print("🧬 TISSUE MOMENTUM LAW (Official Implementation)")
    print("="*50)
    tissue_model = TissueMomentumLawOfficial(lambda_decay=0.999)
    tissue_params, tissue_r2_fit = tissue_model.fit(
        fit_lr, fit_loss, 
        warmup_steps=conceptual_warmup_steps, 
        max_lr=peak_lr_conceptual
    )

    # --- Luo Multi-Power Law ---
    print("\n" + "="*50)
    print("⚡ LUO MULTI-POWER LAW (Tissue-Guided Search)")
    print("="*50)
    luo_model = LuoMultiPowerLawCorrected(warmup_steps=conceptual_warmup_steps, peak_lr=peak_lr_conceptual)
    luo_params, luo_r2_fit = luo_model.fit(fit_lr, fit_loss, peak_lr=peak_lr_conceptual, tissue_params=tissue_params)
    
    # Store results for comprehensive evaluation
    all_results = {}

    # --- Evaluate on All Datasets ---
    print("\n" + "="*50)
    print("📈 COMPREHENSIVE EVALUATION ON ALL SCHEDULES")
    print("="*50)
    
    results_summary = []
    
    for schedule_key, data in datasets.items():
        schedule_name = schedule_key.split(':')[-1].replace('_rope','')
        print(f"\n--- Evaluating: {schedule_name} ---")
        
        current_lr = data['lrs']  # Fixed: use correct field name
        current_loss = data['losses']  # Fixed: use correct field name
        current_steps = data['steps']
        current_peak_lr = current_lr[0]
        
        # Tissue Prediction
        tissue_pred = tissue_model.predict(current_lr, conceptual_warmup_steps, current_peak_lr)
        tissue_eval = evaluate_model_predictions(current_loss, tissue_pred)
         # Luo Prediction
        luo_pred = luo_model.predict(current_lr)
        luo_eval = evaluate_model_predictions(current_loss, luo_pred)
        
        # Print results
        is_fit_data = schedule_key == fit_schedule_key
        fit_indicator = " (FITTED)" if is_fit_data else " (PREDICTED)"
        print(f"  {schedule_name:15}{fit_indicator}")
        print(f"    Tissue Law:    R²={tissue_eval['r2']:.4f}, LogMSE={tissue_eval['log_mse']:.4e}")
        print(f"    Luo Law:       R²={luo_eval['r2']:.4f}, LogMSE={luo_eval['log_mse']:.4e}")
        
        # Store for plotting
        all_results[schedule_key] = {
            'schedule_name': schedule_name,
            'steps': current_steps,
            'lr': current_lr,
            'loss_true': current_loss,
            'tissue_pred': tissue_pred,
            'luo_pred': luo_pred,
            'is_fit_data': is_fit_data,
            'tissue_eval': tissue_eval,
            'luo_eval': luo_eval
        }
        
        # Summary for table
        results_summary.append({
            'schedule': schedule_name,
            'type': 'FIT' if is_fit_data else 'PRED',
            'tissue_r2': tissue_eval['r2'],
            'luo_r2': luo_eval['r2'],
            'tissue_log_mse': tissue_eval['log_mse'],
            'luo_log_mse': luo_eval['log_mse']
        })
    
    # Summary table
    print("\n" + "="*62)
    print("📊 RESULTS SUMMARY TABLE")
    print("="*62)
    print(f"{'Schedule':<12} {'Type':<4} {'Tissue R²':<10} {'Luo R²':<10} {'Tissue LogMSE':<12} {'Luo LogMSE':<12}")
    print("-" * 62)
    for r in results_summary:
        print(f"{r['schedule']:<12} {r['type']:<4} {r['tissue_r2']:<10.4f} {r['luo_r2']:<10.4f} {r['tissue_log_mse']:<12.4e} {r['luo_log_mse']:<12.4e}")
    
    # Create comprehensive plots
    plot_comprehensive_results(all_results, fit_schedule_key)
    
    return all_results, results_summary


def plot_comprehensive_results(all_results, fit_schedule_key):
    """Create comprehensive plots comparing all models and schedules"""
    num_schedules = len(all_results)
    
    # Determine layout
    if num_schedules <= 2:
        fig, axes = plt.subplots(1, num_schedules, figsize=(8 * num_schedules, 6), squeeze=False)
    elif num_schedules <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), squeeze=False)
    else:
        cols = 3
        rows = (num_schedules + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False)
        
    axes = axes.flatten()

    colors = {
        'true': '#1f77b4',      # Blue
        'tissue': '#ff7f0e',    # Orange
        'luo': '#2ca02c'        # Green
    }

    for i, (schedule_key, result_data) in enumerate(all_results.items()):
        ax = axes[i]
        
        title_name = result_data['schedule_name']
        if result_data['is_fit_data']:
            title_name += " (FITTED)"
        else:
            title_name += " (PREDICTED)"

        # Plot subset for clarity if too many points
        num_points = len(result_data['steps'])
        plot_stride = max(1, num_points // 500)  # Aim for ~500 points

        steps_plot = result_data['steps'][::plot_stride]
        loss_true_plot = result_data['loss_true'][::plot_stride]
        tissue_pred_plot = result_data['tissue_pred'][::plot_stride]
        luo_pred_plot = result_data['luo_pred'][::plot_stride]
        lr_plot = result_data['lr'][::plot_stride]
        
        # Plot loss curves
        ax.plot(steps_plot, loss_true_plot, 
                label='True Loss', color=colors['true'], linewidth=2.5, alpha=0.8)
        ax.plot(steps_plot, tissue_pred_plot, 
                label=f'Tissue (R²={result_data["tissue_eval"]["r2"]:.3f})', 
                color=colors['tissue'], linestyle='--', linewidth=2)
        ax.plot(steps_plot, luo_pred_plot, 
                label=f'Luo (R²={result_data["luo_eval"]["r2"]:.3f})', 
                color=colors['luo'], linestyle=':', linewidth=2)
        
        ax.set_title(title_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Steps (Post-Warmup)')
        ax.set_ylabel('Loss')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle=':', alpha=0.6)

        # Secondary y-axis for LR
        ax_lr = ax.twinx()
        ax_lr.plot(steps_plot, lr_plot, 
                   label='Learning Rate', color='purple', linestyle='-.', alpha=0.4, linewidth=1)
        ax_lr.set_ylabel('Learning Rate', color='purple', fontsize=10)
        ax_lr.tick_params(axis='y', labelcolor='purple')
        ax_lr.set_yscale('log')  # Log scale for LR

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fit_name = fit_schedule_key.split(':')[-1].replace('_rope','')
    fig.suptitle(f"Scaling Law Reproduction Study (Fitted on: {fit_name})", fontsize=16, fontweight='bold')
    
    # Save with high DPI
    plt.savefig("scaling_law_reproduction_results_clean.png", dpi=300, bbox_inches='tight')
    print(f"\n📊 Comprehensive plots saved to 'scaling_law_reproduction_results_clean.png'")
    plt.show()
    
    # Create additional comparison plot
    create_comparison_summary_plot(all_results)


def create_comparison_summary_plot(all_results):
    """Create a summary comparison plot showing R² scores across all methods"""
    schedules = []
    tissue_r2s = []
    luo_r2s = []
    is_fitted = []
    
    for schedule_key, result_data in all_results.items():
        schedules.append(result_data['schedule_name'])
        tissue_r2s.append(result_data['tissue_eval']['r2'])
        luo_r2s.append(result_data['luo_eval']['r2'])
        is_fitted.append(result_data['is_fit_data'])
    
    x = np.arange(len(schedules))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color bars differently for fitted vs predicted
    tissue_colors = ['#ff7f0e' if fitted else '#ffbb78' for fitted in is_fitted]
    luo_colors = ['#2ca02c' if fitted else '#98df8a' for fitted in is_fitted]
    
    bars1 = ax.bar(x - width/2, tissue_r2s, width, label='Tissue Law', color=tissue_colors, alpha=0.8)
    bars2 = ax.bar(x + width/2, luo_r2s, width, label='Luo Law', color=luo_colors, alpha=0.8)
    
    ax.set_xlabel('Learning Rate Schedule')
    ax.set_ylabel('R² Score')
    ax.set_title('Model Performance Comparison Across All Schedules', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(schedules, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("model_comparison_summary_clean.png", dpi=300, bbox_inches='tight')
    print(f"📊 Comparison summary saved to 'model_comparison_summary_clean.png'")
    plt.show()


if __name__ == '__main__':
    run_experiments()
