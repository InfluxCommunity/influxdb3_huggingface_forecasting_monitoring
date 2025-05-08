import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def calculate_msre(y_true, y_pred, epsilon=1e-10):
    """
    Calculate Mean Squared Relative Error (MSRE).
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    epsilon : float
        Small constant to avoid division by zero
        
    Returns:
    --------
    float
        Mean Squared Relative Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Calculate relative errors
    relative_errors = (y_true - y_pred) / (np.abs(y_true) + epsilon)
    
    # Square the relative errors and take the mean
    msre = np.mean(np.square(relative_errors))
    
    return msre

def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error (MSE).
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    float
        Mean Squared Error
    """
    return mean_squared_error(y_true, y_pred)

def detect_drift(y_true, y_pred, threshold, metric='msre'):
    """
    Detect if there is drift based on error metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    threshold : float
        Threshold for drift detection
    metric : str
        Metric to use for drift detection ('msre' or 'mse')
        
    Returns:
    --------
    tuple
        (drift_detected, error_value)
    """
    if metric.lower() == 'msre':
        error = calculate_msre(y_true, y_pred)
    elif metric.lower() == 'mse':
        error = calculate_mse(y_true, y_pred)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'msre' or 'mse'.")
    
    # Detect drift if error exceeds threshold
    drift_detected = error > threshold
    
    return drift_detected, error

def calculate_drift_metrics_window(y_true, y_pred, window_size=10, stride=1, metric='msre'):
    """
    Calculate drift metrics over sliding windows.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    window_size : int
        Size of the sliding window
    stride : int
        Step size for the sliding window
    metric : str
        Metric to use for drift detection ('msre' or 'mse')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with window indices and corresponding error metrics
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    if len(y_true) < window_size:
        raise ValueError("Input arrays must be at least as long as window_size")
    
    # Initialize results
    window_indices = []
    error_values = []
    
    # Calculate error for each window
    for i in range(0, len(y_true) - window_size + 1, stride):
        window_indices.append(i)
        
        y_true_window = y_true[i:i + window_size]
        y_pred_window = y_pred[i:i + window_size]
        
        if metric.lower() == 'msre':
            error = calculate_msre(y_true_window, y_pred_window)
        elif metric.lower() == 'mse':
            error = calculate_mse(y_true_window, y_pred_window)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'msre' or 'mse'.")
        
        error_values.append(error)
    
    # Create DataFrame
    results = pd.DataFrame({
        'window_start_idx': window_indices,
        'error': error_values
    })
    
    return results

def plot_drift_metrics(metrics_df, threshold=None, figsize=(12, 6)):
    """
    Plot drift metrics over time.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame with window indices and error metrics
    threshold : float
        Threshold for drift detection, if None no threshold line is plotted
    figsize : tuple
        Figure size for the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(metrics_df['window_start_idx'], metrics_df['error'], marker='o', linestyle='-')
    
    if threshold is not None:
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    
    ax.set_title('Drift Metrics Over Time')
    ax.set_xlabel('Window Start Index')
    ax.set_ylabel('Error Metric')
    ax.legend()
    ax.grid(True)
    
    return fig
