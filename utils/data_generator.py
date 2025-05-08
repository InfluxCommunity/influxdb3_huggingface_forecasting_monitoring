import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sine_wave(start_date, periods, frequency, amplitude=1.0, frequency_factor=0.1, 
                        noise_level=0.1, phase=0.0, offset=0.0):
    """
    Generate a sine wave time series with optional noise and offset.
    
    Parameters:
    -----------
    start_date : datetime
        The start date for the time series
    periods : int
        Number of periods to generate
    frequency : str
        Pandas frequency string (e.g., 'H' for hourly, 'D' for daily)
    amplitude : float
        Amplitude of the sine wave
    frequency_factor : float
        Controls the frequency of the sine wave
    noise_level : float
        Standard deviation of Gaussian noise to add
    phase : float
        Phase shift of the sine wave in radians
    offset : float
        Vertical offset to add to the sine wave
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing time series with columns 'timestamp' and 'value'
    """
    # Create date range
    date_range = pd.date_range(start=start_date, periods=periods, freq=frequency)
    
    # Generate time values for sine wave (normalized between 0 and 2π × periods)
    t = np.arange(periods) * frequency_factor
    
    # Generate sine wave with phase shift and offset
    values = amplitude * np.sin(t + phase) + offset
    
    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, periods)
        values += noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'value': values
    })
    
    return df

def add_drift(df, start_idx, drift_type='offset', offset_value=0.5, noise_amplitude=0.5):
    """
    Add drift to a portion of the time series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the time series with 'timestamp' and 'value' columns
    start_idx : int
        Index at which to start adding drift
    drift_type : str
        Type of drift to add: 'offset' or 'noise'
    offset_value : float
        Amount of offset to add if drift_type is 'offset'
    noise_amplitude : float
        Amplitude of additional noise if drift_type is 'noise'
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added drift
    """
    # Create a copy to avoid modifying the original
    df_drift = df.copy()
    
    if drift_type == 'offset':
        # Add a constant offset
        df_drift.loc[start_idx:, 'value'] += offset_value
    elif drift_type == 'noise':
        # Add additional noise
        additional_noise = np.random.normal(0, noise_amplitude, len(df_drift) - start_idx)
        df_drift.loc[start_idx:, 'value'] += additional_noise
    else:
        raise ValueError(f"Unknown drift type: {drift_type}. Use 'offset' or 'noise'.")
    
    return df_drift

def prepare_data_for_lstm(data, time_steps, target_col='value', features=None):
    """
    Prepare time series data for LSTM model by creating sequences.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the time series
    time_steps : int
        Number of time steps to use for each sequence
    target_col : str
        Column name for the target variable
    features : list
        List of feature column names to include, if None only target_col is used
    
    Returns:
    --------
    tuple
        (X, y) where X is the input sequences and y is the target values
    """
    if features is None:
        features = [target_col]
    
    X, y = [], []
    
    for i in range(len(data) - time_steps):
        # Sequence of features
        feature_seq = data[features].iloc[i:i+time_steps].values
        # Next value to predict
        target = data[target_col].iloc[i+time_steps]
        
        X.append(feature_seq)
        y.append(target)
    
    return np.array(X), np.array(y)

def train_test_split_time_series(data, train_ratio=0.8):
    """
    Split time series data into training and testing sets.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the time series
    train_ratio : float
        Ratio of data to use for training (0 to 1)
        
    Returns:
    --------
    tuple
        (train_data, test_data)
    """
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size].copy()
    test_data = data.iloc[train_size:].copy()
    
    return train_data, test_data
