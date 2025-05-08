import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import torch
import json
import requests
from datetime import datetime, timedelta

# Import utility functions
from utils.data_generator import (
    generate_sine_wave, 
    add_drift, 
    prepare_data_for_lstm, 
    train_test_split_time_series
)
from utils.model import TimeSeriesLSTM
from utils.drift_detection import (
    calculate_msre,
    calculate_mse,
    detect_drift, 
    calculate_drift_metrics_window, 
    plot_drift_metrics
)
from utils.huggingface_utils import (
    upload_model_to_huggingface,
    download_model_from_huggingface,
    check_model_exists_in_huggingface
)
# Import InfluxDB utility
from utils.influxdb_utils import InfluxDBHandler

# Set page configuration
st.set_page_config(
    page_title="LSTM Time Series Forecasting with Drift Detection",
    page_icon="📈",
    layout="wide"
)

# Hugging Face configuration
# Default repository ID - replace with your own repository
HF_REPO_ID = os.environ.get("HF_REPO_ID", "Anaisdg/influx-lstm-forecaster")
HF_MODEL_FILE = "model.pt"

# App title and description
st.title("LSTM Time Series Forecasting & Drift Detection")
st.markdown("""
This application demonstrates time series forecasting with LSTM models, 
drift detection using MSRE (Mean Squared Relative Error), and automated model retraining.

The workflow includes:
1. Generating synthetic time series data (sine wave with noise)
2. Training an initial LSTM model on clean data
3. Forecasting future points using the initial model
4. Injecting drift into the data
5. Detecting model drift using MSRE
6. Retraining the model when drift exceeds a threshold
7. Comparing forecasts from original and retrained models
""")

# Create sidebar for parameters
st.sidebar.header("Parameters")

# Data Generation Parameters
st.sidebar.subheader("Data Generation")
data_points = st.sidebar.slider("Number of data points", 100, 1000, 300)
noise_level = st.sidebar.slider("Noise level", 0.0, 0.5, 0.1, 0.05)
freq_factor = st.sidebar.slider("Frequency factor", 0.01, 0.5, 0.1, 0.01)
amplitude = st.sidebar.slider("Amplitude", 0.1, 5.0, 1.0, 0.1)

# LSTM Model Parameters
st.sidebar.subheader("LSTM Model")
time_steps = st.sidebar.slider("Time steps (sequence length)", 5, 50, 10)
lstm_units = st.sidebar.slider("LSTM units", 10, 100, 50, 10)
dropout_rate = st.sidebar.slider("Dropout rate", 0.0, 0.5, 0.2, 0.05)
epochs = st.sidebar.slider("Training epochs", 10, 200, 50, 10)
batch_size = st.sidebar.slider("Batch size", 8, 64, 32, 8)

# Drift Parameters
st.sidebar.subheader("Drift Configuration")
drift_type = st.sidebar.selectbox("Drift type", ["offset", "noise"])
drift_start_percent = st.sidebar.slider("Drift start point (%)", 50, 90, 70)
drift_magnitude = st.sidebar.slider("Drift magnitude", 0.1, 2.0, 0.5, 0.1)

# Forecasting Parameters
st.sidebar.subheader("Forecasting")
forecast_horizon = st.sidebar.slider("Forecast horizon", 5, 50, 10)

# Drift Detection Parameters
st.sidebar.subheader("Drift Detection")
drift_metric = st.sidebar.selectbox("Drift metric", ["msre", "mse"])
drift_threshold = st.sidebar.slider("Drift threshold", 0.01, 0.5, 0.1, 0.01)
window_size = st.sidebar.slider("Window size for drift calculation", 5, 30, 10)

# Main app functionality
def main():
    # Create tabs for different steps in the workflow
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "0. InfluxDB Configuration",
        "1. Data Generation", 
        "2. Initial Model Training", 
        "3. Drift Injection", 
        "4. Drift Detection", 
        "5. Model Retraining",
        "6. Model Persistence"
    ])
    
    # Setting session state for data persistence between interactions
    if 'original_data' not in st.session_state:
        st.session_state.original_data = None
    if 'drifted_data' not in st.session_state:
        st.session_state.drifted_data = None
    if 'lstm_model' not in st.session_state:
        st.session_state.lstm_model = None
    if 'retrained_model' not in st.session_state:
        st.session_state.retrained_model = None
    if 'drift_detected' not in st.session_state:
        st.session_state.drift_detected = False
    if 'train_data' not in st.session_state:
        st.session_state.train_data = None
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'original_predictions' not in st.session_state:
        st.session_state.original_predictions = None
    if 'original_forecasts' not in st.session_state:
        st.session_state.original_forecasts = None
    if 'retrained_predictions' not in st.session_state:
        st.session_state.retrained_predictions = None
    if 'retrained_forecasts' not in st.session_state:
        st.session_state.retrained_forecasts = None
    if 'drift_metrics' not in st.session_state:
        st.session_state.drift_metrics = None
    if 'influxdb_handler' not in st.session_state:
        st.session_state.influxdb_handler = None
    
    # Tab 0: InfluxDB Configuration
    with tab0:
        st.header("0. InfluxDB Configuration")
        
        st.markdown("""
        Configure your InfluxDB connection settings here. Data will be stored in and retrieved from InfluxDB
        throughout the pipeline. You can use environment variables or enter the values directly.
        """)
        
        # InfluxDB connection settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Connection Settings")
            
            # Get environment variables as defaults
            default_host = os.environ.get("INFLUXDB_HOST", "http://localhost:8181")
            default_token = os.environ.get("INFLUXDB_TOKEN", "")
            default_database = os.environ.get("INFLUXDB_DATABASE", "timeseries")
            default_org = os.environ.get("INFLUXDB_ORG", "")
            
            # Input fields
            host = st.text_input("InfluxDB Host", value=default_host, 
                                help="URL of your InfluxDB instance (e.g., http://localhost:8181)")
            token = st.text_input("InfluxDB Token", value=default_token, type="password", 
                                 help="Authentication token for InfluxDB")
            database = st.text_input("InfluxDB Database", value=default_database, 
                                    help="Name of the database/bucket")
            org = st.text_input("InfluxDB Organization", value=default_org, 
                              help="Organization name (optional)")
            
            # Connect button
            if st.button("Connect to InfluxDB"):
                with st.spinner("Connecting to InfluxDB..."):
                    # Create InfluxDB handler
                    influxdb_handler = InfluxDBHandler(
                        host=host,
                        token=token,
                        database=database,
                        org=org
                    )
                    
                    # Test connection
                    if influxdb_handler.connect():
                        st.session_state.influxdb_handler = influxdb_handler
                        st.success(f"Successfully connected to InfluxDB at {host}")
                    else:
                        st.error("Failed to connect to InfluxDB. Please check your settings.")
        
        with col2:
            st.subheader("Connection Status")
            
            if st.session_state.influxdb_handler and st.session_state.influxdb_handler.connected:
                st.success("Connected to InfluxDB")
                st.json({
                    "host": st.session_state.influxdb_handler.host,
                    "database": st.session_state.influxdb_handler.database,
                    "org": st.session_state.influxdb_handler.org if st.session_state.influxdb_handler.org else "Not specified",
                    "status": "Connected"
                })
                
                # Connection status is displayed above - no additional buttons
            else:
                st.warning("Not connected to InfluxDB")
                st.info("Please enter your connection details and click 'Connect to InfluxDB'")
    
    # Tab 1: Data Generation
    with tab1:
        st.header("1. Data Generation")
        
        # Data generation button
        if st.button("Generate New Data"):
            with st.spinner("Generating synthetic time series data..."):
                    # Generate clean data
                    start_date = datetime.now() - timedelta(days=data_points)
                    df = generate_sine_wave(
                        start_date=start_date,
                        periods=data_points,
                        frequency='D',
                        amplitude=amplitude,
                        frequency_factor=freq_factor,
                        noise_level=noise_level
                    )
                    
                    # Store the data in session state
                    st.session_state.original_data = df
                    
                    # Reset other session state variables when generating new data
                    st.session_state.drifted_data = None
                    st.session_state.lstm_model = None
                    st.session_state.retrained_model = None
                    st.session_state.drift_detected = False
                    st.session_state.train_data = None
                    st.session_state.test_data = None
                    st.session_state.X_train = None
                    st.session_state.y_train = None
                    st.session_state.X_test = None
                    st.session_state.y_test = None
                    st.session_state.original_predictions = None
                    st.session_state.original_forecasts = None
                    st.session_state.retrained_predictions = None
                    st.session_state.retrained_forecasts = None
                    st.session_state.drift_metrics = None
                    
                    # Split data
                    train_data, test_data = train_test_split_time_series(df, train_ratio=0.8)
                    st.session_state.train_data = train_data
                    st.session_state.test_data = test_data
                    
                    # Write to InfluxDB if connected
                    if st.session_state.influxdb_handler and st.session_state.influxdb_handler.connected:
                        with st.spinner("Writing data to InfluxDB..."):
                            # Prepare DataFrame for InfluxDB (ensure timestamp is index)
                            df_for_influx = df.copy()
                            
                            # Write to InfluxDB
                            success = st.session_state.influxdb_handler.write_dataframe(
                                df=df_for_influx,
                                measurement_name="raw_data"
                            )
                            
                            if success:
                                st.success(f"Successfully wrote {len(df)} data points to InfluxDB measurement 'raw_data'")
                            else:
                                st.error("Failed to write data to InfluxDB")
                    else:
                        st.warning("Not connected to InfluxDB. Data was generated but not stored in InfluxDB.")
                        st.info("To store data in InfluxDB, please connect to InfluxDB in the 'InfluxDB Configuration' tab first.")
                    
                    st.success("Data generated successfully!")
        

        
        # Display data if available
        if st.session_state.original_data is not None:
            df = st.session_state.original_data
            
            # Plot the data
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['timestamp'], df['value'], label='Original Data')
            ax.set_title('Generated Synthetic Time Series Data')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True)
            ax.legend()
            
            st.pyplot(fig)
            
            # Display sample of the data
            st.subheader("Sample Data")
            st.dataframe(df.head())
            
            # Display train-test split
            st.subheader("Train-Test Split")
            
            train_data = st.session_state.train_data
            test_data = st.session_state.test_data
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(train_data['timestamp'], train_data['value'], label='Training Data')
            ax.plot(test_data['timestamp'], test_data['value'], label='Testing Data')
            ax.set_title('Train-Test Split of Time Series Data')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True)
            ax.legend()
            
            st.pyplot(fig)
            
            # Display statistics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Data")
                st.write(f"Shape: {train_data.shape}")
                st.write(f"Mean: {train_data['value'].mean():.4f}")
                st.write(f"Std: {train_data['value'].std():.4f}")
            
            with col2:
                st.subheader("Testing Data")
                st.write(f"Shape: {test_data.shape}")
                st.write(f"Mean: {test_data['value'].mean():.4f}")
                st.write(f"Std: {test_data['value'].std():.4f}")
        else:
            st.info("Click 'Generate New Data' to create synthetic time series data.")
    
    # Tab 2: Initial Model Training
    with tab2:
        st.header("2. Initial Model Training")
        
        if st.session_state.original_data is None:
            st.warning("Please generate data first in the 'Data Generation' tab.")
        else:
            # Train Model button - add unique key to avoid collision
            if st.button("Train Initial LSTM Model", key="train_model_button"):
                with st.spinner("Training LSTM model..."):
                    # Prepare data for LSTM
                    train_data = st.session_state.train_data
                    
                    # Scale data
                    lstm_model = TimeSeriesLSTM(
                        time_steps=time_steps,
                        features=1,
                        units=lstm_units,
                        dropout_rate=dropout_rate
                    )
                    
                    # Get scaled values for training
                    train_values = train_data['value'].values.reshape(-1, 1)
                    scaled_train_values = lstm_model.scaler.fit_transform(train_values).flatten()
                    
                    # Prepare sequences
                    X_train, y_train = prepare_data_for_lstm(
                        pd.DataFrame({'value': scaled_train_values}),
                        time_steps=time_steps
                    )
                    
                    # Reshape X_train for LSTM input
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
                    
                    # Train the model
                    lstm_model.build_model()
                    lstm_model.train(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        verbose=0
                    )
                    
                    # Store the model and training data in session state
                    st.session_state.lstm_model = lstm_model
                    st.session_state.X_train = X_train
                    st.session_state.y_train = y_train
                    
                    # Prepare test data for predictions
                    test_data = st.session_state.test_data
                    test_values = test_data['value'].values.reshape(-1, 1)
                    scaled_test_values = lstm_model.scaler.transform(test_values).flatten()
                    
                    X_test, y_test = prepare_data_for_lstm(
                        pd.DataFrame({'value': scaled_test_values}),
                        time_steps=time_steps
                    )
                    
                    # Reshape X_test for LSTM input
                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    # Make predictions on test data
                    scaled_predictions = lstm_model.predict(X_test).flatten()
                    
                    # Inverse scale the predictions
                    predictions = lstm_model.scaler.inverse_transform(
                        scaled_predictions.reshape(-1, 1)
                    ).flatten()
                    
                    # Store predictions
                    st.session_state.original_predictions = predictions
                    
                    # Generate forecasts
                    last_sequence = scaled_test_values[-time_steps:]
                    scaled_forecasts = lstm_model.forecast(last_sequence, steps=forecast_horizon)
                    
                    # Inverse scale the forecasts
                    forecasts = lstm_model.scaler.inverse_transform(
                        scaled_forecasts.reshape(-1, 1)
                    ).flatten()
                    
                    # Store forecasts
                    st.session_state.original_forecasts = forecasts
                    
                    st.success("Model trained successfully!")
            
            # Display model information and predictions if available
            if st.session_state.lstm_model is not None:
                lstm_model = st.session_state.lstm_model
                
                # Display model summary
                st.subheader("LSTM Model Summary")
                
                # Create a custom model summary for PyTorch
                model_summary = []
                model_summary.append(f"Time Steps: {lstm_model.time_steps}")
                model_summary.append(f"Features: {lstm_model.features}")
                model_summary.append(f"LSTM Units: {lstm_model.units}")
                model_summary.append(f"Dropout Rate: {lstm_model.dropout_rate}")
                model_summary.append(f"Learning Rate: {lstm_model.learning_rate}")
                model_summary.append(f"\nModel Architecture:")
                model_summary.append(str(lstm_model.model))
                
                st.text("\n".join(model_summary))
                
                # Plot training history
                st.subheader("Training History")
                fig = lstm_model.plot_loss()
                st.pyplot(fig)
                
                # Plot predictions
                if st.session_state.original_predictions is not None:
                    st.subheader("Model Predictions on Test Data")
                    
                    test_data = st.session_state.test_data
                    predictions = st.session_state.original_predictions
                    
                    # Create indices for predictions (they start after time_steps)
                    pred_indices = test_data.index[time_steps:time_steps + len(predictions)]
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(test_data.index, test_data['value'], label='Actual')
                    ax.plot(pred_indices, predictions, label='Predicted', linestyle='--')
                    ax.set_title('LSTM Model Predictions vs Actual Values')
                    ax.set_xlabel('Index')
                    ax.set_ylabel('Value')
                    ax.grid(True)
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Calculate and display error metrics
                    y_true = test_data['value'].values[time_steps:time_steps + len(predictions)]
                    y_pred = predictions
                    
                    mse = calculate_mse(y_true, y_pred)
                    msre = calculate_msre(y_true, y_pred)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("MSE", f"{mse:.6f}")
                    with col2:
                        st.metric("MSRE", f"{msre:.6f}")
                
                # Plot forecasts
                if st.session_state.original_forecasts is not None:
                    st.subheader("Future Forecasts")
                    
                    forecasts = st.session_state.original_forecasts
                    full_data = st.session_state.original_data
                    
                    # Create forecast dates (continuing from the end of the data)
                    last_date = full_data['timestamp'].iloc[-1]
                    forecast_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=forecast_horizon,
                        freq='D'
                    )
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Plot the original data
                    ax.plot(full_data['timestamp'], full_data['value'], label='Historical Data')
                    
                    # Plot the forecasts
                    ax.plot(forecast_dates, forecasts, label='Forecast', linestyle='--', marker='o')
                    
                    # Add a shaded area to separate history from forecast
                    ax.axvline(x=last_date, color='r', linestyle=':')
                    
                    ax.set_title('LSTM Model Future Forecasts')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Value')
                    ax.grid(True)
                    ax.legend()
                    
                    st.pyplot(fig)
            else:
                st.info("Click 'Train Initial LSTM Model' to train the model on the generated data.")
    
    # Tab 3: Drift Injection
    with tab3:
        st.header("3. Drift Injection")
        
        if st.session_state.original_data is None:
            st.warning("Please generate data first in the 'Data Generation' tab.")
        else:
            # Add Drift button - add unique key to avoid collision
            if st.button("Inject Drift", key="inject_drift_button"):
                with st.spinner("Injecting drift into the data..."):
                    original_data = st.session_state.original_data
                    
                    # Calculate the index to start drift
                    drift_start_idx = int(len(original_data) * drift_start_percent / 100)
                    
                    # Inject drift based on selected type
                    if drift_type == "offset":
                        drifted_data = add_drift(
                            original_data,
                            start_idx=drift_start_idx,
                            drift_type='offset',
                            offset_value=drift_magnitude
                        )
                    else:  # noise
                        drifted_data = add_drift(
                            original_data,
                            start_idx=drift_start_idx,
                            drift_type='noise',
                            noise_amplitude=drift_magnitude
                        )
                    
                    # Store drifted data
                    st.session_state.drifted_data = drifted_data
                    
                    # Write to InfluxDB if connected
                    if st.session_state.influxdb_handler and st.session_state.influxdb_handler.connected:
                        with st.spinner("Writing drifted data to InfluxDB..."):
                            # Prepare DataFrame for InfluxDB
                            df_for_influx = drifted_data.copy()
                            
                            # Write to InfluxDB
                            success = st.session_state.influxdb_handler.write_dataframe(
                                df=df_for_influx,
                                measurement_name="drifted_data"
                            )
                            
                            if success:
                                st.success(f"Successfully wrote {len(drifted_data)} data points to InfluxDB measurement 'drifted_data'")
                            else:
                                st.error("Failed to write drifted data to InfluxDB")
                    else:
                        st.warning("Not connected to InfluxDB. Drifted data was generated but not stored in InfluxDB.")
                    
                    st.success("Drift injected successfully!")
            
            # Display original vs drifted data
            if st.session_state.drifted_data is not None:
                original_data = st.session_state.original_data
                drifted_data = st.session_state.drifted_data
                
                st.subheader("Original vs Drifted Data")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(original_data['timestamp'], original_data['value'], label='Original Data')
                ax.plot(drifted_data['timestamp'], drifted_data['value'], label='Drifted Data', alpha=0.7)
                
                # Add vertical line at drift start
                drift_start_idx = int(len(original_data) * drift_start_percent / 100)
                drift_start_date = original_data['timestamp'].iloc[drift_start_idx]
                ax.axvline(x=drift_start_date, color='r', linestyle=':', label='Drift Start')
                
                ax.set_title('Original vs Drifted Time Series Data')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend()
                
                st.pyplot(fig)
                
                # Show difference
                st.subheader("Drift Magnitude")
                
                fig, ax = plt.subplots(figsize=(12, 6))
                difference = drifted_data['value'] - original_data['value']
                ax.plot(original_data['timestamp'], difference)
                ax.axvline(x=drift_start_date, color='r', linestyle=':', label='Drift Start')
                ax.set_title('Difference between Original and Drifted Data')
                ax.set_xlabel('Time')
                ax.set_ylabel('Difference')
                ax.grid(True)
                ax.legend()
                
                st.pyplot(fig)
            else:
                st.info("Click 'Inject Drift' to add drift to the original data.")
    
    # Tab 4: Drift Detection
    with tab4:
        st.header("4. Drift Detection")
        
        if st.session_state.drifted_data is None or st.session_state.lstm_model is None:
            st.warning("Please generate data, train a model, and inject drift first.")
        else:
            # Detect Drift button - add unique key to avoid collision
            if st.button("Detect Drift", key="detect_drift_button"):
                with st.spinner("Detecting drift..."):
                    lstm_model = st.session_state.lstm_model
                    drifted_data = st.session_state.drifted_data
                    
                    # Prepare drifted data for predictions
                    drifted_values = drifted_data['value'].values.reshape(-1, 1)
                    scaled_drifted_values = lstm_model.scaler.transform(drifted_values).flatten()
                    
                    # Prepare sequences
                    X_drifted_all = []
                    y_drifted_all = []
                    
                    for i in range(len(scaled_drifted_values) - time_steps):
                        X_drifted_all.append(scaled_drifted_values[i:i+time_steps])
                        y_drifted_all.append(scaled_drifted_values[i+time_steps])
                    
                    X_drifted_all = np.array(X_drifted_all).reshape(-1, time_steps, 1)
                    y_drifted_all = np.array(y_drifted_all)
                    
                    # Make predictions
                    y_pred_scaled = lstm_model.predict(X_drifted_all).flatten()
                    
                    # Inverse transform predictions
                    y_pred = lstm_model.scaler.inverse_transform(
                        y_pred_scaled.reshape(-1, 1)
                    ).flatten()
                    
                    # Get actual values (shifted by time_steps)
                    y_true = drifted_data['value'].values[time_steps:]
                    
                    # Calculate drift metrics over sliding windows
                    drift_metrics = calculate_drift_metrics_window(
                        y_true, y_pred,
                        window_size=window_size,
                        stride=1,
                        metric=drift_metric
                    )
                    
                    # Detect if drift exceeds threshold
                    drift_metrics['drift_detected'] = drift_metrics['error'] > drift_threshold
                    
                    # Store metrics in session state
                    st.session_state.drift_metrics = drift_metrics
                    
                    # Check if drift was detected
                    st.session_state.drift_detected = drift_metrics['drift_detected'].any()
                    
                    # Write drift metrics to InfluxDB if connected
                    if st.session_state.influxdb_handler and st.session_state.influxdb_handler.connected:
                        with st.spinner("Writing drift metrics to InfluxDB..."):
                            try:
                                # Prepare drift metrics for InfluxDB
                                # Add timestamp column based on drifted_data timestamps
                                metrics_df = drift_metrics.copy()
                                metrics_df['timestamp'] = drifted_data['timestamp'].iloc[metrics_df['window_start_idx'] + time_steps].values
                                
                                # Write to InfluxDB
                                success = st.session_state.influxdb_handler.write_dataframe(
                                    df=metrics_df,
                                    measurement_name="drift_metrics"
                                )
                                
                                if success:
                                    st.success(f"Successfully wrote {len(metrics_df)} drift metrics to InfluxDB measurement 'drift_metrics'")
                                    
                                    # If drift was detected, write a signal to the model_events table
                                    if st.session_state.drift_detected:
                                        try:
                                            # Find when drift was first detected
                                            first_drift_idx = drift_metrics[drift_metrics['drift_detected']].iloc[0]['window_start_idx']
                                            drift_date = drifted_data['timestamp'].iloc[first_drift_idx + time_steps]
                                            
                                            # Create a point for model_events table
                                            event_success = st.session_state.influxdb_handler.write_point(
                                                measurement="model_events",
                                                # Use the minimal set of tags needed - model_events might have schema constraints
                                                tags={
                                                    "type": "drift_detected"
                                                },
                                                fields={
                                                    "error_value": float(drift_metrics[drift_metrics['drift_detected']].iloc[0]['error']),
                                                    "threshold": float(drift_threshold),
                                                    "metric": drift_metric,
                                                    "requires_retraining": True,
                                                    "model_name": "lstm_forecaster"  # Move to fields if it can't be a tag
                                                },
                                                time=drift_date
                                            )
                                            
                                            if event_success:
                                                st.success("Drift event signal written to 'model_events' measurement")
                                        except Exception as e:
                                            st.warning(f"Could not write drift event to model_events: {str(e)}")
                                            st.info("Drift was still detected, but the signal could not be written to the database.")
                                else:
                                    st.error("Failed to write drift metrics to InfluxDB")
                            except Exception as e:
                                st.error(f"Error writing to InfluxDB: {str(e)}")
                    
                    st.success("Drift detection completed!")
            
            # Display drift detection results
            if st.session_state.drift_metrics is not None:
                drift_metrics = st.session_state.drift_metrics
                
                st.subheader("Drift Detection Results")
                
                # Display drift metrics plot
                fig = plot_drift_metrics(drift_metrics, threshold=drift_threshold)
                st.pyplot(fig)
                
                # Show detection result
                if st.session_state.drift_detected:
                    st.error(f"⚠️ Drift detected! The {drift_metric.upper()} exceeds the threshold of {drift_threshold}.")
                    
                    # Find when drift was first detected
                    first_drift_idx = drift_metrics[drift_metrics['drift_detected']].iloc[0]['window_start_idx']
                    drift_data = st.session_state.drifted_data
                    drift_date = drift_data['timestamp'].iloc[first_drift_idx + time_steps]
                    
                    st.write(f"First drift detected at index {first_drift_idx + time_steps} (date: {drift_date.date()})")
                else:
                    st.success(f"✅ No drift detected. The {drift_metric.upper()} remains below the threshold of {drift_threshold}.")
                
                # Show predictions vs actual values
                st.subheader("Model Predictions on Drifted Data")
                
                drifted_data = st.session_state.drifted_data
                
                # Get values for plotting
                actual_values = drifted_data['value'].values[time_steps:]
                timestamps = drifted_data['timestamp'].iloc[time_steps:]
                
                # Get predictions from LSTM model on drifted data
                lstm_model = st.session_state.lstm_model
                drifted_values = drifted_data['value'].values.reshape(-1, 1)
                scaled_drifted_values = lstm_model.scaler.transform(drifted_values).flatten()
                
                X_drifted = []
                for i in range(len(scaled_drifted_values) - time_steps):
                    X_drifted.append(scaled_drifted_values[i:i+time_steps])
                
                X_drifted = np.array(X_drifted).reshape(-1, time_steps, 1)
                
                scaled_predictions = lstm_model.predict(X_drifted).flatten()
                predictions = lstm_model.scaler.inverse_transform(
                    scaled_predictions.reshape(-1, 1)
                ).flatten()
                
                # Plotting
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot actual values
                ax.plot(timestamps, actual_values, label='Actual (Drifted)')
                
                # Plot predictions
                ax.plot(timestamps, predictions, label='Predicted', linestyle='--')
                
                # Add vertical line at drift start
                drift_start_idx = int(len(drifted_data) * drift_start_percent / 100)
                drift_start_date = drifted_data['timestamp'].iloc[drift_start_idx]
                ax.axvline(x=drift_start_date, color='r', linestyle=':', label='Drift Start')
                
                # If drift detected, add that line too
                if st.session_state.drift_detected:
                    first_drift_idx = drift_metrics[drift_metrics['drift_detected']].iloc[0]['window_start_idx']
                    drift_detected_date = drifted_data['timestamp'].iloc[first_drift_idx + time_steps]
                    ax.axvline(x=drift_detected_date, color='g', linestyle='-.', label='Drift Detected')
                
                ax.set_title('Model Predictions vs Actual Values (Drifted Data)')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend()
                
                st.pyplot(fig)
                
                # Calculate and display error metrics
                mse = calculate_mse(actual_values, predictions)
                msre = calculate_msre(actual_values, predictions)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("MSE on Drifted Data", f"{mse:.6f}")
                with col2:
                    st.metric("MSRE on Drifted Data", f"{msre:.6f}")
            else:
                st.info("Click 'Detect Drift' to analyze the drifted data for model drift.")
    
    # Tab 5: Model Retraining
    with tab5:
        st.header("5. Model Retraining")
        
        if st.session_state.drift_metrics is None:
            st.warning("Please complete drift detection first.")
        else:
            # Retrain model if drift detected
            if st.session_state.drift_detected:
                # Retrain Model button - add unique key to avoid collision
                if st.button("Retrain Model", key="retrain_model_button"):
                    with st.spinner("Retraining LSTM model on drifted data..."):
                        drifted_data = st.session_state.drifted_data
                        
                        # Find when drift was first detected
                        drift_metrics = st.session_state.drift_metrics
                        first_drift_idx = drift_metrics[drift_metrics['drift_detected']].iloc[0]['window_start_idx']
                        
                        # Use data from after drift detection for retraining
                        retrain_start_idx = first_drift_idx
                        retrain_data = drifted_data.iloc[retrain_start_idx:].copy()
                        
                        # Create a new LSTM model with the same parameters
                        retrained_model = TimeSeriesLSTM(
                            time_steps=time_steps,
                            features=1,
                            units=lstm_units,
                            dropout_rate=dropout_rate
                        )
                        
                        # Get scaled values for training
                        retrain_values = retrain_data['value'].values.reshape(-1, 1)
                        scaled_retrain_values = retrained_model.scaler.fit_transform(retrain_values).flatten()
                        
                        # Prepare sequences
                        X_retrain, y_retrain = prepare_data_for_lstm(
                            pd.DataFrame({'value': scaled_retrain_values}),
                            time_steps=time_steps
                        )
                        
                        # Reshape X_retrain for LSTM input
                        X_retrain = X_retrain.reshape(X_retrain.shape[0], X_retrain.shape[1], 1)
                        
                        # Train the model
                        retrained_model.build_model()
                        retrained_model.train(
                            X_retrain, y_retrain,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.2,
                            verbose=0
                        )
                        
                        # Store the retrained model
                        st.session_state.retrained_model = retrained_model
                        
                        # Write retraining event to InfluxDB if connected
                        if st.session_state.influxdb_handler and st.session_state.influxdb_handler.connected:
                            with st.spinner("Writing model retraining event to InfluxDB..."):
                                try:
                                    # Get the current timestamp
                                    retrain_time = datetime.now()
                                    
                                    # Write retraining event to model_events
                                    event_success = st.session_state.influxdb_handler.write_point(
                                        measurement="model_events",
                                        tags={
                                            "type": "model_retrained"
                                        },
                                        fields={
                                            "epochs": epochs,
                                            "batch_size": batch_size,
                                            "time_steps": time_steps,
                                            "units": lstm_units,
                                            "dropout_rate": float(dropout_rate),
                                            "model_name": "lstm_forecaster"
                                        },
                                        time=retrain_time
                                    )
                                    
                                    if event_success:
                                        st.success("Model retraining event written to 'model_events' measurement")
                                except Exception as e:
                                    st.error(f"Error writing to InfluxDB: {str(e)}")
                        
                        # Get predictions on the full drifted data using the retrained model
                        drifted_values = drifted_data['value'].values.reshape(-1, 1)
                        scaled_drifted_values = retrained_model.scaler.transform(drifted_values).flatten()
                        
                        X_drifted = []
                        for i in range(len(scaled_drifted_values) - time_steps):
                            X_drifted.append(scaled_drifted_values[i:i+time_steps])
                        
                        X_drifted = np.array(X_drifted).reshape(-1, time_steps, 1)
                        
                        scaled_predictions = retrained_model.predict(X_drifted).flatten()
                        retrained_predictions = retrained_model.scaler.inverse_transform(
                            scaled_predictions.reshape(-1, 1)
                        ).flatten()
                        
                        st.session_state.retrained_predictions = retrained_predictions
                        
                        # Generate forecasts with retrained model
                        last_sequence = scaled_drifted_values[-time_steps:]
                        scaled_forecasts = retrained_model.forecast(last_sequence, steps=forecast_horizon)
                        
                        # Inverse scale the forecasts
                        retrained_forecasts = retrained_model.scaler.inverse_transform(
                            scaled_forecasts.reshape(-1, 1)
                        ).flatten()
                        
                        st.session_state.retrained_forecasts = retrained_forecasts
                        
                        st.success("Model retrained successfully!")
                
                # Display retrained model results
                if st.session_state.retrained_model is not None:
                    retrained_model = st.session_state.retrained_model
                    original_model = st.session_state.lstm_model
                    
                    # Display model summary comparison
                    st.subheader("Model Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Original Model")
                        # Plot training history
                        fig = original_model.plot_loss()
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("### Retrained Model")
                        # Plot training history
                        fig = retrained_model.plot_loss()
                        st.pyplot(fig)
                    
                    # Compare predictions
                    if st.session_state.retrained_predictions is not None:
                        st.subheader("Prediction Comparison on Drifted Data")
                        
                        drifted_data = st.session_state.drifted_data
                        
                        # Get original model predictions
                        original_model = st.session_state.lstm_model
                        drifted_values = drifted_data['value'].values.reshape(-1, 1)
                        scaled_drifted_values = original_model.scaler.transform(drifted_values).flatten()
                        
                        X_drifted = []
                        for i in range(len(scaled_drifted_values) - time_steps):
                            X_drifted.append(scaled_drifted_values[i:i+time_steps])
                        
                        X_drifted = np.array(X_drifted).reshape(-1, time_steps, 1)
                        
                        scaled_predictions = original_model.predict(X_drifted).flatten()
                        original_predictions = original_model.scaler.inverse_transform(
                            scaled_predictions.reshape(-1, 1)
                        ).flatten()
                        
                        # Get retrained model predictions
                        retrained_predictions = st.session_state.retrained_predictions
                        
                        # Get actual values
                        actual_values = drifted_data['value'].values[time_steps:]
                        timestamps = drifted_data['timestamp'].iloc[time_steps:]
                        
                        # Find when drift was first detected
                        drift_metrics = st.session_state.drift_metrics
                        first_drift_idx = drift_metrics[drift_metrics['drift_detected']].iloc[0]['window_start_idx']
                        
                        # Plotting
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot actual values
                        ax.plot(timestamps, actual_values, label='Actual (Drifted)')
                        
                        # Plot original model predictions
                        ax.plot(timestamps, original_predictions, label='Original Model', linestyle='--')
                        
                        # Plot retrained model predictions
                        ax.plot(timestamps, retrained_predictions, label='Retrained Model', linestyle=':')
                        
                        # Add vertical line at drift start
                        drift_start_idx = int(len(drifted_data) * drift_start_percent / 100)
                        drift_start_date = drifted_data['timestamp'].iloc[drift_start_idx]
                        ax.axvline(x=drift_start_date, color='r', linestyle=':', label='Drift Start')
                        
                        # Add vertical line at drift detection
                        drift_detected_date = drifted_data['timestamp'].iloc[first_drift_idx + time_steps]
                        ax.axvline(x=drift_detected_date, color='g', linestyle='-.', label='Drift Detected')
                        
                        # Add vertical line at retraining start
                        retrain_start_date = drifted_data['timestamp'].iloc[first_drift_idx]
                        ax.axvline(x=retrain_start_date, color='b', linestyle='--', label='Retraining Start')
                        
                        ax.set_title('Model Predictions Comparison on Drifted Data')
                        ax.set_xlabel('Time')
                        ax.set_ylabel('Value')
                        ax.grid(True)
                        ax.legend()
                        
                        st.pyplot(fig)
                        
                        # Calculate and display error metrics
                        # Original model metrics
                        orig_mse = calculate_mse(actual_values, original_predictions)
                        orig_msre = calculate_msre(actual_values, original_predictions)
                        
                        # Retrained model metrics
                        new_mse = calculate_mse(actual_values, retrained_predictions)
                        new_msre = calculate_msre(actual_values, retrained_predictions)
                        
                        # Calculate improvement percentages
                        mse_improvement = ((orig_mse - new_mse) / orig_mse) * 100
                        msre_improvement = ((orig_msre - new_msre) / orig_msre) * 100
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Original Model Metrics")
                            st.metric("MSE", f"{orig_mse:.6f}")
                            st.metric("MSRE", f"{orig_msre:.6f}")
                        
                        with col2:
                            st.subheader("Retrained Model Metrics")
                            st.metric("MSE", f"{new_mse:.6f}", f"{mse_improvement:.2f}%")
                            st.metric("MSRE", f"{new_msre:.6f}", f"{msre_improvement:.2f}%")
                    
                    # Compare forecasts
                    if st.session_state.retrained_forecasts is not None and st.session_state.original_forecasts is not None:
                        st.subheader("Future Forecast Comparison")
                        
                        original_forecasts = st.session_state.original_forecasts
                        retrained_forecasts = st.session_state.retrained_forecasts
                        
                        drifted_data = st.session_state.drifted_data
                        
                        # Create forecast dates
                        last_date = drifted_data['timestamp'].iloc[-1]
                        forecast_dates = pd.date_range(
                            start=last_date + timedelta(days=1),
                            periods=forecast_horizon,
                            freq='D'
                        )
                        
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Plot the historical data
                        ax.plot(drifted_data['timestamp'], drifted_data['value'], label='Historical Data')
                        
                        # Plot the original model forecasts
                        ax.plot(forecast_dates, original_forecasts, label='Original Model Forecast', 
                               linestyle='--', marker='o', alpha=0.7)
                        
                        # Plot the retrained model forecasts
                        ax.plot(forecast_dates, retrained_forecasts, label='Retrained Model Forecast', 
                               linestyle=':', marker='x')
                        
                        # Add a shaded area to separate history from forecast
                        ax.axvline(x=last_date, color='r', linestyle=':')
                        
                        ax.set_title('Future Forecasts Comparison')
                        ax.set_xlabel('Time')
                        ax.set_ylabel('Value')
                        ax.grid(True)
                        ax.legend()
                        
                        st.pyplot(fig)
                else:
                    st.info("Click 'Retrain Model' to train a new model on the drifted data.")
            else:
                st.success("No significant drift detected. Model retraining is not necessary.")
                st.info("If you want to experiment with retraining anyway, try lowering the drift threshold.")

# Add Model Persistence tab implementation
    with tab6:
        st.header("6. Model Persistence with Hugging Face")
        
        st.markdown("""
        This tab allows you to save your trained models to Hugging Face Hub or load existing models. 
        Models are saved with their configuration and preprocessing scalers to ensure reproducibility.
        """)
        
        # Hugging Face Settings
        st.subheader("Hugging Face Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            repo_id = st.text_input("Repository ID", value=HF_REPO_ID)
            st.info("The repository ID should be in the format 'username/repo-name'")
        
        with col2:
            model_filename = st.text_input("Model Filename", value=HF_MODEL_FILE)
            st.info("The filename to use when saving the model (e.g., 'model.pt')")
        
        # Save Model Section
        st.subheader("Save Model to Hugging Face")
        
        save_col1, save_col2 = st.columns(2)
        with save_col1:
            model_to_save = st.radio(
                "Select model to save:",
                ["Initial Model", "Retrained Model"],
                disabled=(st.session_state.lstm_model is None and st.session_state.retrained_model is None)
            )
        
        with save_col2:
            if model_to_save == "Initial Model":
                st.info("This will save the initially trained model (before drift)")
                save_disabled = st.session_state.lstm_model is None
            else:
                st.info("This will save the retrained model (after drift detection)")
                save_disabled = st.session_state.retrained_model is None
        
        if st.button("Upload to Hugging Face", disabled=save_disabled):
            with st.spinner("Uploading model to Hugging Face..."):
                if model_to_save == "Initial Model":
                    model_to_upload = st.session_state.lstm_model
                else:
                    model_to_upload = st.session_state.retrained_model
                
                # Upload the model
                success = upload_model_to_huggingface(
                    model=model_to_upload,
                    model_name=model_filename,
                    repo_id=repo_id
                )
                
                if success:
                    st.success(f"Model successfully uploaded to {repo_id}")
                    
                    # Log model upload event to InfluxDB if connected
                    if st.session_state.influxdb_handler and st.session_state.influxdb_handler.connected:
                        with st.spinner("Logging model upload to InfluxDB..."):
                            try:
                                # Get the current timestamp
                                upload_time = datetime.now()
                                
                                # Write upload event to model_events
                                event_success = st.session_state.influxdb_handler.write_point(
                                    measurement="model_events",
                                    tags={
                                        "event_type": "model_uploaded",
                                        "model_name": "lstm_forecaster",
                                        "model_type": model_to_save,
                                        "repository": repo_id
                                    },
                                    fields={
                                        "filename": model_filename,
                                        "time_steps": model_to_upload.time_steps,
                                        "features": model_to_upload.features,
                                        "units": model_to_upload.units,
                                        "dropout_rate": float(model_to_upload.dropout_rate)
                                    },
                                    time=upload_time
                                )
                                
                                if event_success:
                                    st.success("Model upload event logged to InfluxDB")
                            except Exception as e:
                                st.error(f"Error writing to InfluxDB: {str(e)}")
                else:
                    st.error("Failed to upload model. Please check your Hugging Face credentials and repository settings.")
        
        # Load Model Section
        st.subheader("Load Model from Hugging Face")
        
        if st.button("Check if Model Exists"):
            with st.spinner("Checking Hugging Face repository..."):
                model_exists = check_model_exists_in_huggingface(
                    model_name=model_filename, 
                    repo_id=repo_id
                )
                
                if model_exists:
                    st.success(f"Model found in repository {repo_id}")
                else:
                    st.warning(f"Model not found in repository {repo_id}")
        
        if st.button("Load Model from Hugging Face"):
            with st.spinner("Downloading model from Hugging Face..."):
                try:
                    # Download the model
                    loaded_model = download_model_from_huggingface(
                        model_class=TimeSeriesLSTM,
                        model_name=model_filename,
                        repo_id=repo_id
                    )
                    
                    # Store the loaded model in session state
                    st.session_state.lstm_model = loaded_model
                    
                    st.success(f"Model successfully loaded from {repo_id}")
                    
                    # Log model download event to InfluxDB if connected
                    if st.session_state.influxdb_handler and st.session_state.influxdb_handler.connected:
                        with st.spinner("Logging model download to InfluxDB..."):
                            try:
                                # Get the current timestamp
                                download_time = datetime.now()
                                
                                # Write download event to model_events
                                event_success = st.session_state.influxdb_handler.write_point(
                                    measurement="model_events",
                                    tags={
                                        "event_type": "model_downloaded",
                                        "model_name": "lstm_forecaster",
                                        "repository": repo_id
                                    },
                                    fields={
                                        "filename": model_filename,
                                        "time_steps": loaded_model.time_steps,
                                        "features": loaded_model.features,
                                        "units": loaded_model.units,
                                        "dropout_rate": float(loaded_model.dropout_rate)
                                    },
                                    time=download_time
                                )
                                
                                if event_success:
                                    st.success("Model download event logged to InfluxDB")
                            except Exception as e:
                                st.error(f"Error writing to InfluxDB: {str(e)}")
                    
                    # Display model information
                    st.subheader("Loaded Model Information")
                    st.write(f"Time Steps: {loaded_model.time_steps}")
                    st.write(f"Features: {loaded_model.features}")
                    st.write(f"LSTM Units: {loaded_model.units}")
                    st.write(f"Dropout Rate: {loaded_model.dropout_rate}")
                    
                except Exception as e:
                    st.error(f"Failed to load model: {str(e)}")



if __name__ == "__main__":
    main()
