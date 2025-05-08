# Quick Start Guide

This guide will help you get up and running with the LSTM Time Series Forecasting & Drift Detection application quickly.

## Prerequisites

- Python 3.11+
- Docker (for running Influxdb3 Core)
- A Hugging Face account (for model persistence)

## Step 1: Start InfluxDB 3 with Docker

Run the following command to start an InfluxDB 3 Core instance:

```bash
docker run -it --rm --name influxdb3core \
  -v ~/influxdb3/data:/var/lib/influxdb3 \
  -p 8181:8181 \
  quay.io/influxdb/influxdb3-core:latest serve \
  --node-id my_host \
  --object-store file \
  --data-dir /var/lib/influxdb3
```

## Step 2: Create a Database and Token

In a new terminal window, run:

```bash
# Create a database
influxdb3 create database timeseries

# Create a token
influxdb3 create token
```

Copy the token from the output for the next step.

## Step 3: Set Up Environment Variables

Create a `.env` file with your configuration:

```bash
# InfluxDB Configuration
INFLUXDB_HOST=http://localhost:8181
INFLUXDB_TOKEN=your_influxdb_token_here
INFLUXDB_DATABASE=timeseries
INFLUXDB_ORG=

# Hugging Face Configuration
HF_TOKEN=your_huggingface_token_here
HF_REPO_ID=your_username/your_repo_name
```

## Step 4: Run the Application

Run the application with:

```bash
streamlit run app.py
```

The application will be available at http://localhost:5000

## Step 5: Using the Application

1. In the "InfluxDB Configuration" tab, click "Connect to InfluxDB"
2. Generate synthetic data in the "Data Generation" tab
3. Train the initial model in the "Initial Model Training" tab
4. Inject drift in the "Drift Injection" tab
5. Detect drift in the "Drift Detection" tab
6. Retrain the model in the "Model Retraining" tab
7. Save and load models using the "Model Persistence" tab

## Troubleshooting

- If you see an InfluxDB connection error, check that the Docker container is running and your token is correct
- If models fail to save to Hugging Face, ensure your HF_TOKEN is valid and you have write permissions to the repository
- For more detailed information, refer to the full README.md file

## Next Steps

- Try adjusting the model hyperparameters to improve forecasting performance
- Experiment with different drift types and magnitudes
- Create your own custom time series data to test model adaptability