# LSTM Time Series Forecasting with Drift Detection

This application demonstrates time series forecasting with LSTM neural networks, model drift detection, and automated retraining pipeline using InfluxDB 3 for data storage.

## Features

- Synthetic time series data generation
- LSTM-based time series forecasting
- Data drift injection and detection
- Automated model retraining when drift is detected
- InfluxDB 3 integration for data persistence
- Hugging Face integration for model storage and retrieval
- Streamlit web interface for visualization and interaction

## Prerequisites

- Python 3.11+ 
- Docker (for running InfluxDB 3)
- Hugging Face account (for model persistence)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Anaisdg/influxdb3_huggingface_forecasting_monitoring.git
cd influxdb3_huggingface_forecasting_monitoring
```

### 2. Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## InfluxDB 3 Setup

### 1. Start InfluxDB 3 Core with Docker

```bash
docker run -it --rm --name influxdb3core \
  -v ~/influxdb3/data:/var/lib/influxdb3 \
  -p 8181:8181 \
  quay.io/influxdb/influxdb3-core:latest serve \
  --node-id my_host \
  --object-store file \
  --data-dir /var/lib/influxdb3
```

### 2. Create a database

In a new terminal, run:

```bash
influxdb3 database create --name timeseries
```

### 3. Create an API token

```bash
influxdb3 create token
```

Save the generated token for later use.

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
# InfluxDB Configuration
INFLUXDB_HOST=http://localhost:8181
INFLUXDB_TOKEN=your_influxdb_token_here
INFLUXDB_DATABASE=timeseries
INFLUXDB_ORG=""

# Hugging Face Configuration
HF_TOKEN=your_huggingface_token_here
HF_REPO_ID=your_username/your_repo_name
```

You can also set these variables directly in your environment:

```bash
export INFLUXDB_HOST="http://localhost:8181"
export INFLUXDB_TOKEN="your_influxdb_token_here"
export INFLUXDB_DATABASE="timeseries"
export HF_TOKEN="your_huggingface_token_here"
export HF_REPO_ID="your_username/your_repo_name"
```

### Hugging Face Repository

1. Create a new Hugging Face repository to store your models
2. Update the `HF_REPO_ID` in `app.py` to your repository ID (format: "username/repo-name")

## Running the Application

```bash
streamlit run app.py
```

The application will be available at http://localhost:5000

## Workflow

1. **InfluxDB Configuration**: Connect to your InfluxDB instance
2. **Data Generation**: Generate synthetic time series data
3. **Initial Model Training**: Train the LSTM model on the generated data
4. **Drift Injection**: Inject drift into the data to simulate changing data patterns
5. **Drift Detection**: Detect the drift using MSRE or MSE metrics
6. **Model Retraining**: Retrain the model on the drifted data
7. **Model Persistence**: Save and load models to/from Hugging Face

## Project Structure

```
├── app.py                   # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── data_generator.py    # Data generation utilities
│   ├── drift_detection.py   # Drift detection algorithms
│   ├── huggingface_utils.py # Hugging Face integration
│   ├── influxdb_utils.py    # InfluxDB utilities
│   └── model.py             # LSTM model definition
├── plugins/                 # Optional InfluxDB plugins
├── .streamlit/
│   └── config.toml          # Streamlit configuration
└── requirements.txt         # Project dependencies
```

## Customization

- Adjust LSTM hyperparameters in the Streamlit UI
- Modify drift detection thresholds and metrics
- Change data generation parameters
- Connect to different InfluxDB instances

## License

MIT

## Acknowledgements

- InfluxData for InfluxDB 3
- Hugging Face for model hosting
- PyTorch deep learning frameworks
- Streamlit for the web interface
