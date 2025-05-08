import os
import tempfile
import json
import torch
import requests
from pathlib import Path
import joblib

def upload_model_to_huggingface(model, model_name, repo_id, token=None):
    """
    Upload a trained PyTorch model to Hugging Face Hub.
    
    Parameters:
    -----------
    model : TimeSeriesLSTM
        The trained model to upload
    model_name : str
        Name of the model file to save (without path)
    repo_id : str
        Hugging Face repository ID (username/repo_name)
    token : str
        Hugging Face API token, if None will use HF_TOKEN environment variable
    
    Returns:
    --------
    bool
        True if upload was successful, False otherwise
    """
    # Get token from environment variable if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
        
    if token is None:
        raise ValueError("Hugging Face token not provided and HF_TOKEN environment variable not set")
    
    # Create a temporary directory to save model files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Paths for model and scaler
        model_path = os.path.join(tmp_dir, model_name)
        scaler_path = os.path.splitext(model_path)[0] + "_scaler.pkl"
        
        # Save model and scaler
        model.save_model(model_path, scaler_path)
        
        # Save model metadata
        metadata = {
            "time_steps": model.time_steps,
            "features": model.features,
            "units": model.units,
            "dropout_rate": model.dropout_rate,
            "learning_rate": model.learning_rate
        }
        
        metadata_path = os.path.join(tmp_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        
        # Upload files to Hugging Face
        api_url = f"https://huggingface.co/api/repos/{repo_id}/upload"
        headers = {"Authorization": f"Bearer {token}"}
        
        # Upload model
        with open(model_path, "rb") as f:
            model_data = f.read()
        
        response = requests.post(
            f"{api_url}/{model_name}",
            headers=headers,
            files={"file": model_data}
        )
        
        if response.status_code != 200:
            print(f"Error uploading model: {response.text}")
            return False
        
        # Upload scaler
        with open(scaler_path, "rb") as f:
            scaler_data = f.read()
        
        scaler_filename = os.path.basename(scaler_path)
        response = requests.post(
            f"{api_url}/{scaler_filename}",
            headers=headers,
            files={"file": scaler_data}
        )
        
        if response.status_code != 200:
            print(f"Error uploading scaler: {response.text}")
            return False
        
        # Upload metadata
        with open(metadata_path, "rb") as f:
            metadata_data = f.read()
        
        response = requests.post(
            f"{api_url}/metadata.json",
            headers=headers,
            files={"file": metadata_data}
        )
        
        if response.status_code != 200:
            print(f"Error uploading metadata: {response.text}")
            return False
        
        return True

def download_model_from_huggingface(model_class, model_name, repo_id, token=None):
    """
    Download a model from Hugging Face Hub.
    
    Parameters:
    -----------
    model_class : class
        The model class to instantiate
    model_name : str
        Name of the model file to download (without path)
    repo_id : str
        Hugging Face repository ID (username/repo_name)
    token : str
        Hugging Face API token, if None will use HF_TOKEN environment variable
    
    Returns:
    --------
    TimeSeriesLSTM
        Loaded model instance
    """
    # Get token from environment variable if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    # Create a temporary directory to save downloaded files
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Paths for model and scaler
        model_path = os.path.join(tmp_dir, model_name)
        scaler_path = os.path.splitext(model_path)[0] + "_scaler.pkl"
        metadata_path = os.path.join(tmp_dir, "metadata.json")
        
        # Download files from Hugging Face
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        # Download model
        model_url = f"https://huggingface.co/{repo_id}/resolve/main/{model_name}"
        response = requests.get(model_url, headers=headers)
        
        if response.status_code != 200:
            raise ValueError(f"Error downloading model: {response.text}")
        
        with open(model_path, "wb") as f:
            f.write(response.content)
        
        # Download scaler
        scaler_filename = os.path.basename(scaler_path)
        scaler_url = f"https://huggingface.co/{repo_id}/resolve/main/{scaler_filename}"
        response = requests.get(scaler_url, headers=headers)
        
        if response.status_code != 200:
            raise ValueError(f"Error downloading scaler: {response.text}")
        
        with open(scaler_path, "wb") as f:
            f.write(response.content)
        
        # Download metadata
        metadata_url = f"https://huggingface.co/{repo_id}/resolve/main/metadata.json"
        response = requests.get(metadata_url, headers=headers)
        
        # Create a new model instance
        model = model_class()
        
        # Load the model and scaler
        model.load_model(model_path, scaler_path)
        
        return model
        
def check_model_exists_in_huggingface(model_name, repo_id, token=None):
    """
    Check if a model exists in Hugging Face Hub.
    
    Parameters:
    -----------
    model_name : str
        Name of the model file to check (without path)
    repo_id : str
        Hugging Face repository ID (username/repo_name)
    token : str
        Hugging Face API token, if None will use HF_TOKEN environment variable
    
    Returns:
    --------
    bool
        True if model exists, False otherwise
    """
    # Get token from environment variable if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    # Check if model exists
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    model_url = f"https://huggingface.co/{repo_id}/resolve/main/{model_name}"
    response = requests.head(model_url, headers=headers)
    
    return response.status_code == 200