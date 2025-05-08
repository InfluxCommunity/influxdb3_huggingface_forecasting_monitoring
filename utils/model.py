import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib

class LSTMModel(nn.Module):
    """
    PyTorch LSTM model for time series forecasting.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_size)
        
        # Use the output from the last time step
        out = out[:, -1, :]  # out: (batch_size, hidden_size)
        out = self.dropout(out)
        out = self.fc(out)  # out: (batch_size, 1)
        
        return out

class TimeSeriesLSTM:
    """
    LSTM model for time series forecasting with utilities for training,
    prediction, saving, and loading models.
    """
    
    def __init__(self, time_steps=10, features=1, units=50, 
                 dropout_rate=0.2, learning_rate=0.001):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        time_steps : int
            Number of time steps for input sequences
        features : int
            Number of features for input data
        units : int
            Number of LSTM units in the model
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for optimization
        """
        self.time_steps = time_steps
        self.features = features
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = {"loss": [], "val_loss": []}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def build_model(self):
        """Build the LSTM model architecture."""
        # Create model if it doesn't exist
        if self.model is None:
            model = LSTMModel(
                input_size=self.features,
                hidden_size=self.units,
                dropout=self.dropout_rate
            )
            model = model.to(self.device)
            self.model = model
        
        return self.model
    
    def scale_data(self, data):
        """Scale the input data using MinMaxScaler."""
        return self.scaler.fit_transform(data.reshape(-1, 1)).reshape(data.shape)
    
    def inverse_scale(self, data):
        """Inverse scale the data back to original range."""
        return self.scaler.inverse_transform(data.reshape(-1, 1)).reshape(data.shape)
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.1, verbose=1,
             early_stopping_patience=10):
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        X : ndarray
            Input sequences
        y : ndarray
            Target values
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data to use for validation
        verbose : int
            Verbose mode (0=silent, 1=progress bar, 2=one line per epoch)
        early_stopping_patience : int
            Number of epochs with no improvement after which training will be stopped
            
        Returns:
        --------
        dict
            Training history
        """
        # Build model if not already built
        if self.model is None:
            model = self.build_model()
        
        # Convert numpy arrays to PyTorch tensors
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)
        
        # Split data for validation
        if validation_split > 0:
            val_size = int(len(X) * validation_split)
            train_size = len(X) - val_size
            
            # Use random_split for validation
            indices = torch.randperm(len(X))
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]
            
            train_dataset = TensorDataset(X_train, y_train.view(-1, 1))
            val_dataset = TensorDataset(X_val, y_val.view(-1, 1))
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            train_dataset = TensorDataset(X, y.view(-1, 1))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = None
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Initialize variables for early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Initialize history
        self.history = {"loss": [], "val_loss": []}
        
        # Train the model
        self.model.train()
        for epoch in range(epochs):
            # Training
            epoch_loss = 0.0
            for inputs, targets in train_loader:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Calculate average training loss
            avg_train_loss = epoch_loss / len(train_loader)
            self.history["loss"].append(avg_train_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self.model(inputs)
                        val_loss += criterion(outputs, targets).item()
                
                avg_val_loss = val_loss / len(val_loader)
                self.history["val_loss"].append(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose > 0:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
                
                if verbose > 0 and epoch % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if verbose > 0 and epoch % max(1, epochs // 10) == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.6f}")
        
        # Load the best model if early stopping occurred and validation was used
        if best_model_state is not None and val_loader is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : ndarray
            Input sequences for prediction
            
        Returns:
        --------
        ndarray
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert to PyTorch tensor
        X = torch.FloatTensor(X).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()
        
        return predictions.flatten()
    
    def forecast(self, initial_sequence, steps=10):
        """
        Generate multi-step forecasts.
        
        Parameters:
        -----------
        initial_sequence : ndarray
            Initial sequence to start forecasting from
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        ndarray
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make a copy of the initial sequence
        curr_sequence = initial_sequence.copy()
        forecasts = []
        
        self.model.eval()
        
        for _ in range(steps):
            # Get the last time_steps values
            sequence = curr_sequence[-self.time_steps:]
            
            # Reshape for prediction
            X_pred = sequence.reshape(1, self.time_steps, self.features)
            X_pred = torch.FloatTensor(X_pred).to(self.device)
            
            # Predict the next value
            with torch.no_grad():
                next_pred = self.model(X_pred).cpu().numpy()[0][0]
            
            # Add the prediction to the list
            forecasts.append(next_pred)
            
            # Update the sequence for the next prediction
            if self.features == 1:
                curr_sequence = np.append(curr_sequence, next_pred)
            else:
                # For multivariate case - this is a simplified approach
                # In a real application, you'd need to update all features
                new_point = np.zeros(self.features)
                new_point[0] = next_pred  # assuming first feature is the target
                curr_sequence = np.vstack([curr_sequence, new_point])
        
        return np.array(forecasts)
    
    def save_model(self, model_path, scaler_path=None):
        """
        Save the trained model and scaler.
        
        Parameters:
        -----------
        model_path : str
            Path to save the model
        scaler_path : str
            Path to save the scaler, if None will use model_path + '_scaler.pkl'
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        if scaler_path is None:
            scaler_path = os.path.splitext(model_path)[0] + '_scaler.pkl'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model and scaler
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'time_steps': self.time_steps,
            'features': self.features,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'history': self.history
        }, model_path)
        
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path, scaler_path=None):
        """
        Load a trained model and scaler.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model
        scaler_path : str
            Path to the saved scaler, if None will use model_path + '_scaler.pkl'
        """
        if scaler_path is None:
            scaler_path = os.path.splitext(model_path)[0] + '_scaler.pkl'
        
        # Load model and scaler
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model parameters
        self.time_steps = checkpoint['time_steps']
        self.features = checkpoint['features']
        self.units = checkpoint['units']
        self.dropout_rate = checkpoint['dropout_rate']
        self.learning_rate = checkpoint['learning_rate']
        self.history = checkpoint['history']
        
        # Build model
        self.build_model()
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        print(f"Model loaded from {model_path}")
        print(f"Scaler loaded from {scaler_path}")
    
    def plot_loss(self, figsize=(12, 6)):
        """
        Plot training and validation loss.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size for the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        if not self.history or len(self.history["loss"]) == 0:
            raise ValueError("No training history available. Train a model first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.history["loss"], label='Training Loss')
        
        if "val_loss" in self.history and len(self.history["val_loss"]) > 0:
            ax.plot(self.history["val_loss"], label='Validation Loss')
        
        ax.set_title('Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        
        return fig
