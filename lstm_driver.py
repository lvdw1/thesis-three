#!/usr/bin/env python3
"""
lstm_driver.py - LSTM-based imitation learning for vehicle control

This script extends the existing neural network driver architecture by
implementing a Long Short-Term Memory (LSTM) network for sequence-based
predictions of steering, throttle, and brake values.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Import necessary components from the original driver
from nndriver import Processor, FeatureTransformer

# ---------------------------------------------------------------------
# ------------------ LSTM Model --------------------------------------
# ---------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self,
                 input_size=None,
                 hidden_size=64,
                 num_layers=2,
                 dropout=0.1,
                 sequence_length=10,
                 output_size=3,
                 bidirectional=False,
                 learning_rate_init=0.001,
                 max_iter=100000,
                 tol=1e-6,
                 random_state=42,
                 verbose=True):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        # These will be initialized when input_size is known
        self.lstm = None
        self.fc_steering = None
        self.fc_throttle = None
        self.fc_brake = None
        
        # Other attributes
        self.optimizer = None
        self.criterion = None
        self.input_cols = None
        self.output_cols = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.loss_history = []
        
        # Initialize model architecture if input_size is provided
        if input_size is not None:
            self._initialize_architecture()
        
        # Set random seed for PyTorch
        torch.manual_seed(random_state)
    
    def _initialize_architecture(self):
        """Initialize the LSTM architecture"""
        # Number of directions (1 for unidirectional, 2 for bidirectional)
        directions = 2 if self.bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Separate output heads for each control
        lstm_output_size = self.hidden_size * directions
        self.fc_steering = nn.Linear(lstm_output_size, 1)
        self.fc_throttle = nn.Linear(lstm_output_size, 1)
        self.fc_brake = nn.Linear(lstm_output_size, 1)
        
        # Move model to the appropriate device
        self.to(self.device)
    
    def forward(self, x):
        """Forward pass through the LSTM model"""
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM output
        lstm_out, _ = self.lstm(x)
        
        # We take only the last timestep's output for prediction
        last_output = lstm_out[:, -1, :]
        
        # Apply output heads
        steering = self.fc_steering(last_output)
        throttle = self.fc_throttle(last_output)
        brake = self.fc_brake(last_output)
        
        # Combine outputs
        return torch.cat((steering, throttle, brake), dim=1)
    
    def train_model(self, train_loader, val_loader=None, input_cols=None, output_cols=None):
        """
        Train the LSTM model using the provided DataLoader.
        
        Parameters:
        -----------
        train_loader : DataLoader
            PyTorch DataLoader containing training data
        val_loader : DataLoader, optional
            PyTorch DataLoader containing validation data
        input_cols : list, optional
            Names of input columns
        output_cols : list, optional
            Names of output columns
        """
        if input_cols is not None:
            self.input_cols = list(input_cols)
            self.input_size = len(self.input_cols)
        
        if output_cols is not None:
            self.output_cols = list(output_cols)
            self.output_size = len(self.output_cols)
        
        # Initialize model architecture if not already done
        if self.lstm is None:
            self._initialize_architecture()
        
        # Initialize weights for better convergence
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                # Check if we're on MPS (Apple Silicon)
                if self.device.type == 'mps':
                    # Use xavier_uniform for all weights on MPS to avoid unsupported ops
                    for name, param in m.named_parameters():
                        if 'weight' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'bias' in name:
                            nn.init.zeros_(param)
                else:
                    # On other devices, we can use orthogonal init for hidden-to-hidden weights
                    nn.init.xavier_uniform_(m.weight_ih_l0)
                    nn.init.orthogonal_(m.weight_hh_l0)
                    
                    # Initialize biases
                    nn.init.zeros_(m.bias_ih_l0)
                    nn.init.zeros_(m.bias_hh_l0)
                    
                    # If bidirectional, initialize the reverse direction weights too
                    if m.bidirectional:
                        nn.init.xavier_uniform_(m.weight_ih_l0_reverse)
                        nn.init.orthogonal_(m.weight_hh_l0_reverse)
                        nn.init.zeros_(m.bias_ih_l0_reverse)
                        nn.init.zeros_(m.bias_hh_l0_reverse)
                    
                    # Initialize weights for additional layers if num_layers > 1
                    for layer in range(1, self.num_layers):
                        layer_str = f'l{layer}'
                        nn.init.xavier_uniform_(getattr(m, f'weight_ih_{layer_str}'))
                        nn.init.orthogonal_(getattr(m, f'weight_hh_{layer_str}'))
                        nn.init.zeros_(getattr(m, f'bias_ih_{layer_str}'))
                        nn.init.zeros_(getattr(m, f'bias_hh_{layer_str}'))
                        
                        if m.bidirectional:
                            nn.init.xavier_uniform_(getattr(m, f'weight_ih_{layer_str}_reverse'))
                            nn.init.orthogonal_(getattr(m, f'weight_hh_{layer_str}_reverse'))
                            nn.init.zeros_(getattr(m, f'bias_ih_{layer_str}_reverse'))
                            nn.init.zeros_(getattr(m, f'bias_hh_{layer_str}_reverse'))
        
        self.apply(init_weights)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate_init,
            weight_decay=0.0001  # L2 regularization
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=self.verbose
        )
        
        # Use MSE loss
        self.criterion = nn.MSELoss()
        
        # Train the model
        self.train()  # Put in training mode
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10  # Early stopping patience
        
        for epoch in range(self.max_iter):
            # Training loop
            train_loss = 0.0
            self.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.loss_history.append(avg_train_loss)
            
            # Validation loop
            if val_loader is not None:
                val_loss = 0.0
                self.eval()
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self(inputs)
                        loss = self.criterion(outputs, targets)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                
                # Learning rate scheduling based on validation loss
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if self.verbose:
                            print(f'Early stopping at epoch {epoch+1}')
                        break
            else:
                # If no validation set, schedule based on training loss
                scheduler.step(avg_train_loss)
            
            # Print progress
            if self.verbose and (epoch + 1) % 1 == 0:
                if val_loader is not None:
                    print(f'Epoch [{epoch+1}/{self.max_iter}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
                else:
                    print(f'Epoch [{epoch+1}/{self.max_iter}], Train Loss: {avg_train_loss:.6f}')
            
            # Convergence check
            if epoch > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < self.tol:
                if self.verbose:
                    print(f'Converged at epoch {epoch+1}')
                break
    
    def predict(self, test_loader):
        """
        Make predictions using the trained LSTM model.
        
        Parameters:
        -----------
        test_loader : DataLoader
            PyTorch DataLoader containing test data
        
        Returns:
        --------
        numpy.ndarray
            Model predictions
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self(inputs).cpu().numpy()
                predictions.append(outputs)
        
        return np.vstack(predictions)
    
    def predict_from_dataframe(self, df, sequence_length=None):
        """
        Make predictions from a DataFrame, handling the sequence creation.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing features
        sequence_length : int, optional
            Length of sequences to create (defaults to self.sequence_length)
        
        Returns:
        --------
        numpy.ndarray
            Model predictions
        """
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        if self.input_cols is None:
            raise RuntimeError("Model not trained yet: input_cols is None")
        
        # Create sequences
        sequences = []
        for i in range(len(df) - sequence_length + 1):
            seq = df.iloc[i:i+sequence_length][self.input_cols].values
            sequences.append(seq)
        
        if not sequences:
            return np.array([])
        
        # Convert to tensor
        X = torch.FloatTensor(np.array(sequences)).to(self.device)
        
        # Make predictions
        self.eval()
        with torch.no_grad():
            preds = self(X).cpu().numpy()
        
        # For the first (sequence_length-1) rows, we can't make predictions
        # So we'll pad with zeros
        padding = np.zeros((sequence_length - 1, self.output_size))
        return np.vstack([padding, preds])
    
    def evaluate(self, test_loader):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        test_loader : DataLoader
            PyTorch DataLoader containing test data
        
        Returns:
        --------
        float
            Mean squared error
        """
        self.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                preds = self(inputs).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.numpy())
        
        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)
        
        return mean_squared_error(y_true, y_pred)
    
    def get_loss(self):
        """Get the latest loss value"""
        return self.loss_history[-1] if self.loss_history else float('inf')
    
    def save(self, path="lstm_model.pt"):
        """Save the model to a file"""
        if self.lstm is not None:
            # Save both model state and metadata
            state_dict = self.state_dict()
            metadata = {
                'input_cols': self.input_cols,
                'output_cols': self.output_cols,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'sequence_length': self.sequence_length,
                'bidirectional': self.bidirectional,
                'loss_history': self.loss_history
            }
            torch.save({'state_dict': state_dict, 'metadata': metadata}, path)
    
    def load(self, path="lstm_model.pt"):
        """Load the model from a file"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load metadata
        metadata = checkpoint['metadata']
        self.input_cols = metadata['input_cols']
        self.output_cols = metadata['output_cols']
        self.input_size = metadata['input_size']
        self.hidden_size = metadata['hidden_size']
        self.num_layers = metadata['num_layers']
        self.dropout = metadata['dropout']
        self.sequence_length = metadata['sequence_length']
        self.bidirectional = metadata['bidirectional']
        self.loss_history = metadata.get('loss_history', [])
        
        # Initialize architecture
        self._initialize_architecture()
        
        # Load state dict
        self.load_state_dict(checkpoint['state_dict'])
        self.eval()

# ---------------------------------------------------------------------
# ------------------ Sequence Data Creation Utils ---------------------
# ---------------------------------------------------------------------
def create_sequences(df, sequence_length, input_cols, output_cols):
    """
    Create sequences from DataFrame for LSTM training.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    sequence_length : int
        Length of sequences to create
    input_cols : list
        Names of input columns
    output_cols : list
        Names of output columns
    
    Returns:
    --------
    tuple
        (X, y) NumPy arrays containing sequences and their targets
    """
    X = []
    y = []
    
    for i in range(len(df) - sequence_length):
        # Input sequence
        X.append(df.iloc[i:i+sequence_length][input_cols].values)
        # Target (the values right after the sequence)
        y.append(df.iloc[i+sequence_length][output_cols].values)
    
    return np.array(X), np.array(y)

def create_data_loaders(X, y, batch_size=8, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create PyTorch DataLoaders from sequences.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input sequences
    y : numpy.ndarray
        Target values
    batch_size : int
        Batch size for DataLoaders
    test_size : float
        Proportion of data to use for testing
    val_size : float
        Proportion of training data to use for validation
    random_state : int
        Random seed for splitting
    
    Returns:
    --------
    tuple
        (train_loader, val_loader, test_loader)
    """
    # First split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Then split train into train and validation
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state)
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, val_loader, test_loader
    else:
        # Create PyTorch datasets
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        return train_loader, None, test_loader

# ---------------------------------------------------------------------
# ------------------ LSTM Trainer Class ------------------------------
# ---------------------------------------------------------------------
class LSTMTrainer:
    """Class for training the LSTM model"""
    
    def __init__(self, processor, transformer, lstm_model):
        self.processor = processor
        self.transformer = transformer
        self.lstm_model = lstm_model
    
    def train(self, data, json_path=None, output_csv_path=None, pca_variance=0.99, 
              sequence_length=10, test_split=0.2, val_split=0.1, batch_size=8, use_postprocessed=False):
        """
        Train the LSTM model.
        
        Parameters:
        -----------
        data : dict or pandas.DataFrame
            Raw data dictionary or preprocessed DataFrame
        json_path : str, optional
            Path to JSON track file (required if use_postprocessed=False)
        output_csv_path : str, optional
            Path to save processed features
        pca_variance : float
            Explained variance ratio for PCA
        sequence_length : int
            Length of sequences for LSTM
        test_split : float
            Proportion of data to use for testing
        val_split : float
            Proportion of training data to use for validation
        batch_size : int
            Batch size for DataLoaders
        use_postprocessed : bool
            If True, assumes data is already processed
        """
        print("[LSTMTrainer] Training mode...")
        
        if not use_postprocessed:
            if json_path is None:
                raise ValueError("JSON path must be provided when use_postprocessed=False")
            
            track_data = self.processor.build_track_data(json_path)
            df_features = self.processor.process_csv(data, track_data)
            df_features = df_features.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
        else:
            df_features = data.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
        
        # Fit transformer (scaler + PCA) on features
        df_trans = self.transformer.fit_transform(
            df_features,
            exclude_cols=["steering", "throttle", "brake"],
            pca_variance=pca_variance
        )
        
        if output_csv_path:
            df_trans.to_csv(output_csv_path, index=False)
            print(f"[LSTMTrainer] Processed CSV saved to {output_csv_path}")
        
        self.transformer.save("lstm_transformer.joblib")
        print("[LSTMTrainer] Transformer saved.")
        
        # Prepare sequences for LSTM
        pc_cols = [c for c in df_trans.columns if c.startswith("PC")]
        out_cols = ["steering", "throttle", "brake"]
        
        # Create sequences
        X_sequences, y_targets = create_sequences(df_trans, sequence_length, pc_cols, out_cols)
        
        # Create DataLoaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_sequences, y_targets, 
            batch_size=batch_size, 
            test_size=test_split, 
            val_size=val_split
        )
        
        # Set sequence length in model
        self.lstm_model.sequence_length = sequence_length
        
        # Train LSTM model
        self.lstm_model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            input_cols=pc_cols,
            output_cols=out_cols
        )
        
        self.lstm_model.save("lstm_model.pt")
        print("[LSTMTrainer] LSTM model saved.")
        
        # Evaluate on test set
        test_mse = self.lstm_model.evaluate(test_loader)
        print(f"[LSTMTrainer] Test MSE: {test_mse:.6f}")
        print(f"[LSTMTrainer] Final Loss: {self.lstm_model.get_loss():.6f}")

# ---------------------------------------------------------------------
# ------------------ LSTM Driver Class -------------------------------
# ---------------------------------------------------------------------
class LSTMDriver:
    """
    Handles sequence-based inference using the LSTM model.
    """
    
    def __init__(self, processor, transformer, lstm_model, output_csv=None):
        self.processor = processor
        self.transformer = transformer
        self.lstm_model = lstm_model
        self.output_csv = output_csv
        self.sequence_buffer = []
    
    def inference_mode(self, csv_path, json_path):
        """
        Run inference on a CSV file.
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file with test data
        json_path : str
            Path to JSON track file
        """
        print("[LSTMDriver] Inference mode...")
        self.transformer.load("lstm_transformer.joblib")
        self.lstm_model.load("lstm_model.pt")
        
        # Read and process data
        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return
        
        track_data = self.processor.build_track_data(json_path)
        df_features = self.processor.process_csv(data_dict, track_data)
        
        # Transform features
        df_features_no_pos = df_features.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
        df_trans = self.transformer.transform(df_features_no_pos)
        
        # Generate predictions
        predictions = self.lstm_model.predict_from_dataframe(df_trans)
        
        # Collect results
        results = []
        times = df_features["time"].values
        
        for i in range(len(predictions)):
            if i < len(times):
                t = times[i]
                if i < self.lstm_model.sequence_length - 1:
                    # For the first (sequence_length-1) rows, we'll use actual values as placeholders
                    st = df_features.iloc[i]["steering"]
                    th = df_features.iloc[i]["throttle"]
                    br = df_features.iloc[i]["brake"]
                else:
                    # Use LSTM predictions
                    st, th, br = predictions[i]
                
                results.append({
                    "time": t,
                    "steering": st,
                    "throttle": th,
                    "brake": br
                })
        
        print("[LSTMDriver] Inference complete.")
        
        # Save predictions to CSV if output path is provided
        if self.output_csv is not None:
            df_results = pd.DataFrame(results)
            df_results.to_csv(self.output_csv, index=False)
            print(f"[Inference] Predictions saved to {self.output_csv}")
    
    def realtime_mode(self, json_path, host='127.0.0.1', port=65432):
        """
        Run the LSTM model in realtime mode.
        
        Parameters:
        -----------
        json_path : str
            Path to JSON track file
        host : str
            Host for TCP server
        port : int
            Port for TCP server
        """
        print("[LSTMDriver] Realtime mode...")
        self.transformer.load("lstm_transformer.joblib")
        self.lstm_model.load("lstm_model.pt")
        
        track_data = self.processor.build_track_data(json_path)
        
        # Initialize sequence buffer
        self.sequence_buffer = []
        
        # Set up TCP server
        import socket
        import time
        
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"[Realtime] Server listening on {host}:{port}...")
        
        client_socket, addr = server_socket.accept()
        print(f"[Realtime] Connection from {addr}")
        
        all_frames = []
        
        try:
            while True:
                raw_data = client_socket.recv(1024).decode('utf-8').strip()
                if not raw_data:
                    break
                
                fields = raw_data.split(',')
                
                sensor_data = {
                    "time": time.time(),
                    "x_pos": float(fields[0]),
                    "z_pos": float(fields[1]),
                    "yaw_angle": -float(fields[2])+90,
                    "long_vel": float(fields[3]),
                    "lat_vel": float(fields[4]),
                    "yaw_rate": float(fields[5]),
                    "steering": None,
                    "throttle": None,
                    "brake": None
                }
                
                # Process frame
                frame = self.processor.process_frame(sensor_data, track_data)
                all_frames.append(frame)
                
                # Transform frame
                df_single = pd.DataFrame([frame])
                df_features = df_single.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
                df_trans = self.transformer.transform(df_features)
                
                # Add to sequence buffer
                self.sequence_buffer.append(df_trans.values[0])
                
                # Keep only the most recent sequence_length frames
                if len(self.sequence_buffer) > self.lstm_model.sequence_length:
                    self.sequence_buffer.pop(0)
                
                # Make prediction if we have enough data
                if len(self.sequence_buffer) == self.lstm_model.sequence_length:
                    # Create sequence tensor
                    sequence = np.array([self.sequence_buffer])
                    sequence_tensor = torch.FloatTensor(sequence).to(self.lstm_model.device)
                    
                    # Make prediction
                    with torch.no_grad():
                        prediction = self.lstm_model(sequence_tensor)[0].cpu().numpy()
                    
                    st_pred, th_pred, br_pred = prediction
                    
                    # Apply some basic post-processing
                    if br_pred < 0.2:
                        br_pred = 0.0
                else:
                    # Not enough data yet, use default values
                    st_pred, th_pred, br_pred = 0.0, 0.5, 0.0
                
                # Send prediction
                message = f"{st_pred},{th_pred},{br_pred}\n"
                print(f"[Realtime] Sending: {message.strip()}")
                client_socket.sendall(message.encode())
                
        except Exception as e:
            print(f"[Realtime] Error: {e}")
        finally:
            client_socket.close()
            server_socket.close()
            print("[LSTMDriver] Server closed.")
            
            if self.output_csv:
                df_all = pd.DataFrame(all_frames)
                df_all.to_csv(self.output_csv, index=False)
                print(f"[Realtime] Processed CSV saved to {self.output_csv}")

# ---------------------------------------------------------------------
# ------------------ Visualizer Extensions ---------------------------
# ---------------------------------------------------------------------
class LSTMVisualizer:
    """
    Visualizes LSTM predictions against ground truth.
    """
    
    def __init__(self, processor=None, transformer=None, lstm_model=None):
        self.processor = processor
        self.transformer = transformer
        self.lstm_model = lstm_model
    
    def visualize_predictions(self, actual_csv_path, predicted_csv_path=None):
        """
        Visualize actual vs. predicted control values.
        
        Parameters:
        -----------
        actual_csv_path : str
            Path to CSV with actual values
        predicted_csv_path : str, optional
            Path to CSV with predicted values (if None, will use lstm_model to generate predictions)
        """
        import matplotlib.pyplot as plt
        
        # Read actual data
        df_actual = pd.read_csv(actual_csv_path)
        
        if predicted_csv_path:
            # Use provided predictions
            df_pred = pd.read_csv(predicted_csv_path)
        else:
            # Generate predictions using the model
            if self.processor is None or self.transformer is None or self.lstm_model is None:
                raise ValueError("processor, transformer, and lstm_model must be provided if predicted_csv_path is None")
            
            self.transformer.load("lstm_transformer.joblib")
            self.lstm_model.load("lstm_model.pt")
            
            # Process and transform features
            track_data = self.processor.build_track_data(json_path)
            df_features = self.processor.process_csv(df_actual, track_data)
            df_features_no_pos = df_features.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
            df_trans = self.transformer.transform(df_features_no_pos)
            
            # Generate predictions
            predictions = self.lstm_model.predict_from_dataframe(df_trans)
            
            # Create DataFrame with predictions
            df_pred = pd.DataFrame({
                "time": df_actual["time"],
                "steering": predictions[:, 0],
                "throttle": predictions[:, 1],
                "brake": predictions[:, 2]
            })
        
        # Create plots
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        # Plot steering
        axs[0].plot(df_actual["time"], df_actual["steering"], label="Actual", color="blue")
        axs[0].plot(df_pred["time"], df_pred["steering"], label="Predicted", color="red", linestyle="--")
        axs[0].set_ylabel("Steering")
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot throttle
        axs[1].plot(df_actual["time"], df_actual["throttle"], label="Actual", color="blue")
        axs[1].plot(df_pred["time"], df_pred["throttle"], label="Predicted", color="red", linestyle="--")
        axs[1].set_ylabel("Throttle")
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot brake
        axs[2].plot(df_actual["time"], df_actual["brake"], label="Actual", color="blue")
        axs[2].plot(df_pred["time"], df_pred["brake"], label="Predicted", color="red", linestyle="--")
        axs[2].set_ylabel("Brake")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, actual_csv_path, lstm_pred_path, nn_pred_path):
        """
        Compare predictions from LSTM and feedforward NN models.
        
        Parameters:
        -----------
        actual_csv_path : str
            Path to CSV with actual values
        lstm_pred_path : str
            Path to CSV with LSTM predictions
        nn_pred_path : str
            Path to CSV with feedforward NN predictions
        """
        import matplotlib.pyplot as plt
        
        # Read data
        df_actual = pd.read_csv(actual_csv_path)
        df_lstm = pd.read_csv(lstm_pred_path)
        df_nn = pd.read_csv(nn_pred_path)
        
        # Create plots
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot steering
        axs[0].plot(df_actual["time"], df_actual["steering"], label="Actual", color="black")
        axs[0].plot(df_lstm["time"], df_lstm["steering"], label="LSTM", color="blue")
        axs[0].plot(df_nn["time"], df_nn["steering"], label="Feedforward NN", color="red", linestyle="--")
        axs[0].set_ylabel("Steering")
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot throttle
        axs[1].plot(df_actual["time"], df_actual["throttle"], label="Actual", color="black")
        axs[1].plot(df_lstm["time"], df_lstm["throttle"], label="LSTM", color="blue")
        axs[1].plot(df_nn["time"], df_nn["throttle"], label="Feedforward NN", color="red", linestyle="--")
        axs[1].set_ylabel("Throttle")
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot brake
        axs[2].plot(df_actual["time"], df_actual["brake"], label="Actual", color="black")
        axs[2].plot(df_lstm["time"], df_lstm["brake"], label="LSTM", color="blue")
        axs[2].plot(df_nn["time"], df_nn["brake"], label="Feedforward NN", color="red", linestyle="--")
        axs[2].set_ylabel("Brake")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate and print metrics
        print("=== LSTM Model Performance ===")
        lstm_steering_mse = mean_squared_error(df_actual["steering"], df_lstm["steering"])
        lstm_throttle_mse = mean_squared_error(df_actual["throttle"], df_lstm["throttle"])
        lstm_brake_mse = mean_squared_error(df_actual["brake"], df_lstm["brake"])
        print(f"Steering MSE: {lstm_steering_mse:.6f}")
        print(f"Throttle MSE: {lstm_throttle_mse:.6f}")
        print(f"Brake MSE: {lstm_brake_mse:.6f}")
        print(f"Average MSE: {(lstm_steering_mse + lstm_throttle_mse + lstm_brake_mse)/3:.6f}")
        
        print("\n=== Feedforward NN Model Performance ===")
        nn_steering_mse = mean_squared_error(df_actual["steering"], df_nn["steering"])
        nn_throttle_mse = mean_squared_error(df_actual["throttle"], df_nn["throttle"])
        nn_brake_mse = mean_squared_error(df_actual["brake"], df_nn["brake"])
        print(f"Steering MSE: {nn_steering_mse:.6f}")
        print(f"Throttle MSE: {nn_throttle_mse:.6f}")
        print(f"Brake MSE: {nn_brake_mse:.6f}")
        print(f"Average MSE: {(nn_steering_mse + nn_throttle_mse + nn_brake_mse)/3:.6f}")

# ---------------------------------------------------------------------
# ------------------ Main Function -----------------------------------
# ---------------------------------------------------------------------
def read_csv_data(file_path):
    """
    Read CSV data using the utility function from the original driver.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
    
    Returns:
    --------
    dict
        Dictionary containing data arrays
    """
    from utils import read_csv_data as original_read_csv_data
    return original_read_csv_data(file_path)

def main():
    parser = argparse.ArgumentParser(description="LSTM-based imitation learning for vehicle control")
    parser.add_argument("--mode", type=str, default="train",
                      help="train, infer, realtime, or visualize")
    parser.add_argument("--csv", type=str,
                      help="Path to CSV file(s) (can include wildcards in quotes)")
    parser.add_argument("--json", type=str, default="default.json",
                      help="Path to track JSON file(s) (can include wildcards in quotes)")
    parser.add_argument("--transformer", type=str, default="lstm_transformer.joblib",
                      help="Path to save/load the fitted scaler/PCA")
    parser.add_argument("--model", type=str, default="lstm_model.pt",
                      help="Path to save/load the trained LSTM model")
    parser.add_argument("--output_csv", type=str, default=None,
                      help="Output path for processed or predicted data")
    parser.add_argument("--sequence_length", type=int, default=10,
                      help="Sequence length for LSTM")
    parser.add_argument("--hidden_size", type=int, default=64,
                      help="Hidden size for LSTM")
    parser.add_argument("--num_layers", type=int, default=2,
                      help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout rate for LSTM")
    parser.add_argument("--bidirectional", action="store_true",
                      help="Use bidirectional LSTM")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Initial learning rate")
    parser.add_argument("--max_epochs", type=int, default=100,
                      help="Maximum number of training epochs")
    parser.add_argument("--compare_csv", type=str, default=None,
                      help="Path to CSV with predictions from another model for comparison")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                      help="TCP server host for realtime mode")
    parser.add_argument("--port", type=int, default=65432,
                      help="TCP server port for realtime mode")
    parser.add_argument("--option", type=str, default="onthefly",
                      help="'onthefly' or 'postprocessed'")
    args = parser.parse_args()
    
    # Create common components
    processor = Processor()
    transformer = FeatureTransformer()
    
    # Create LSTM model with specified parameters
    lstm_model = LSTMModel(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        bidirectional=args.bidirectional,
        learning_rate_init=args.learning_rate,
        max_iter=args.max_epochs,
        verbose=True
    )
    
    use_postprocessed = (args.option.lower() == 'postprocessed')
    
    # Process command based on mode
    if args.mode == "train":
        if not args.csv:
            print("Must provide --csv for training.")
            return
        
        # Expand wildcards in the CSV pattern
        import glob
        csv_files = glob.glob(args.csv)
        if not csv_files:
            print(f"No CSV files match the pattern: {args.csv}")
            return
        
        if use_postprocessed:
            # For postprocessed mode, read DataFrames directly
            training_dfs = [pd.read_csv(f) for f in csv_files]
            unified_training_data = pd.concat(training_dfs, ignore_index=True)
        else:
            # For on-the-fly mode, process each CSV with its JSON
            import re
            training_dfs = []
            
            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                match = re.search(r'track(\d+)', filename)
                if match:
                    track_number = match.group(1)
                else:
                    print(f"Could not extract track number from {filename}, skipping.")
                    continue
                
                json_file = args.json.replace('*', track_number)
                if not os.path.exists(json_file):
                    print(f"JSON file {json_file} not found for {csv_file}, skipping.")
                    continue
                
                data_dict = read_csv_data(csv_file)
                if data_dict is None:
                    print(f"Could not load CSV data from {csv_file}, skipping.")
                    continue
                
                track_data = processor.build_track_data(json_file)
                df_features = processor.process_csv(data_dict, track_data)
                training_dfs.append(df_features)
            
            if not training_dfs:
                print("No training data available, exiting.")
                return
            
            unified_training_data = pd.concat(training_dfs, ignore_index=True)
        
        print(f"Unified training data shape: {unified_training_data.shape}")
        
        # Create trainer and train
        trainer = LSTMTrainer(processor, transformer, lstm_model)
        trainer.train(
            data=unified_training_data,
            json_path=args.json if not use_postprocessed else None,
            output_csv_path=args.output_csv,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            use_postprocessed=use_postprocessed
        )
    
    elif args.mode == "infer":
        if not args.csv:
            print("Must provide --csv for inference.")
            return
        
        driver = LSTMDriver(processor, transformer, lstm_model, output_csv=args.output_csv)
        driver.inference_mode(csv_path=args.csv, json_path=args.json)
    
    elif args.mode == "realtime":
        driver = LSTMDriver(processor, transformer, lstm_model, output_csv=args.output_csv)
        driver.realtime_mode(json_path=args.json, host=args.host, port=args.port)
    
    elif args.mode == "visualize":
        if not args.csv:
            print("Must provide --csv with actual data for visualization.")
            return
        
        visualizer = LSTMVisualizer(processor, transformer, lstm_model)
        
        if args.compare_csv:
            # Compare LSTM with another model
            visualizer.compare_models(
                actual_csv_path=args.csv,
                lstm_pred_path=args.output_csv,
                nn_pred_path=args.compare_csv
            )
        else:
            # Just visualize LSTM predictions
            visualizer.visualize_predictions(
                actual_csv_path=args.csv,
                predicted_csv_path=args.output_csv
            )
    
    else:
        print("Unknown mode. Use --mode train, infer, realtime, or visualize.")

if __name__ == "__main__":
    main()
