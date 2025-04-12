#!/usr/bin/env python3
"""
tfdriver.py
"""

import os
import csv
import json
import math
import logging
import argparse
import copy  
import socket
import time

import numpy as np
import pandas as pd
import joblib

from utils import *
from transformer import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------
# 1. Helper Function for Creating Sequences
# ---------------------------------------------------------------------
def create_sequences(X, y, seq_length):
    """
    Converts independent samples into sequences using a sliding window.
    Each sequence has a length of `seq_length`, and the target is the control
    outputs corresponding to the last frame of the sequence.
    """
    sequences = []
    targets = []
    for i in range(len(X) - seq_length + 1):
        seq = X[i : i + seq_length]
        target = y[i + seq_length - 1]  # target from the last frame in the sequence
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)   


# ---------------------------------------------------------------------
# 2. Positional Encoding (for providing sequence order)
# ---------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create constant 'pe' matrix with values dependent on position and dimension
        pe = torch.zeros(max_len, d_model)  # Shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # apply sin to even indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)  # apply cos to odd indices in the array
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------
# 3. Transformer-Based Model
# ---------------------------------------------------------------------
class TFModel(nn.Module):
    def __init__(self,
                 input_size,
                 seq_length=10,          # Number of frames (time steps) in each sequence
                 d_model=32,             # Embedding dimension for transformer
                 nhead=4,                # Number of attention heads
                 num_encoder_layers=3,   # Number of transformer encoder layers
                 dim_feedforward=64,    # Feedforward network size inside transformer encoder layers
                 dropout=0.1,
                 output_size=3,          # Steering, throttle, brake
                 learning_rate_init=0.01,
                 max_iter=200,
                 tol=1e-3,
                 random_state=42,
                 verbose=True,
                 early_stopping=False):
        super(TFModel, self).__init__()
        
        # Store architecture hyperparameters for later restoration.
        self.input_size = input_size
        self.seq_length = seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.output_size = output_size
        
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.loss_history = []
        
        self.input_cols = None
        self.output_cols = None

        # Initialize the network architecture
        self._initialize_architecture()
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(self.device)
        torch.manual_seed(random_state)
    
    def _initialize_architecture(self):
        """Initializes (or reinitializes) the transformer architecture."""
        # Input projection layer
        self.input_proj = nn.Linear(self.input_size, self.d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.dropout)
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_encoder_layers)
        # Decoder that maps the final transformer output to control predictions
        self.decoder = nn.Linear(self.d_model, self.output_size)
    
    def forward(self, x):
        """
        Forward pass expects input of shape (batch_size, seq_length, input_size).
        """
        x = self.input_proj(x)            # (batch_size, seq_length, d_model)
        x = self.pos_encoder(x)             # (batch_size, seq_length, d_model)
        # Transformer expects input as (seq_length, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)     # (seq_length, batch_size, d_model)
        # Use output from the last time step of the sequence
        x = x[-1, :, :]                    # (batch_size, d_model)
        output = self.decoder(x)            # (batch_size, output_size)
        return output

    def train_model(self, df, y, df_val, y_val, input_cols=None, output_cols=None):
        """
        Trains the transformer model using sequence data.
        The input DataFrames are converted into sequences (using a sliding window)
        and then trained using full‑batch gradient descent with early stopping.
        """
        if input_cols is None:
            input_cols = list(df.columns)
        self.input_cols = input_cols
        self.output_cols = list(output_cols) if output_cols is not None else None
        
        # Extract feature and target arrays
        X_train = df[self.input_cols].values  # shape: (num_samples, input_size)
        y_train = y                          # shape: (num_samples, output_size)
        X_val = df_val[self.input_cols].values
        y_val = y_val

        # Create sequences for transformer input
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, self.seq_length)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, self.seq_length)

        # Convert sequences to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)

        # Initialize weights for better convergence.
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_weights)
        
        # Setup optimizer and learning rate scheduler
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=self.learning_rate_init,
            momentum=0.9,
            weight_decay=0.001  # L2 regularization
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=10
        )
        
        self.criterion = nn.MSELoss()

        patience = 10
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(self.max_iter):
            self.train()
            self.optimizer.zero_grad()
            outputs = self(X_train_tensor)
            loss = self.criterion(outputs, y_train_tensor)
            loss.backward()
            self.optimizer.step()

            # Evaluate on validation set
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
            scheduler.step(val_loss.item())
            self.loss_history.append(loss.item())
            
            if self.verbose and ((epoch + 1) % 1 == 0):
                current_lr = scheduler.optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{self.max_iter}] | "
                      f"Train Loss: {loss.item():.6f} | "
                      f"Val Loss: {val_loss.item():.6f} | "
                      f"LR: {current_lr:.6f}")
            
            if val_loss.item() < best_val_loss - self.tol:
                best_val_loss = val_loss.item()
                best_model_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch+1} with val_loss={val_loss.item():.6f}")
                break
        
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

    def predict(self, df):
        """
        Predicts control outputs using the most recent sequence of frames.
        The DataFrame should be ordered chronologically and contain at least `seq_length` rows.
        """
        if self.input_cols is None:
            raise RuntimeError("TransformerModel not trained yet: input_cols is None or model is not initialized.")
        
        self.eval()
        X = df[self.input_cols].values  # shape: (num_frames, input_size)
        
        if len(X) < self.seq_length:
            raise ValueError("Insufficient data: need at least seq_length frames for prediction.")
        
        # Use the last seq_length frames to form a single sequence
        X_seq = X[-self.seq_length:].reshape(1, self.seq_length, -1)
        X_seq_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            preds = self(X_seq_tensor).cpu().numpy()
        
        return preds  # shape: (1, output_size)

    def evaluate(self, df, y_true):
        """
        Evaluates the model on the given test data by computing the mean squared error.
        """
        if self.input_cols is None:
            raise RuntimeError("TransformerModel not trained yet: input_cols is None or model is not initialized.")
        
        self.eval()
        X = df[self.input_cols].values  # shape: (num_samples, input_size)
        # Create sequences from test data
        X_seq, y_seq = create_sequences(X, y_true, self.seq_length)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        with torch.no_grad():
            preds = self(X_tensor).cpu().numpy()
        
        mse_value = mean_squared_error(y_seq, preds)
        return mse_value   

    def get_loss(self):
        return self.loss_history[-1] if self.loss_history else float('inf')
    
    def save(self, path="models/networks/transformer_v0.pt"):
        """
        Saves the model state along with important metadata.
        """
        metadata = {
            'input_cols': self.input_cols,
            'output_cols': self.output_cols,
            'input_size': self.input_size,
            'seq_length': self.seq_length,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'loss_history': self.loss_history
        }
        torch.save({'state_dict': self.state_dict(), 'metadata': metadata}, path)
    
    def load(self, path="models/networks/transformer_v0.pt"):
        """
        Loads the model state and metadata from a checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        metadata = checkpoint['metadata']
        self.input_cols = metadata['input_cols']
        self.output_cols = metadata['output_cols']
        self.input_size = metadata['input_size']
        self.seq_length = metadata['seq_length']
        self.d_model = metadata['d_model']
        self.nhead = metadata['nhead']
        self.num_encoder_layers = metadata['num_encoder_layers']
        self.dim_feedforward = metadata['dim_feedforward']
        self.dropout = metadata['dropout']
        self.loss_history = metadata.get('loss_history', [])
        
        # Reinitialize the architecture with the loaded parameters
        self._initialize_architecture()
        self.to(self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.eval()


# ---------------------------------------------------------------------
# 4. TFTrainer Class
# ---------------------------------------------------------------------
class TFTrainer:
    def __init__(self, processor, transformer_pipeline, nn_model):
        self.processor = processor
        self.transformer = transformer_pipeline
        self.nn_model = nn_model

    def train(self, data, json_path=None, output_csv_path=None, pca_variance=0.99, 
              test_split=0.01, val_split=0.2, use_postprocessed=False):
        """
        Trains the NN model with a dedicated validation set and an independent test set.
        """
        print("[TFTrainer] Training mode...")
        if not use_postprocessed:
            track_data = self.processor.build_track_data(json_path)
            df_features = self.processor.process_csv(data, track_data)
            # Drop columns not needed for training
            df_features = df_features.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
        else:
            df_features = data.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])

        # Fit transformer (e.g. scaler + PCA) on features
        df_trans = self.transformer.fit_transform(
            df_features,
            exclude_cols=["steering", "throttle", "brake"],
            pca_variance=pca_variance
        )
        if output_csv_path:
            df_trans.to_csv(output_csv_path, index=False)
            print(f"[TFTrainer] Processed CSV saved to {output_csv_path}")

        self.transformer.save()
        print("[TFTrainer] Transformer saved.")

        # Prepare data for model input
        pc_cols = [c for c in df_trans.columns if c.startswith("PC")]
        out_cols = ["steering", "throttle", "brake"]
        X = df_trans[pc_cols].values
        y = df_trans[out_cols].values

        # 1) Split for final test set
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

        # 2) Further split trainval into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_split, random_state=42
        )

        # Train the NN model using the training and validation splits
        self.nn_model.train_model(
            df=pd.DataFrame(X_train, columns=pc_cols),
            y=y_train,
            df_val=pd.DataFrame(X_val, columns=pc_cols),
            y_val=y_val,
            input_cols=pc_cols,
            output_cols=out_cols
        )

        # Save the trained NN model
        self.nn_model.save()
        print("[TFTrainer] NN model saved.")

        # Evaluate on the test set
        mse_value = self.nn_model.evaluate(pd.DataFrame(X_test, columns=pc_cols), y_test)
        print("[TFTrainer] Test MSE:", mse_value)
        print("[TFTrainer] Final Loss:", self.nn_model.get_loss())
        

# ---------------------------------------------------------------------
# 5. TFDriver Class
# ---------------------------------------------------------------------
class TFDriver:
    """
    Handles realtime (or batch) inference:
      - Uses the Processor to extract features from each frame.
      - Uses the FeatureTransformer (e.g. scaler/PCA) to transform features.
      - Returns NN model predictions.
    """
    def __init__(self, processor, transformer_pipeline, nn_model, output_csv=None):
        self.processor = processor
        self.transformer = transformer_pipeline
        self.nn_model = nn_model
        self.output_csv = output_csv  
        self.buffer = []  # rolling buffer for realtime sequence formation

    def inference_mode(self, csv_path, json_path):
        print("[TFDriver] Inference mode...")
        self.transformer.load()
        # Load the trained transformer model
        self.nn_model.load()
        
        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return

        track_data = self.processor.build_track_data(json_path)
        df_features = self.processor.process_csv(data_dict, track_data)
        df_trans = self.transformer.transform(df_features)

        # For batch inference, assume the CSV has enough rows (ordered chronologically)
        predictions = self.nn_model.predict(df_trans)
        times = df_features["time"].values
        
        # Collect prediction results
        results = []
        # Since predict returns one prediction from the last seq_length rows,
        # align the output with the corresponding time (last frame of the sequence)
        result_time = times[-1]
        st_pred, th_pred, br_pred = predictions[0]
        results.append({
            "time": result_time,
            "steering": st_pred,
            "throttle": th_pred,
            "brake": br_pred
        })
        print("[TFDriver] Inference complete.")
        
        # If output_csv path is provided, write results to CSV.
        if self.output_csv is not None:
            df_results = pd.DataFrame(results)
            df_results.to_csv(self.output_csv, index=False)
            print(f"[Inference] Predictions saved to {self.output_csv}")

    def realtime_mode(self, json_path, host='127.0.0.1', port=65432):
        """
        Realtime inference using a rolling buffer. For every new frame received,
        the buffer is updated. Once the buffer contains at least `seq_length` frames,
        a prediction is made.
        """
        print("[TFDriver] Realtime mode...")
        self.transformer.load()
        self.nn_model.load()
        
        track_data = self.processor.build_track_data(json_path)

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
                messages = raw_data.splitlines()
                latest = messages[-1].strip()
                fields = latest.split(',')
                sensor_data = {
                    "time": time.time(),
                    "x_pos": float(fields[0]),
                    "z_pos": float(fields[1]),
                    "yaw_angle": -float(fields[2]) + 90,
                    "long_vel": float(fields[3]),
                    "lat_vel": float(fields[4]),
                    "yaw_rate": float(fields[5]),
                    "steering": None,
                    "throttle": None,
                    "brake": None
                }
                frame = self.processor.process_frame(sensor_data, track_data)
                all_frames.append(frame)

                # Preprocess the frame (drop columns not used for prediction)
                df_single = pd.DataFrame([frame])
                df_features = df_single.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
                df_trans = self.transformer.transform(df_features)
                
                # Append the transformed row (as a dict) to the rolling buffer
                self.buffer.append(df_trans.iloc[0].to_dict())
                if len(self.buffer) > self.nn_model.seq_length:
                    self.buffer.pop(0)  # keep the buffer size equal to seq_length
                
                # Only run prediction when we have enough frames
                if len(self.buffer) == self.nn_model.seq_length:
                    df_seq = pd.DataFrame(self.buffer)
                    # Ensure the columns match
                    X_seq = df_seq[self.nn_model.input_cols].values.reshape(1, self.nn_model.seq_length, -1)
                    X_seq_tensor = torch.FloatTensor(X_seq).to(self.nn_model.device)
                    
                    with torch.no_grad():
                        prediction = self.nn_model(X_seq_tensor).cpu().numpy()[0]
                    st_pred, th_pred, br_pred = prediction
                    if br_pred < 0.05:
                        br_pred = 0.0
                    message = f"{st_pred},{th_pred},{br_pred}\n"
                    print(f"[Realtime] Sending: {message.strip()}")
                    client_socket.sendall(message.encode())
        except Exception as e:
            print(f"[Realtime] Error: {e}")
        finally:
            client_socket.close()
            server_socket.close()
            print("[TFDriver] Server closed.")
            if self.output_csv:
                df_all = pd.DataFrame(all_frames)
                df_all.to_csv(self.output_csv, index=False)
                print(f"[Realtime] Processed CSV saved to {self.output_csv}")
