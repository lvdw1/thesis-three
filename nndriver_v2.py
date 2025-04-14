#!/usr/bin/env python3
"""
nndriver.py
"""
import os
import csv
import json
import math
import logging
import argparse
import copy  

import numpy as np
import pandas as pd
import joblib
import socket
import time

from utils import *
from transformer import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# ------------------ PyTorchNN Model ---------------------------------
# ---------------------------------------------------------------------
class NNModel(nn.Module):
    def __init__(self,
                 input_size=None,
                 hidden_layer_sizes=(26,20,12,8),
                 output_size=3,
                 alpha_value=0.001,
                 learning_rate_init=0.03,
                 max_iter=10000,
                 tol=1e-6,
                 random_state=42,
                 verbose=True,
                 early_stopping=False):
        super(NNModel, self).__init__()
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.alpha = alpha_value  # L2 regularization
        
        # Will be initialized when input_size is known
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize model architecture if input_size is provided
        if input_size is not None:
            self._initialize_architecture()
        
        # Other attributes
        self.optimizer = None
        self.criterion = None
        self.input_cols = None
        self.output_cols = None
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.loss_history = []

        if self.input_size is not None:
            self.to(self.device)
        
        # Set random seed for PyTorch
        torch.manual_seed(random_state)
    
    def _initialize_architecture(self):
        # Create layers for feature extraction
        feature_layers = []
        
        # Input layer
        feature_layers.append(nn.Linear(self.input_size, self.hidden_layer_sizes[0]))
        feature_layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(self.hidden_layer_sizes) - 1):
            feature_layers.append(nn.Linear(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i+1]))
            feature_layers.append(nn.ReLU())
        
        # Create sequential model for feature extraction
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # Separate output layers for each control
        self.steering_head = nn.Linear(self.hidden_layer_sizes[-1], 1)
        self.throttle_head = nn.Linear(self.hidden_layer_sizes[-1], 1)
        self.brake_head = nn.Linear(self.hidden_layer_sizes[-1], 1)
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply specific activations for each control
        steering = self.steering_head(features)
        throttle = self.throttle_head(features)
        brake = self.brake_head(features)
        
        # Combine outputs
        return torch.cat((steering, throttle, brake), dim=1)
    
    def train_model(self, df, y, df_val, y_val, input_cols=None, output_cols=None):
        """
        Trains the NN model using full-batch gradient descent and validation-based LR reduction.

        Args:
            df (DataFrame): Training features.
            y (ndarray): Training targets (steering, throttle, brake).
            df_val (DataFrame): Validation features.
            y_val (ndarray): Validation targets.
            input_cols (list): Optional list of columns for the input.
            output_cols (list): Optional list of columns for the output.
        """
        if input_cols is None:
            input_cols = list(df.columns)
        self.input_cols = list(input_cols)
        self.input_size = len(self.input_cols)

        # Store output_cols if provided
        self.output_cols = list(output_cols) if output_cols is not None else None
        self.output_size = y.shape[1]

        # Initialize model architecture if not already done
        if not hasattr(self, 'feature_extractor') or self.feature_extractor is None:
            self._initialize_architecture()
            self.to(self.device)

        # Convert data to PyTorch tensors (Training set)
        X = df[self.input_cols].values
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # Convert data to PyTorch tensors (Validation set)
        X_val = df_val[self.input_cols].values
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Initialize weights for better convergence
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init_weights)

        # Initialize optimizer 
        self.optimizer = optim.SGD(
            self.parameters(),
            lr=self.learning_rate_init,
            momentum=0.9,  
            weight_decay=self.alpha
        )

        # Scheduler to automatically reduce learning rate on plateau,
        # though we will also manually reduce it in the early stopping check.
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',         # We want to minimize the loss.
            factor=0.1,         # Multiply LR by this factor when triggered.
            patience=10,        # Number of epochs with no improvement (on val set) before LR is reduced.
        )

        # Use MSE loss
        self.criterion = nn.MSELoss()

        # Train the model
        self.train()  # Put in training mode

        # Early stopping parameters (modified to only reduce LR, not break)
        patience = 10
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(self.max_iter):
            # ----------------------
            # 1) Full-batch training
            # ----------------------
            self.optimizer.zero_grad()
            outputs = self(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

            # ----------------------
            # 2) Validation step
            # ----------------------
            self.eval()
            with torch.no_grad():
                val_outputs = self(X_val_tensor)
                val_loss = self.criterion(val_outputs, y_val_tensor)
            self.train()  # switch back to training mode

            # ----------------------
            # 3) Scheduler update (based on val loss)
            # ----------------------
            scheduler.step(val_loss.item())

            # ----------------------
            # 4) Monitor improvement and conditionally reduce LR
            # ----------------------
            if val_loss.item() < best_val_loss - self.tol:
                best_val_loss = val_loss.item()
                best_model_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Optionally print progress
            if self.verbose and (epoch + 1) % 100 == 0:
                current_lr = scheduler.optimizer.param_groups[0]['lr']
                self.loss_history.append(loss.item())
                print(f"Epoch [{epoch+1}/{self.max_iter}] | "
                      f"Train Loss: {loss.item():.6f} | "
                      f"Val Loss: {val_loss.item():.6f} | "
                      f"LR: {current_lr:.6f}")

            # Instead of early stopping, reduce learning rate manually if no improvement for 'patience' epochs
            if epochs_no_improve >= patience:
                break

        # Restore the best model parameters (lowest validation loss)
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
    
    def predict(self, df):
        if self.input_cols is None or not hasattr(self, 'feature_extractor') or self.feature_extractor is None:
            raise RuntimeError("NNModel not trained yet: input_cols is None or model is not initialized.")
        
        self.eval()
        X = df[self.input_cols].values
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            preds = self(X_tensor).cpu().numpy()
        
        return preds

    def evaluate(self, df, y_true):
        if not hasattr(self, 'feature_extractor') or self.feature_extractor is None:
            raise RuntimeError("NNModel not trained yet: model is not initialized.")
        
        self.eval()
        X = df[self.input_cols].values
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            preds = self(X_tensor).cpu().numpy()
        
        mse_value = mean_squared_error(y_true, preds)
        return mse_value   
    
    def get_loss(self):
        return self.loss_history[-1] if self.loss_history else float('inf')
    
    def save(self, path="nn_model_corrected_validation_double_005.pt"):
        if hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
            # Save both model state and metadata
            state_dict = self.state_dict()
            metadata = {
                'input_cols': self.input_cols,
                'output_cols': self.output_cols,
                'input_size': self.input_size,
                'hidden_layer_sizes': self.hidden_layer_sizes,
                'loss_history': self.loss_history
            }
            torch.save({'state_dict': state_dict, 'metadata': metadata}, path)
    
    def load(self, path="nn_model_corrected_validation_double_005.pt"):
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load metadata
        metadata = checkpoint['metadata']
        self.input_cols = metadata['input_cols']
        self.output_cols = metadata['output_cols']
        self.input_size = metadata['input_size']
        self.hidden_layer_sizes = metadata['hidden_layer_sizes']
        self.loss_history = metadata.get('loss_history', [])
        
        # Initialize architecture
        self._initialize_architecture()
        self.to(self.device)
        
        # Load state dict
        self.load_state_dict(checkpoint['state_dict'])
        self.eval()

###############################################
# 2. NNTrainer Class
###############################################
class NNTrainer:
    def __init__(self, processor, transformer, nn_model):
        self.processor = processor
        self.transformer = transformer
        self.nn_model = nn_model

    def train(self, data, json_path = None, output_csv_path=None, pca_variance=0.99, 
              test_split=0.01, val_split=0.2, use_postprocessed=False):
        """
        Trains the NN model with a dedicated validation set and an independent test set.

        Args:
            data (DataFrame or dict): Raw or pre-processed data.
            json_path (str): Path to a JSON track definition file (if needed).
            output_csv_path (str): Where to save the processed CSV.
            pca_variance (float): Variance to keep for PCA.
            test_split (float): Fraction of data to reserve for final testing.
            val_split (float): Fraction of the *training* data to reserve for validation.
            use_postprocessed (bool): If True, assume 'data' is already processed.
        """
        print("[NNTrainer] Training mode...")
        if not use_postprocessed:
            track_data = self.processor.build_track_data(json_path)
            df_features = self.processor.process_csv(data, track_data)
            # Drop columns not needed for training
            df_features = df_features.drop(columns=["time","x_pos", "z_pos", "yaw_angle"])
        else:
            df_features = data.drop(columns=["time","x_pos", "z_pos", "yaw_angle"])

        # Fit transformer (scaler + PCA) on features
        df_trans = self.transformer.fit_transform(
            df_features,
            exclude_cols=["steering", "throttle", "brake"],
            pca_variance=pca_variance
        )

        print(len(df_trans.columns), "features after PCA")


        if output_csv_path:
            df_trans.to_csv(output_csv_path, index=False)
            print(f"[NNTrainer] Processed CSV saved to {output_csv_path}")

        self.transformer.save()
        print("[NNTrainer] Transformer saved.")

        # Prepare data for model input
        pc_cols = [c for c in df_trans.columns if c.startswith("PC")]
        out_cols = ["steering", "throttle", "brake"]
        X = df_trans[pc_cols].values
        y = df_trans[out_cols].values

        # 1) Split for final test set
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

        # 2) Further split trainval into train and val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_split, random_state=42
        )

        # Train NN model (train + val)
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
        print("[NNTrainer] NN model saved.")

        # Evaluate on the test set
        mse_value = self.nn_model.evaluate(pd.DataFrame(X_test, columns=pc_cols), y_test)
        print("[NNTrainer] Test MSE:", mse_value)
        print("[NNTrainer] Final Loss:", self.nn_model.get_loss())
        
###############################################
# 3. NNDriver Class
###############################################
class NNDriver:
    """
    Handles realtime (or batch) inference:
      - Uses the Processor to extract features from each frame.
      - Uses the FeatureTransformer to transform features.
      - Returns NN model predictions.
    """
    def __init__(self, processor, transformer, nn_model, output_csv=None):
        self.processor = processor
        self.transformer = transformer
        self.nn_model = nn_model
        self.output_csv = output_csv  

    def inference_mode(self, csv_path, json_path):
        print("[NNDriver] Inference mode...")
        self.transformer.load()
        # Update to load PyTorch model
        self.nn_model.load()
        
        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return

        track_data = self.processor.build_track_data(json_path)
        df_features = self.processor.process_csv(data_dict, track_data)
        df_trans = self.transformer.transform(df_features)

        predictions = self.nn_model.predict(df_trans)
        times = df_features["time"].values
        
        # Collect prediction results
        results = []
        for i, (st, th, br) in enumerate(predictions):
            t = times[i]
            results.append({
                "time": t,
                "steering": st,
                "throttle": th,
                "brake": br
            })
        print("[NNDriver] Inference complete.")
        
        # If output_csv path is provided, write results to CSV.
        if self.output_csv is not None:
            df_results = pd.DataFrame(results)
            df_results.to_csv(self.output_csv, index=False)
            print(f"[Inference] Predictions saved to {self.output_csv}")

    def realtime_mode(self, json_path, host='127.0.0.1', port=65432):
        print("[NNDriver] Realtime mode...")
        self.transformer.load()
        # Update to load PyTorch model
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
                prev_time = time.time()
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
                    "yaw_angle": -float(fields[2])+90,
                    "long_vel": float(fields[3]),
                    "lat_vel": float(fields[4]),
                    "yaw_rate": float(fields[5]),
                    "steering": None,
                    "throttle": None,
                    "brake": None
                }
                frame = self.processor.process_frame(sensor_data, track_data)
                all_frames.append(frame)

                df_single = pd.DataFrame([frame])
                df_features = df_single.drop(columns=["time","x_pos", "z_pos", "yaw_angle"])

                df_trans = self.transformer.transform(df_features)
                prediction = self.nn_model.predict(df_trans)[0]
                st_pred, th_pred, br_pred = prediction
                if br_pred < 0.05:
                    br_pred = 0.0
                message = f"{st_pred},{th_pred},{br_pred}\n"
                curr_time = time.time()
                print(curr_time-prev_time)
                prev_time = curr_time
                print(f"[Realtime] Sending: {message.strip()}")
                client_socket.sendall(message.encode())
        except Exception as e:
            print(f"[Realtime] Error: {e}")
        finally:
            client_socket.close()
            server_socket.close()
            print("[NNDriver] Server closed.")
            if self.output_csv:
                df_all = pd.DataFrame(all_frames)
                df_all.to_csv(self.output_csv, index = False)
                print(f"[Realtime] Processed CSV saved to {self.output_csv}")
