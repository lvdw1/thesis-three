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
import numpy as np
import pandas as pd
import joblib
import socket
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------
# ------------------ FEATURE TRANSFORMER (SCALER + PCA) ---------------
# ---------------------------------------------------------------------

class FeatureTransformer:
    def __init__(self):
        self.scaler = None
        self.pca = None
        self.numeric_cols = []
        self.exclude_cols = []

    def fit_transform(self, df, exclude_cols=None, pca_variance=0.99):
        if exclude_cols is None:
            exclude_cols = ["steering", "throttle", "brake"]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
        self.numeric_cols = numeric_cols
        self.exclude_cols = exclude_cols
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(df[numeric_cols])
        self.pca = PCA(n_components=pca_variance)
        pca_data = self.pca.fit_transform(scaled_data)
        pc_columns = [f"PC{i+1}" for i in range(pca_data.shape[1])]
        df_pca = pd.DataFrame(pca_data, columns=pc_columns)
        df_excl = df[exclude_cols].reset_index(drop=True)
        df_out = pd.concat([df_pca, df_excl], axis=1)
        return df_out

    def transform(self, df):
        if self.scaler is None or self.pca is None:
            raise RuntimeError("FeatureTransformer not fitted yet!")
        # If missing columns, fill with 0.0
        for c in self.numeric_cols:
            if c not in df.columns:
                df[c] = 0.0
        scaled_data = self.scaler.transform(df[self.numeric_cols])
        pca_data = self.pca.transform(scaled_data)
        pc_columns = [f"PC{i+1}" for i in range(pca_data.shape[1])]
        df_pca = pd.DataFrame(pca_data, columns=pc_columns)
        df_excl = pd.DataFrame()
        for c in self.exclude_cols:
            if c in df.columns:
                df_excl[c] = df[c].values
        df_excl = df_excl.reset_index(drop=True)
        df_out = pd.concat([df_pca, df_excl], axis=1)
        return df_out

    def save(self, path="transformer.joblib"):
        joblib.dump({
            "scaler": self.scaler,
            "pca": self.pca,
            "numeric_cols": self.numeric_cols,
            "exclude_cols": self.exclude_cols
        }, path)

    def load(self, path="transformer.joblib"):
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.pca = data["pca"]
        self.numeric_cols = data["numeric_cols"]
        self.exclude_cols = data["exclude_cols"]

# ---------------------------------------------------------------------
# ------------------ PyTorchNN Model ---------------------------------
# ---------------------------------------------------------------------
class PyTorchNN(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes=(64, 56, 48, 24, 16, 8), output_size=3):
        super(PyTorchNN, self).__init__()
        
        # Create layers for feature extraction
        self.feature_layers = []
        
        # Input layer
        self.feature_layers.append(nn.Linear(input_size, hidden_layer_sizes[0]))
        self.feature_layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_layer_sizes) - 1):
            self.feature_layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            self.feature_layers.append(nn.ReLU())
        
        # Create sequential model for feature extraction
        self.feature_extractor = nn.Sequential(*self.feature_layers)
        
        # Separate output layers for each control
        self.steering_head = nn.Linear(hidden_layer_sizes[-1], 1)
        self.throttle_head = nn.Linear(hidden_layer_sizes[-1], 1)
        self.brake_head = nn.Linear(hidden_layer_sizes[-1], 1)
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Apply specific activations for each control
        steering = (self.steering_head(features))
        throttle = (self.throttle_head(features))
        brake = (self.brake_head(features))
        
        # Combine outputs
        return torch.cat((steering, throttle, brake), dim=1)
# ---------------------------------------------------------------------
# ------------------ NNModel ------------------------------------------
# ---------------------------------------------------------------------
class NNModel:
    def __init__(self,
                 hidden_layer_sizes=(64, 56, 48, 24, 16, 8),
                 alpha_value=0.001,
                 learning_rate='adaptive',
                 learning_rate_init=0.001,
                 max_iter=100000,
                 tol=1e-6,
                 random_state=42,
                 verbose=True,
                 early_stopping=False):
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.alpha = alpha_value  # L2 regularization
        
        # Will be initialized in train()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.input_cols = None
        self.output_cols = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_history = []
        
        # Set random seed for PyTorch
        torch.manual_seed(random_state)
        
    def train(self, df, y, input_cols=None, output_cols=None):
        if input_cols is None:
            input_cols = list(df.columns)
        self.input_cols = list(input_cols)
        self.input_size = len(self.input_cols)
        
        # Store output_cols if provided
        self.output_cols = list(output_cols) if output_cols is not None else None
        
        # Convert data to PyTorch tensors
        X = df[self.input_cols].values
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Initialize model
        self.model = PyTorchNN(
            input_size=self.input_size,
            hidden_layer_sizes=self.hidden_layer_sizes,
            output_size=y.shape[1]
        ).to(self.device)
        
        # Initialize weights for better convergence
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)
        
        # Initialize optimizer 
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate_init,
            momentum=0.9,  
            weight_decay=self.alpha
        )
        
        # Use MSE loss
        self.criterion = nn.MSELoss()
        
        # Train the model
        self.model.train()
        prev_loss = float('inf')
        
        for epoch in range(self.max_iter):
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            
            # Record loss
            current_loss = loss.item()
            self.loss_history.append(current_loss)
            
            # Print progress
            if self.verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{self.max_iter}], Loss: {current_loss:.6f}')
            
            # Check convergence
            if abs(prev_loss - current_loss) < self.tol:
                if self.verbose:
                    print(f'Converged at epoch {epoch+1} with loss {current_loss:.6f}')
                break
            
            prev_loss = current_loss
    
    def predict(self, df):
        if self.input_cols is None or self.model is None:
            raise RuntimeError("NNModel not trained yet: input_cols is None or model is None.")
        
        self.model.eval()
        X = df[self.input_cols].values
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        
        return preds
    
    def evaluate(self, df, y_true):
        if self.model is None:
            raise RuntimeError("NNModel not trained yet: model is None.")
            
        self.model.eval()
        X = df[self.input_cols].values
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        
        mse_value = mean_squared_error(y_true, preds)
        return mse_value
    
    def get_loss(self):
        return self.loss_history[-1] if self.loss_history else float('inf')
    
    def save(self, path="nn_model.pt"):
        if self.model is not None:
            # Save both model state and metadata
            state_dict = self.model.state_dict()
            metadata = {
                'input_cols': self.input_cols,
                'output_cols': self.output_cols,
                'input_size': self.input_size,
                'hidden_layer_sizes': self.hidden_layer_sizes,
                'loss_history': self.loss_history
            }
            torch.save({'state_dict': state_dict, 'metadata': metadata}, path)
    
    def load(self, path="nn_model.pt"):
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load metadata
        metadata = checkpoint['metadata']
        self.input_cols = metadata['input_cols']
        self.output_cols = metadata['output_cols']
        self.input_size = metadata['input_size']
        self.hidden_layer_sizes = metadata['hidden_layer_sizes']
        self.loss_history = metadata.get('loss_history', [])
        
        # Initialize and load model
        output_size = len(self.output_cols) if self.output_cols else 3
        self.model = PyTorchNN(
            input_size=self.input_size,
            hidden_layer_sizes=self.hidden_layer_sizes,
            output_size=output_size
        ).to(self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
###############################################
#  Processor Class
###############################################

class Processor:
    """
    Handles processing of raw sensor data.
    Can process a single frame or an entire CSV file.
    Maintains realtime state for acceleration calculation.
    """
    def __init__(self):
        self.reset_realtime_state()

    def reset_realtime_state(self):
        self.last_time = None
        self.last_vx = None
        self.last_vy = None

    def process_frame(self, sensor_data, track_data):
        """
        Process a single sensor reading and compute features.
        """
        t = sensor_data["time"]
        car_x = sensor_data["x_pos"]
        car_z = sensor_data["z_pos"]
        yaw_deg = sensor_data["yaw_deg"]
        long_vel = sensor_data["long_vel"]
        lat_vel = sensor_data["lat_vel"]
        yaw_rate = sensor_data["yaw_rate"]
        steering = sensor_data["steering"]
        throttle = sensor_data["throttle"]
        brake = sensor_data["brake"]

        # Shift car position to a lookahead point
        x_shifted, z_shifted = shift_position_single(car_x, car_z, yaw_deg, shift_distance=2.5)

        # Compute local acceleration (using previous frame if available)
        if self.last_time is None or self.last_vx is None or self.last_vy is None:
            ax, ay = 0.0, 0.0
        else:
            time_arr = np.array([self.last_time, t])
            vx_arr = np.array([self.last_vx, long_vel])
            vy_arr = np.array([self.last_vy, lat_vel])
            ax_arr, ay_arr = compute_acceleration(time_arr, vx_arr, vy_arr)
            ax, ay = ax_arr[0], ay_arr[0]

        self.last_time = t
        self.last_vx = long_vel
        self.last_vy = lat_vel

        yaw_rad = math.radians(yaw_deg)
        yrd, brd = raycast_for_state(
            x_shifted, z_shifted, yaw_rad,
            track_data["ordered_blue"], track_data["ordered_yellow"],
            max_distance=20.0
        )
        dc = compute_signed_distance_to_centerline(
            x_shifted, z_shifted, track_data["r_clx"], track_data["r_clz"]
        )
        dh = compute_heading_difference(
            x_shifted, z_shifted, yaw_rad,
            track_data["r_clx"], track_data["r_clz"]
        )

        # Find the projection of the car on the centerline
        dists = [math.hypot(px - car_x, pz - car_z) for (px, pz) in track_data["centerline_pts"]]
        i_proj = int(np.argmin(dists))
        if i_proj < len(track_data["r_clx"]):
            c0 = track_data["curvatures_all"][i_proj]
            tw0 = track_data["track_widths_all"][i_proj]["width"]
        else:
            c0 = 0.0
            tw0 = 4.0

        # Compute local forward points (and interpolation)
        front_local, behind_local, _, _ = get_local_centerline_points_by_distance(
            x_shifted, z_shifted, yaw_rad, track_data["centerline_pts"],
            front_distance=20.0, behind_distance=5.0
        )
        if front_local:
            fl = np.array(front_local)
            x_front = fl[:, 0]
            z_front = fl[:, 1]
            target_x = np.arange(1, 21)
            target_z = np.interp(target_x, x_front, z_front, left=z_front[0], right=z_front[-1])
        else:
            target_x = np.arange(1, 21)
            target_z = np.zeros(20)

        if behind_local:
            bl = np.array(behind_local)
            x_behind = bl[:, 0]
            z_behind = bl[:, 1]
            target_x_b = np.arange(-5, 0)
            target_z_b = np.interp(target_x_b, x_behind, z_behind, left=z_behind[0], right=z_behind[-1])
        else:
            target_x_b = np.arange(-5, 0)
            target_z_b = np.zeros(5)

        # Build the output feature dictionary
        row_dict = {
            "time": t,
            "x_pos": x_shifted,
            "z_pos": z_shifted,
            "yaw_deg": yaw_deg,
            "long_vel": long_vel,
            "lat_vel": lat_vel,
            "yaw_rate": yaw_rate,
            "steering": steering,
            "throttle": throttle,
            "brake": brake,
            "ax": ax,
            "ay": ay,
            "dist_center": -dc,
            "head_diff": dh,
            "track_width": tw0,
            "curvature": c0,
        }
        for idx, dist_val in enumerate(yrd, start=1):
            row_dict[f"yr{idx}"] = dist_val
        for idx, dist_val in enumerate(brd, start=1):
            row_dict[f"br{idx}"] = dist_val
        for j, d in enumerate(target_x, start=1):
            row_dict[f"rel_z{j}"] = target_z[j - 1]
            # Wrap-around index calculation
            idx_front = (i_proj + int(round(d))) % len(track_data["r_clx"])
            row_dict[f"c{j}"] = track_data["curvatures_all"][idx_front]
            row_dict[f"tw{j}"] = track_data["track_widths_all"][idx_front]["width"]
        for j, d in enumerate(target_x_b, start=1):
            row_dict[f"b_rel_z{j}"] = target_z_b[j - 1]
            idx_behind = i_proj + int(round(d))
            if 0 <= idx_behind < len(track_data["r_clx"]):
                row_dict[f"b_c{j}"] = track_data["curvatures_all"][idx_behind]
                row_dict[f"b_tw{j}"] = track_data["track_widths_all"][idx_behind]["width"]
            else:
                row_dict[f"b_c{j}"] = 0.0
                row_dict[f"b_tw{j}"] = 4.0

        row_dict["c0"] = c0
        row_dict["tw0"] = tw0

        return row_dict

    def process_csv(self, data_dict, track_data):
        """
        Process an entire CSV (frame-by-frame) using process_frame.
        """
        self.reset_realtime_state()
        frames = []
        t_arr = data_dict["time"]
        x_arr = data_dict["x_pos"]
        z_arr = data_dict["z_pos"]
        yaw_arr = data_dict["yaw_angle"]
        vx_arr = data_dict["long_vel"]
        vy_arr = data_dict["lat_vel"]
        yr_arr = data_dict["yaw_rate"]
        st_arr = data_dict["steering"]
        th_arr = data_dict["throttle"]
        br_arr = data_dict["brake"]

        for i in range(len(t_arr)):
            sensor_data = {
                "time": t_arr[i],
                "x_pos": x_arr[i],
                "z_pos": z_arr[i],
                "yaw_deg": yaw_arr[i],
                "long_vel": vx_arr[i],
                "lat_vel": vy_arr[i],
                "yaw_rate": yr_arr[i],
                "steering": st_arr[i],
                "throttle": th_arr[i],
                "brake": br_arr[i],
            }
            frame = self.process_frame(sensor_data, track_data)
            frames.append(frame)
        return pd.DataFrame(frames)

    def build_track_data(self, json_path):
        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)
        clx_rev = clx[::-1]
        clz_rev = clz[::-1]
        r_clx, r_clz = resample_centerline(clx_rev, clz_rev, resolution=1.0)
        centerline_pts = list(zip(r_clx, r_clz))
        ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx_rev, clz_rev)
        curvatures_all = compute_local_curvature(r_clx, r_clz, window_size=5)
        track_widths_all = compute_local_track_widths(r_clx, r_clz, ordered_blue, ordered_yellow, max_width=10.0)
        return {
            "r_clx": r_clx,
            "r_clz": r_clz,
            "centerline_pts": centerline_pts,
            "ordered_blue": ordered_blue,
            "ordered_yellow": ordered_yellow,
            "curvatures_all": curvatures_all,
            "track_widths_all": track_widths_all
            }


###############################################
# 2. NNTrainer Class
###############################################
class NNTrainer:
    """
    Handles the training workflow:
      - Loads raw CSV/JSON data.
      - Uses the Processor to extract features.
      - Applies a FeatureTransformer to scale/PCA the features.
      - Trains the NN model.
    """
    def __init__(self, processor, transformer, nn_model):
        self.processor = processor
        self.transformer = transformer
        self.nn_model = nn_model

    def train(self, csv_path, json_path, output_csv_path=None, pca_variance=0.99, test_split=0.2):
        print("[NNTrainer] Training mode...")
        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return

        track_data = self.processor.build_track_data(json_path)
        df_features = self.processor.process_csv(data_dict, track_data)
        df_features = df_features.drop(columns=["time","x_pos", "z_pos", "yaw_deg"])

        # Fit transformer (scaler + PCA) on features
        df_trans = self.transformer.fit_transform(
            df_features,
            exclude_cols=["steering", "throttle", "brake"],
            pca_variance=pca_variance
        )
        if output_csv_path:
            df_trans.to_csv(output_csv_path, index=False)
            print(f"[NNTrainer] Processed CSV saved to {output_csv_path}")

        self.transformer.save("transformer.joblib")
        print("[NNTrainer] Transformer saved.")

        # Prepare training data and split into train/test sets
        pc_cols = [c for c in df_trans.columns if c.startswith("PC")]
        out_cols = ["steering", "throttle", "brake"]
        X = df_trans[pc_cols].values
        y = df_trans[out_cols].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

        # Train NN model
        self.nn_model.train(pd.DataFrame(X_train, columns=pc_cols), y_train,
                    input_cols=pc_cols, output_cols=out_cols)
        mse_value = self.nn_model.evaluate(pd.DataFrame(X_test, columns=pc_cols), y_test)
        print("[NNTrainer] Test MSE:", mse_value)
        print("[NNTrainer] Final Loss:", self.nn_model.get_loss())
        
        # Save the PyTorch model
        self.nn_model.save("nn_model.pt")
        print("[NNTrainer] NN model saved.")
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
        self.transformer.load("transformer.joblib")
        # Update to load PyTorch model
        self.nn_model.load("nn_model.pt")
        
        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return

        track_data = self.processor.build_track_data(json_path)
        df_features = self.processor.process_csv(data_dict, track_data)
        df_trans = self.transformer.transform(df_features)

        predictions = self.nn_model.predict(df_trans)
        times = df_features["time"].values
        
        # Collect prediction results in a list
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
        self.transformer.load("transformer.joblib")
        # Update to load PyTorch model
        self.nn_model.load("nn_model.pt")
        
        track_data = self.processor.build_track_data(json_path)

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(5)
        print(f"[Realtime] Server listening on {host}:{port}...")

        client_socket, addr = server_socket.accept()
        print(f"[Realtime] Connection from {addr}")

        try:
            while True:
                raw_data = client_socket.recv(4096).decode('utf-8').strip()
                if not raw_data:
                    time.sleep(0.01)
                    continue
                fields = raw_data.split(',')
                print(fields)
                if len(fields) < 6:
                    print("Incomplete data:", raw_data)
                    continue
                sensor_data = {
                    "time": time.time(),
                    "x_pos": float(fields[0]),
                    "z_pos": float(fields[1]),
                    "yaw_deg": float(fields[2]),
                    "long_vel": float(fields[3]),
                    "lat_vel": float(fields[4]),
                    "yaw_rate": float(fields[5]),
                    "steering": None,
                    "throttle": None,
                    "brake": None
                }
                frame = self.processor.process_frame(sensor_data, track_data)
                df_single = pd.DataFrame([frame])
                df_features = df_single.drop(columns=["time","x_pos", "z_pos", "yaw_deg"])

                df_trans = self.transformer.transform(df_single)
                prediction = self.nn_model.predict(df_trans)[0]
                st_pred, th_pred, br_pred = prediction
                message = f"{st_pred},{th_pred},{br_pred}\n"
                print(f"[Realtime] Sending: {message.strip()}")
                client_socket.sendall(message.encode())
        except Exception as e:
            print(f"[Realtime] Error: {e}")
        finally:
            client_socket.close()
            server_socket.close()
            print("[NNDriver] Server closed.")
###############################################
# 3. Visualizer Class
###############################################

class Visualizer:
    """
    Visualizes a run frame-by-frame.
    
    Provides two visualization modes:
      - Relative: the local (vehicle-centered) coordinate system.
      - Absolute: the global coordinate system with scene context.
    """
    def __init__(self, processor, transformer, nn_model):
        """
        processor: instance of a Processor class that provides process_frame().
        transformer: feature transformer (e.g., scaler/PCA) used in the pipeline.
        nn_model: trained NN model (used here only if you wish to overlay predictions).
        """
        self.processor = processor
        self.transformer = transformer
        self.nn_model = nn_model

    def visualize_relative(self, csv_path, json_path, heading_length=3.0):
        """
        Visualize frame-by-frame in the local (relative) coordinate system.
        The car is fixed at (0,0) and local features (cast rays, centerline points, etc.)
        are plotted relative to the vehicle.
        """
        print("[Visualizer] Starting relative visualization...")
        # Reset realtime state so each run starts fresh.
        self.processor.reset_realtime_state()
        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return

        track_data = self.processor.build_track_data(json_path)

        # --- Set up Matplotlib figure and axes ---
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 10), 
                                                  gridspec_kw={'height_ratios': [3, 1]})
        ax_top.set_xlim(-10, 30)
        ax_top.set_ylim(-10, 10)
        ax_top.set_aspect('equal', adjustable='box')
        ax_top.set_title("Local Frame - Relative Visualization")
        ax_top.set_xlabel("Local X (m)")
        ax_top.set_ylabel("Local Z (m)")

        # A small subplot for displaying heading difference
        ax_heading = fig.add_axes([0.75, 0.1, 0.1, 0.19])
        ax_heading.set_title("Heading Diff (rad)")
        ax_heading.set_ylim(-1, 1)
        ax_heading.set_xticks([])
        heading_bar = ax_heading.bar(0, 0, width=0.5, color='purple')

        # Plot elements for the car and its heading.
        car_point, = ax_top.plot([], [], 'ko', ms=8, label='Car')
        heading_line, = ax_top.plot([], [], 'r-', lw=2, label='Heading')

        # Scatter plots for local centerline points.
        front_scatter = ax_top.scatter([], [], c='magenta', s=25, label='Front Centerline')
        behind_scatter = ax_top.scatter([], [], c='green', s=25, label='Behind Centerline')
        centerline_line, = ax_top.plot([], [], 'k-', lw=1, label='Centerline')
        centerline_bline, = ax_top.plot([], [], 'k-', lw=1)

        # Set up cast rays (yellow and blue)
        yellow_angles_deg = np.arange(-20, 111, 10)
        blue_angles_deg = np.arange(20, -111, -10)
        yellow_angles = np.deg2rad(yellow_angles_deg)
        blue_angles = np.deg2rad(blue_angles_deg)
        yellow_ray_lines = [ax_top.plot([], [], color='yellow', linestyle='--', lw=1)[0]
                            for _ in yellow_angles]
        blue_ray_lines = [ax_top.plot([], [], color='cyan', linestyle='--', lw=1)[0]
                          for _ in blue_angles]

        # Bottom panel for track width and curvature metrics.
        ax_bottom.set_title("Track Width and Centerline Curvature")
        ax_bottom.set_xlabel("Local X (m)")
        ax_bottom.set_ylabel("Track Width (m)")
        ax_bottom.set_xlim(-5, 20)
        ax_bottom.set_ylim(0, 10)
        track_width_line, = ax_bottom.plot([], [], 'bo-', label='Fwd Track Width')
        track_width_line_back, = ax_bottom.plot([], [], 'bo-')
        ax_bottom.legend(loc='upper left')

        ax_curv = ax_bottom.twinx()
        ax_curv.set_ylim(-0.5, 0.5)
        curvature_line, = ax_curv.plot([], [], 'r.-', label='Fwd Curvature')
        curvature_line_back, = ax_curv.plot([], [], 'r.-')
        ax_curv.legend(loc='upper right')

        # --- Process and visualize each frame ---
        t_arr = data_dict["time"]
        x_arr = data_dict["x_pos"]
        z_arr = data_dict["z_pos"]
        yaw_arr = data_dict["yaw_angle"]
        vx_arr = data_dict["long_vel"]
        vy_arr = data_dict["lat_vel"]
        yr_arr = data_dict["yaw_rate"]
        st_arr = data_dict["steering"]
        th_arr = data_dict["throttle"]
        br_arr = data_dict["brake"]

        for i in range(len(t_arr)):
            sensor_data = {
                "time": t_arr[i],
                "x_pos": x_arr[i],
                "z_pos": z_arr[i],
                "yaw_deg": yaw_arr[i],
                "long_vel": vx_arr[i],
                "lat_vel": vy_arr[i],
                "yaw_rate": yr_arr[i],
                "steering": st_arr[i],
                "throttle": th_arr[i],
                "brake": br_arr[i],
            }
            frame = self.processor.process_frame(sensor_data, track_data)

            # In the relative frame the car is always at (0,0).
            car_point.set_data([0], [0])
            # The heading line is drawn along the x-axis.
            heading_line.set_data([0, heading_length], [0, 0])

            # Update forward local centerline points.
            front_pts = []
            for j in range(1, 21):
                x_val = float(j)
                z_val = frame.get(f"rel_z{j}", 0.0)
                front_pts.append([x_val, z_val])
            front_scatter.set_offsets(np.array(front_pts))

            # Update behind local centerline points.
            behind_pts = []
            for j, x_val in enumerate(range(-5, 0), start=1):
                z_val = frame.get(f"b_rel_z{j}", 0.0)
                behind_pts.append([float(x_val), z_val])
            behind_scatter.set_offsets(np.array(behind_pts))

            # Update drawn centerline segments.
            cl_x_fwd = np.arange(1, 21)
            cl_z_fwd = [frame.get(f"rel_z{k}", 0.0) for k in range(1, 21)]
            centerline_line.set_data(cl_x_fwd, cl_z_fwd)

            cl_x_b = np.arange(-5, 0)
            cl_z_b = [frame.get(f"b_rel_z{k}", 0.0) for k in range(1, 6)]
            centerline_bline.set_data(cl_x_b, cl_z_b)

            # Update cast rays.
            for idx, angle in enumerate(yellow_angles):
                dist_val = frame.get(f"yr{idx+1}", 0)
                end_x = dist_val * math.cos(angle)
                end_z = dist_val * math.sin(angle)
                yellow_ray_lines[idx].set_data([0, end_x], [0, end_z])
            for idx, angle in enumerate(blue_angles):
                dist_val = frame.get(f"br{idx+1}", 0)
                end_x = dist_val * math.cos(angle)
                end_z = dist_val * math.sin(angle)
                blue_ray_lines[idx].set_data([0, end_x], [0, end_z])

            # Update bottom panel metrics.
            tws = [frame.get(f"tw{j}", 0.0) for j in range(21)]
            curvs = [frame.get(f"c{j}", 0.0) for j in range(21)]
            track_width_line.set_data(np.arange(21), tws)
            curvature_line.set_data(np.arange(21), curvs)

            btws = [frame.get(f"b_tw{j}", 0.0) for j in range(1, 6)]
            bcurvs = [frame.get(f"b_c{j}", 0.0) for j in range(1, 6)]
            track_width_line_back.set_data(np.array(list(range(-5, 0))), btws)
            curvature_line_back.set_data(np.array(list(range(-5, 0))), bcurvs)

            # Update the heading difference bar.
            dh = frame.get("head_diff", 0)
            if dh >= 0:
                heading_bar[0].set_y(0)
                heading_bar[0].set_height(dh)
            else:
                heading_bar[0].set_y(dh)
                heading_bar[0].set_height(-dh)

            plt.draw()
            plt.pause(0.001)
        print("[Visualizer] Finished relative visualization.")

    def visualize_absolute(self, csv_path, json_path, heading_length=3.0):
        """
        Visualize frame-by-frame in absolute (global) coordinates.
        The scene (cones, track edges, centerline) is drawn from the JSON file,
        and the car (and its associated cast rays and local centerline points) is animated.
        """
        print("[Visualizer] Starting absolute visualization...")

        # Build track data from the JSON.
        track_data = self.processor.build_track_data(json_path)
        # Also, get the raw cone and centerline information.
        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)
        clx_rev = clx[::-1]
        clz_rev = clz[::-1]
        r_clx, r_clz = resample_centerline(clx_rev, clz_rev, resolution=1.0)
        # For the absolute view, also resample the centerline in the original order.
        clx_abs, clz_abs = resample_centerline(clx, clz, resolution=1.0)
        centerline_pts_fw = list(zip(clx_abs, clz_abs))

        # --- Set up Matplotlib figure and axes ---
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10), 
                                                  gridspec_kw={'height_ratios': [3, 1]})
        fig.subplots_adjust(right=0.7)

        # Top panel: plot cones, track edges, and centerline.
        if blue_cones:
            bx, bz = zip(*blue_cones)
            ax_top.scatter(bx, bz, c='blue', marker='o', label="Blue Cones")
        if yellow_cones:
            yx, yz = zip(*yellow_cones)
            ax_top.scatter(yx, yz, c='gold', marker='o', label="Yellow Cones")
        ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx_rev, clz_rev)
        if ordered_blue:
            blue_x, blue_z = zip(*ordered_blue)
            ax_top.plot(blue_x, blue_z, 'b-', label="Blue Edge")
        if ordered_yellow:
            yellow_x, yellow_z = zip(*ordered_yellow)
            ax_top.plot(yellow_x, yellow_z, 'y-', label="Yellow Edge")
        if clx_abs and clz_abs:
            ax_top.plot(clx_abs, clz_abs, 'm--', label="Centerline")
        ax_top.set_xlabel("X (m)")
        ax_top.set_ylabel("Z (m)")
        ax_top.set_title("Absolute Scene with Cast Rays & Local Points")
        ax_top.legend()

        # Persistent markers for the car and its heading.
        car_marker, = ax_top.plot([], [], 'ro', markersize=10, label="Car")
        heading_line, = ax_top.plot([], [], 'r-', lw=2, label="Heading")

        # Cast ray lines.
        yellow_angles = np.deg2rad(np.arange(-20, 111, 10))
        blue_angles = np.deg2rad(np.arange(20, -111, -10))
        yellow_ray_lines = [ax_top.plot([], [], color='yellow', linestyle='--', lw=1)[0]
                            for _ in yellow_angles]
        blue_ray_lines = [ax_top.plot([], [], color='cyan', linestyle='--', lw=1)[0]
                          for _ in blue_angles]

        # Scatter objects for local centerline points (transformed to absolute coordinates).
        front_scatter = ax_top.scatter([], [], c='magenta', marker='o', s=50, label="Forward Local Points")
        behind_scatter = ax_top.scatter([], [], c='green', marker='o', s=50, label="Behind Local Points")
        ax_top.legend()

        # Bottom panel: local metrics (track width and curvature)
        ax_bottom.set_title("Local Metrics: Track Width & Curvature")
        ax_bottom.set_xlabel("Local X (m)")
        ax_bottom.set_ylabel("Track Width (m)")
        ax_bottom.set_xlim(-5, 21)
        ax_bottom.set_ylim(0, 10)
        f_track_line, = ax_bottom.plot([], [], 'bo-', label="Fwd Track Width")
        b_track_line, = ax_bottom.plot([], [], 'bo-')
        ax_bottom.legend(loc='upper left')
        ax_curv_bottom = ax_bottom.twinx()
        ax_curv_bottom.set_ylim(-0.5, 0.5)
        f_curv_line, = ax_curv_bottom.plot([], [], 'r.-', label="Curvature")
        b_curv_line, = ax_curv_bottom.plot([], [], 'r.-')
        ax_curv_bottom.legend(loc='upper right')

        # Reset realtime state.
        self.processor.reset_realtime_state()

        data = read_csv_data(csv_path)
        if data is None:
            print("Could not load CSV data.")
            return

        t_arr = data["time"]
        x_arr = data["x_pos"]
        z_arr = data["z_pos"]
        yaw_arr = data["yaw_angle"]
        vx_arr = data["long_vel"]
        vy_arr = data["lat_vel"]
        yr_arr = data["yaw_rate"]
        st_arr = data["steering"]
        th_arr = data["throttle"]
        br_arr = data["brake"]

        for i in range(len(t_arr)):
            sensor_data = {
                "time": t_arr[i],
                "x_pos": x_arr[i],
                "z_pos": z_arr[i],
                "yaw_deg": yaw_arr[i],
                "long_vel": vx_arr[i],
                "lat_vel": vy_arr[i],
                "yaw_rate": yr_arr[i],
                "steering": st_arr[i],
                "throttle": th_arr[i],
                "brake": br_arr[i],
            }
            frame = self.processor.process_frame(sensor_data, track_data)

            # --- Top Panel Update: Absolute View ---
            car_x = frame["x_pos"]
            car_z = frame["z_pos"]
            heading_deg = frame["yaw_deg"]
            heading_rad = math.radians(heading_deg)
            car_marker.set_data([car_x], [car_z])
            hx = car_x + heading_length * math.cos(heading_rad)
            hy = car_z + heading_length * math.sin(heading_rad)
            heading_line.set_data([car_x, hx], [car_z, hy])

            # Update cast rays.
            for idx, angle in enumerate(yellow_angles, start=1):
                ray_dist = frame.get(f"yr{idx}", 0)
                local_x = ray_dist * math.cos(angle)
                local_y = ray_dist * math.sin(angle)
                abs_x = car_x + local_x * math.cos(heading_rad) - local_y * math.sin(heading_rad)
                abs_y = car_z + local_x * math.sin(heading_rad) + local_y * math.cos(heading_rad)
                yellow_ray_lines[idx-1].set_data([car_x, abs_x], [car_z, abs_y])
            for idx, angle in enumerate(blue_angles, start=1):
                ray_dist = frame.get(f"br{idx}", 0)
                local_x = ray_dist * math.cos(angle)
                local_y = ray_dist * math.sin(angle)
                abs_x = car_x + local_x * math.cos(heading_rad) - local_y * math.sin(heading_rad)
                abs_y = car_z + local_x * math.sin(heading_rad) + local_y * math.cos(heading_rad)
                blue_ray_lines[idx-1].set_data([car_x, abs_x], [car_z, abs_y])

            # Update local centerline points (transformed to absolute).
            front_local, behind_local, global_front, global_behind = get_local_centerline_points_by_distance(
                car_x, car_z, heading_deg, centerline_pts_fw,
                front_distance=5.0, behind_distance=20.0
            )
            if global_front:
                front_scatter.set_offsets(np.array(global_front))
            else:
                front_scatter.set_offsets(np.empty((0, 2)))
            if global_behind:
                behind_scatter.set_offsets(np.array(global_behind))
            else:
                behind_scatter.set_offsets(np.empty((0, 2)))

            # --- Bottom Panel Update: Local Metrics ---
            f_tw = [frame.get(f"tw{j}", 0.0) for j in range(21)]
            f_curv = [frame.get(f"c{j}", 0.0) for j in range(21)]
            f_track_line.set_data(np.arange(21), f_tw)
            f_curv_line.set_data(np.arange(21), f_curv)
            b_tw = [frame.get(f"b_tw{j}", 0.0) for j in range(1, 6)]
            b_curv = [frame.get(f"b_c{j}", 0.0) for j in range(1, 6)]
            b_track_line.set_data(np.array(list(range(-5, 0))), b_tw)
            b_curv_line.set_data(np.array(list(range(-5, 0))), b_curv)

            plt.draw()
            plt.pause(0.001)
        print("[Visualizer] Finished absolute visualization.")
        plt.show()

    def visualizer_inferred(self, actual_csv_path, inferred_csv_path):
        # Read the actual and inferred CSV files
        df_actual = pd.read_csv(actual_csv_path)
        df_inferred = pd.read_csv(inferred_csv_path)

        # Create subplots: one each for steering, throttle, and brake.
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot steering values
        axs[0].plot(df_actual['time'], df_actual['steering'], label="Actual Steering", color='blue')
        axs[0].plot(df_inferred['time'], df_inferred['steering'], label="Inferred Steering", color='red', linestyle='--')
        axs[0].set_ylabel("Steering")
        axs[0].legend()
        axs[0].grid(True)

        # Plot throttle values
        axs[1].plot(df_actual['time'], df_actual['throttle'], label="Actual Throttle", color='blue')
        axs[1].plot(df_inferred['time'], df_inferred['throttle'], label="Inferred Throttle", color='red', linestyle='--')
        axs[1].set_ylabel("Throttle")
        axs[1].legend()
        axs[1].grid(True)

        # Plot brake values
        axs[2].plot(df_actual['time'], df_actual['brake'], label="Actual Brake", color='blue')
        axs[2].plot(df_inferred['time'], df_inferred['brake'], label="Inferred Brake", color='red', linestyle='--')
        axs[2].set_ylabel("Brake")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

