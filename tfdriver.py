#!/usr/bin/env python3
"""
tfdriver.py
"""

import math
import copy  
import socket
import time

import numpy as np
import pandas as pd

from utils import *
from transformer import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------
# 1. Helper Function for Creating Sequence-to-Sequence Data
# ---------------------------------------------------------------------
def create_sequences(X, y, seq_length):
    """
    Converts independent samples into sequences using a sliding window.
    Each sequence has a length of `seq_length`, and the target is the entire
    sequence of control outputs corresponding to the input sequence.
    """
    sequences = []
    targets = []
    for i in range(len(X) - seq_length + 1):
        seq = X[i : i + seq_length]
        target_seq = y[i : i + seq_length]
        sequences.append(seq)
        targets.append(target_seq)
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
        pe[:, 0::2] = torch.sin(position * div_term)  # apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # apply cos to odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------
# 3. Transformer-Based Model (Sequence-to-Sequence)
# ---------------------------------------------------------------------
class TFModel(nn.Module):
    def __init__(self,
                 input_size,
                 seq_length=5,          # Number of frames (time steps) in each sequence
                 d_model=8,             # Embedding dimension for transformer
                 nhead=2,                # Number of attention heads
                 num_encoder_layers=2,   # Number of transformer encoder layers
                 dim_feedforward=32,     # Feedforward network size inside transformer encoder layers
                 dropout=0.1,
                 output_size=3,          # Steering, throttle, brake
                 learning_rate_init=0.001,
                 max_iter=10,
                 tol=1e-6,
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
        # Decoder: apply the same linear layer at every timestep
        self.decoder = nn.Linear(self.d_model, self.output_size)
    
    def forward(self, x):
        """
        Forward pass expects input of shape (batch_size, seq_length, input_size).
        Returns predictions for each timestep, i.e. shape (batch_size, seq_length, output_size).
        """
        x = self.input_proj(x)            # (batch_size, seq_length, d_model)
        x = self.pos_encoder(x)           # (batch_size, seq_length, d_model)
        # Transformer expects input as (seq_length, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)   # (seq_length, batch_size, d_model)
        x = self.decoder(x)               # (seq_length, batch_size, output_size)
        output = x.transpose(0, 1)        # (batch_size, seq_length, output_size)
        return output

    def train_model(self, df, y, df_val, y_val, input_cols=None, output_cols=None, batch_size=64, warmup_steps=100):
        """
        Trains the transformer model using sequence-to-sequence data.
        The input DataFrames are converted into sequences (using a sliding window)
        and then trained using mini-batch gradient descent with AdamW optimizer and warm-up.
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

        # Create sequences for transformer input (sequence-to-sequence)
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
        
        # Setup AdamW optimizer with warm-up scheduler
        self.optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate_init, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min((step + 1) / warmup_steps, 1.0)
        )
        
        self.criterion = nn.MSELoss()

        patience = 3
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0
        global_step = 0

        # Create DataLoaders for mini-batch training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        for epoch in range(self.max_iter):
            self.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self(X_batch)  # (batch_size, seq_length, output_size)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                scheduler.step()  # Update LR with warm-up schedule per mini-batch
                epoch_loss += loss.item() * X_batch.size(0)
                global_step += 1
            epoch_loss /= len(train_loader.dataset)

            # Evaluate on validation set
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)
            self.loss_history.append(epoch_loss)
            
            current_lr = scheduler.get_last_lr()[0]
            if self.verbose:
                print(f"Epoch [{epoch+1}/{self.max_iter}] | "
                      f"Train Loss: {epoch_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.6f}")
            
            if val_loss < best_val_loss - self.tol:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch+1} with val_loss={val_loss:.6f}")
                break
        
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

    def predict(self, df):
        """
        Predicts control outputs using the most recent sequence of frames.
        The DataFrame should be ordered chronologically and contain at least `seq_length` rows.
        Returns predictions for each timestep in the input sequence.
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
        
        return preds  # shape: (1, seq_length, output_size)

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
    
    def save(self, path="models/networks/transformer_v7.pt"):
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
    
    def load(self, path="models/networks/transformer_v7.pt"):
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
    def __init__(self, processor, transformer_pipeline, tf_model):
        self.processor = processor
        self.transformer = transformer_pipeline
        self.tf_model = tf_model

    def train(self, data, json_path=None, output_csv_path=None, pca_variance=0.99, 
              test_split=0.01, val_split=0.2, use_postprocessed=False):
        """
        Trains the TF model with a dedicated validation set and an independent test set.
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
        self.tf_model.train_model(
            df=pd.DataFrame(X_train, columns=pc_cols),
            y=y_train,
            df_val=pd.DataFrame(X_val, columns=pc_cols),
            y_val=y_val,
            input_cols=pc_cols,
            output_cols=out_cols,
            batch_size=64,
            warmup_steps=100
        )

        # Save the trained NN model
        self.tf_model.save()
        print("[TFTrainer] TF model saved.")

        # Evaluate on the test set
        mse_value = self.tf_model.evaluate(pd.DataFrame(X_test, columns=pc_cols), y_test)
        print("[TFTrainer] Test MSE:", mse_value)
        print("[TFTrainer] Final Loss:", self.tf_model.get_loss())
        

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
    def __init__(self, processor, transformer_pipeline, tf_model, output_csv=None):
        self.processor = processor
        self.transformer = transformer_pipeline
        self.tf_model = tf_model
        self.output_csv = output_csv  
        self.buffer = []  # rolling buffer for realtime sequence formation

    def inference_mode(self, csv_path, json_path):
        """
        Inference mode for a full CSV using a sliding window approach.
        For every sliding window of length `seq_length`, the method predicts a control output.
        """
        print("[TFDriver] Inference mode (sliding window)...")
        self.transformer.load()
        self.tf_model.load()  # load trained transformer model

        # Read the CSV data and process it.
        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return

        track_data = self.processor.build_track_data(json_path)
        df_features = self.processor.process_csv(data_dict, track_data)
        df_trans = self.transformer.transform(df_features)
        times = df_features["time"].values

        if len(df_trans) < self.tf_model.seq_length:
            print("Not enough data for inference")
            return

        results = []
        # Slide the window over every valid sequence in the transformed data.
        for i in range(len(df_trans) - self.tf_model.seq_length + 1):
            # Get the current window with seq_length rows.
            window = df_trans.iloc[i:i + self.tf_model.seq_length]
            # Reshape the window for model input: (1, seq_length, num_features)
            X_window = window[self.tf_model.input_cols].values.reshape(1, self.tf_model.seq_length, -1)
            X_window_tensor = torch.FloatTensor(X_window).to(self.tf_model.device)
            
            # Run the model and take the prediction from the last time step.
            with torch.no_grad():
                # The model returns a full sequence of outputs; select the last one.
                pred_sequence = self.tf_model(X_window_tensor).cpu().numpy()
                pred = pred_sequence[0, -1, :]  # (output_size,) for steering, throttle, brake

            # Use the timestamp corresponding to the last row of the window.
            results.append({
                "time": times[i + self.tf_model.seq_length - 1],
                "steering": pred[0],
                "throttle": pred[1],
                "brake": pred[2]
            })

        print("[TFDriver] Inference complete.")
        
        # Optionally save the results to CSV if an output path is provided.
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
        self.tf_model.load()
        
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
                # t0 = time.time()
                frame = self.processor.process_frame(sensor_data, track_data)
                # t1 = time.time()
                # print(f"Processed frame in {t1 - t0:.4f} seconds")
                all_frames.append(frame)

                # Preprocess the frame (drop columns not used for prediction)
                df_single = pd.DataFrame([frame])
                df_features = df_single.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
                df_trans = self.transformer.transform(df_features)
                # t2 = time.time()
                # print(f"Transformed frame in {t2 - t1:.4f} seconds")
                
                # Append the transformed row (as a dict) to the rolling buffer
                self.buffer.append(df_trans.iloc[0].to_dict())
                if len(self.buffer) > self.tf_model.seq_length:
                    self.buffer.pop(0)  # keep the buffer size equal to seq_length
                
                # Only run prediction when we have enough frames
                if len(self.buffer) == self.tf_model.seq_length:
                    df_seq = pd.DataFrame(self.buffer)
                    X_seq = df_seq[self.tf_model.input_cols].values.reshape(1, self.tf_model.seq_length, -1)
                    X_seq_tensor = torch.FloatTensor(X_seq).to(self.tf_model.device)
                    t3 = time.time() 
                    with torch.no_grad():
                        prediction_seq = self.tf_model(X_seq_tensor).cpu().numpy()[0]  # (seq_length, output_size)
                    # t4 = time.time()
                    # print(f"Predicted in {t4 - t3:.4f} seconds")
                    # Use the last timestep's prediction for realtime control
                    st_pred, th_pred, br_pred = prediction_seq[-1]
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
