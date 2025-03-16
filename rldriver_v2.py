import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
from collections import deque
import socket
import time

from utils import *
from processor import *
from transformer import *
from nndriver import NNModel

class UnityEnv:
    """
    Handles communication with Unity.
    """
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"[UnityEnv] Server listening on {self.host}:{self.port}...")
        self.client_socket, self.addr = self.server_socket.accept()
        print(f"[UnityEnv] Connection from {self.addr}")

    def receive_state(self):
        buffer = ""
        while "\n" not in buffer:
            buffer += self.client_socket.recv(1024).decode('utf-8')
        # split by newline and take the first complete message.
        line, remainder = buffer.split('\n', 1)
        # optionally, save 'remainder' for the next call.
        raw_data = line.strip()
        fields = raw_data.split(',')
        state = {
            "time": time.time(),
            "x_pos": float(fields[0]),
            "z_pos": float(fields[1]),
            "yaw_angle": -float(fields[2]) + 90,
            "long_vel": float(fields[3]),
            "lat_vel": float(fields[4]),
            "yaw_rate": float(fields[5]),
            "steering": 0.0,
            "throttle": 0.0,
            "brake": 0.0
        }
        # print(f"[UnityEnv] received sensor data: {state}")
        return state


    def send_command(self, steering, throttle, brake):
        """
        Sends command (steering, throttle, brake) back to Unity.
        """
        message = f"{steering},{throttle},{brake}\n"
        # print(f"[UnityEnv] Sending: {message.strip()}")
        self.client_socket.sendall(message.encode())

    def close(self):
        self.client_socket.close()
        self.server_socket.close()
        print("[UnityEnv] Connection closed.")

class Actor:
    """
    Loads a pretrained NNModel and uses it to produce control outputs (steering, throttle, brake)
    based on sensor data coming from Unity. It leverages the Processor and Transformer to prepare
    the input data.
    """
    def __init__(self, processor, transformer, model_path=None, track_data=None, output_csv=None):
        self.output_csv = output_csv
        
        # Initialize processor and transformer instances.
        self.processor = processor
        self.transformer = transformer
        self.transformer.load()
        
        # Instantiate NNModel without requiring input_size. The NNModel will be properly initialized
        # when we load the saved model.
        self.nn_model = NNModel()
        self.nn_model.load(model_path)  # Load model state and metadata
        self.nn_model.eval()  # Set to evaluation mode
        
        # Build track data if provided.
        self.track_data = track_data

    def process(self, state):
        """
        Processes the raw sensor data once and returns the transformed features.
        """
        # Process the sensor data to generate a feature frame.
        frame = self.processor.process_frame(state, self.track_data)
        # Create a DataFrame from the frame.
        df_single = pd.DataFrame([frame])
        # Drop columns that are not used as input features.
        df_features = df_single.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
        # Apply the transformer to process features.
        df_trans = self.transformer.transform(df_features)
        return df_trans.astype(float)

    def act(self, state, preprocessed=None):
        """
        Given a sensor data dictionary, uses the (preprocessed) features to return a tuple of
        (steering, throttle, brake).
        """
        if preprocessed is None:
            preprocessed = self.process(state)
        # Use the NNModel's predict method to obtain outputs.
        # Replace the undefined 'df_trans' with the 'preprocessed' DataFrame.
        prediction = self.nn_model.predict(preprocessed)[0]
        st_pred, th_pred, br_pred = prediction
        
        # Optionally apply a threshold to brake.
        if br_pred < 0.05:
            br_pred = 0.0
        
        return st_pred, th_pred, br_pred

class Critic(nn.Module):
    """
    A simple feedforward network to estimate the state value.
    Expects preprocessed input features (e.g. from the transformer)
    and outputs a single scalar value.
    """
    def __init__(self, input_size, hidden_layer_sizes=(20, 20)):
        """
        Args:
            input_size (int): Number of input features.
            hidden_layer_sizes (tuple): Sizes of the hidden layers.
        """
        super(Critic, self).__init__()
        layers = []
        last_size = input_size
        for hidden in hidden_layer_sizes:
            layers.append(nn.Linear(last_size, hidden))
            layers.append(nn.ReLU())
            last_size = hidden
        # Final layer outputs a single value (state value estimate)
        layers.append(nn.Linear(last_size, 1))
        self.value_network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass that outputs the state value.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Predicted value of shape (batch_size, 1).
        """
        return self.value_network(x)

if __name__ == "__main__":

    unity = UnityEnv(host='127.0.0.1', port = 65432)

    processor = Processor()
    transformer = FeatureTransformer()

    actor = Actor(processor, transformer, model_path="models/networks/nn_model.pt")

    track_data = processor.build_track_data("sim/tracks/track17.json")

    sample_state = unity.receive_state()
    frame_sample = processor.process_frame(sample_state, track_data)
    df_sample = pd.DataFrame([frame_sample])
    df_features_sample = df_sample.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
    df_trans_sample = transformer.transform(df_features_sample)
    input_size = df_trans_sample.shape[1]

    critic = Critic(input_size=input_size, hidden_layer_sizes=(20,20))
    critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)

    try:
        previous_time = time.time()
        while True:
            state = unity.receive_state()
            if state is None:
                break

            frame = processor.process_frame(state, track_data)
            df_single = pd.DataFrame([frame])
            df_features = df_single.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
            df_trans = transformer.transform(df_features)

            features_tensor = torch.FloatTensor(df_trans.values.astype(np.float32))

            steering, throttle, brake = actor.act(state, preprocessed=df_trans)
            with torch.no_grad():
                state_value = critic(features_tensor)
            print(f"State value: {state_value.item():.4f}")

            unity.send_command(steering, throttle, brake)
    except Exception as e:
        print(f"[Main] Error: {e}")
    finally:
        unity.close()
