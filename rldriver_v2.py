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
            "steering": None,
            "throttle": None,
            "brake": None
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
    def __init__(self, model_path=None, json_path=None, output_csv=None):
        self.output_csv = output_csv
        
        # Initialize processor and transformer instances.
        self.processor = Processor()
        self.transformer = FeatureTransformer()
        self.transformer.load()
        
        # Instantiate NNModel without requiring input_size. The NNModel will be properly initialized
        # when we load the saved model.
        self.nn_model = NNModel()
        self.nn_model.load(model_path)  # Load model state and metadata
        self.nn_model.eval()  # Set to evaluation mode
        
        # Build track data if a JSON path is provided.
        if json_path:
            self.track_data = self.processor.build_track_data(json_path)
        else:
            self.track_data = None

    def act(self, state):
        """
        Given a sensor_data dictionary, process it and return a tuple of (steering, throttle, brake).
        """
        # Process the sensor data to generate a feature frame.
        frame = self.processor.process_frame(state, self.track_data)
        
        # Create a DataFrame from the frame.
        df_single = pd.DataFrame([frame])
        # Drop columns that are not used as input features.
        df_features = df_single.drop(columns=["time", "x_pos", "z_pos", "yaw_angle"])
        
        # Apply the transformer to process features.
        df_trans = self.transformer.transform(df_features)
        
        # Use the NNModel's predict method to obtain outputs.
        # The NNModel expects a DataFrame with the same input columns used during training.
        prediction = self.nn_model.predict(df_trans)[0]
        st_pred, th_pred, br_pred = prediction
        
        # Optionally apply a threshold to brake.
        if br_pred < 0.05:
            br_pred = 0.0
        
        return st_pred, th_pred, br_pred

if __name__ == "__main__":

    unity = UnityEnv(host='127.0.0.1', port = 65432)
    actor = Actor(model_path="models/networks/nn_model.pt", json_path="sim/tracks/track17.json")

    try:
        previous_time = time.time()
        while True:
            state = unity.receive_state()
            if state is None:
                break

            steering, throttle, brake = actor.act(state)
            unity.send_command(steering, throttle, brake)
    except Exception as e:
        print(f"[Main] Error: {e}")
    finally:
        unity.close()
