import csv
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

########################
# Hyperparameters
########################
BATCH_SIZE = 16
EPOCHS = 300
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8
NUM_BLUE = 10
NUM_YELLOW = 10

# 54 inputs total: 40 cones (20 blue * 2 + 20 yellow * 2) + 3 velocities + 10 curvature + 1 distance
INPUT_SIZE = (NUM_BLUE + NUM_YELLOW)*2 + 3 + 10 + 1

########################
# Dataset
########################

class CarDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

########################
# Model Definition
########################

class SimpleNN(nn.Module):
    def __init__(self, input_size=INPUT_SIZE, output_size=3, hidden_dim=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)
        # steering: tanh -> [-1,1]
        steering = torch.tanh(out[:, 0:1])
        # throttle: sigmoid -> [0,1]
        throttle = torch.sigmoid(out[:, 1:2])
        # brake: sigmoid -> [0,1]
        brake = torch.sigmoid(out[:, 2:3])

        return torch.cat([steering, throttle, brake], dim=1)


########################
# Training Code
########################

def train_model(dataset, epochs=EPOCHS, lr=LEARNING_RATE):
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, targets in train_loader:
            if torch.isnan(inputs).any() or torch.isnan(targets).any():
                # Skip batches with NaNs if any
                continue

            optimizer.zero_grad()
            outputs = model(inputs)
            if torch.isnan(outputs).any():
                continue
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                continue

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * inputs.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                if torch.isnan(inputs).any() or torch.isnan(targets).any():
                    continue
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                if torch.isnan(loss):
                    continue
                total_val_loss += loss.item() * inputs.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    return model

########################
# Main Execution
########################

if __name__ == "__main__":

    # List of CSV file paths
    csv_file_paths = ["car_data_run1_relative.csv",
                      "car_data_run2_relative.csv",
                      "car_data_run3_relative.csv"]

    # Read and concatenate the CSVs
    df = pd.concat([pd.read_csv(file) for file in csv_file_paths], ignore_index=True)

    # Check for NaNs
    if df.isna().any().any():
        df = df.dropna()

    # Columns we need:
    # The CSV has columns for car state and also the 54 input features.
    # We know the order from humandriver:
    # "Time", "X Position", "Z Position", "Yaw", "Longitudinal Velocity", "Sliding Velocity",
    # "Yaw Rate", "Steering Angle", "Throttle", "Brake",
    # Bx1,Bz1,...,Bx10,Bz10,Yx1,Yz1,...,Yx10,Yz10,C1,...,C10,DistCenterline
    # We must extract these columns from df

    # Identify input columns for cones
    cone_cols = []
    for i in range(NUM_BLUE):
        cone_cols.append(f"Bx{i+1}")
        cone_cols.append(f"Bz{i+1}")
    for i in range(NUM_YELLOW):
        cone_cols.append(f"Yx{i+1}")
        cone_cols.append(f"Yz{i+1}")

    # Velocity, slide, yaw_rate are already in df:
    vel_cols = ["Longitudinal Velocity", "Sliding Velocity", "Yaw Rate"]

    # Curvature columns
    curv_cols = [f"C{i+1}" for i in range(10)]

    # Distance centerline
    dist_col = ["DistCenterline"]

    # Inputs: cones (40), vel (3), curv (10), dist (1) = 54 total
    input_cols = cone_cols + vel_cols + curv_cols + dist_col

    # Targets: Steering Angle, Throttle, Brake
    target_cols = ["Steering Angle", "Throttle", "Brake"]

    # Extract arrays
    all_inputs = df[input_cols].to_numpy(dtype=np.float32)
    all_targets = df[target_cols].to_numpy(dtype=np.float32)

    # Compute mean and std of all input features
    means = all_inputs.mean(axis=0)
    stds = all_inputs.std(axis=0)
    # Avoid division by zero
    stds[stds == 0] = 1.0

    # Normalize inputs
    all_inputs = (all_inputs - means) / stds

    # Save means and stds for inference if needed
    np.save("input_means.npy", means)
    np.save("input_stds.npy", stds)

    # Create dataset with normalized data
    dataset = CarDataset(all_inputs, all_targets)

    model = train_model(dataset)
    torch.save(model.state_dict(), "car_controller_model.pth")
