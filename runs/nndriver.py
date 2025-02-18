#!/usr/bin/env python3
"""
nndriver.py

Unified codebase for:
  - Reading and post-processing data
  - Applying StandardScaler + PCA
  - Training and/or running inference with an MLPRegressor-based NN.

Usage:
  python nndriver.py --mode train --csv session3/run1.csv --json ../../../sim/tracks/default.json
     (Trains the pipeline on the CSV data, fits StandardScaler & PCA, saves them,
      and trains a model.)

  python nndriver.py --mode infer --csv new_data.csv --json ../../../sim/tracks/default.json
     (Loads the pre-fitted StandardScaler & PCA, processes each timestep,
      applies the NN for real-time/inference.)
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

# ---------------------------------------------------------------------
# ------------------ UTILITY / POSTPROCESSING CODE --------------------
# ---------------------------------------------------------------------

def compute_centerline_cumulative_distance(centerline_x, centerline_z):
    cum_dist = [0.0]
    for i in range(1, len(centerline_x)):
        dx = centerline_x[i] - centerline_x[i - 1]
        dz = centerline_z[i] - centerline_z[i - 1]
        dist = np.hypot(dx, dz)
        cum_dist.append(cum_dist[-1] + dist)
    return cum_dist

def parse_cone_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    x_values = data.get("x", [])
    y_values = data.get("y", [])
    colors   = data.get("color", [])
    clx      = data.get("centerline_x", [])
    clz      = data.get("centerline_y", [])  # Typically "z" coords

    if not (len(x_values) == len(y_values) == len(colors)):
        raise ValueError("JSON file data lengths for 'x', 'y', 'color' must be equal.")

    blue_cones = [(x, z) for x, z, c in zip(x_values, y_values, colors) if c.lower() == "blue"]
    yellow_cones = [(x, z) for x, z, c in zip(x_values, y_values, colors) if c.lower() == "yellow"]

    return blue_cones, yellow_cones, clx, clz

def project_cone_onto_centerline(cone, centerline_x, centerline_z, cum_dist):
    cone_x, cone_z = cone
    min_dist = float('inf')
    best_idx = 0
    for i, (cx, cz) in enumerate(zip(centerline_x, centerline_z)):
        d = np.hypot(cone_x - cx, cone_z - cz)
        if d < min_dist:
            min_dist = d
            best_idx = i
    return cum_dist[best_idx]

def order_cones_by_centerline(cones, centerline_x, centerline_z):
    if not cones:
        return []
    cum_dist = compute_centerline_cumulative_distance(centerline_x, centerline_z)
    cones_with_projection = []
    for cone in cones:
        proj_distance = project_cone_onto_centerline(cone, centerline_x, centerline_z, cum_dist)
        cones_with_projection.append((proj_distance, cone))
    cones_with_projection.sort(key=lambda item: item[0])
    ordered_cones = [cone for _, cone in cones_with_projection]
    return ordered_cones

def create_track_edges(blue_cones, yellow_cones, centerline_x, centerline_z):
    ordered_blue = order_cones_by_centerline(blue_cones, centerline_x, centerline_z)
    ordered_yellow = order_cones_by_centerline(yellow_cones, centerline_x, centerline_z)

    if ordered_blue and (np.hypot(ordered_blue[0][0] - ordered_blue[-1][0],
                                  ordered_blue[0][1] - ordered_blue[-1][1]) > 1e-6):
        ordered_blue.append(ordered_blue[0])

    if ordered_yellow and (np.hypot(ordered_yellow[0][0] - ordered_yellow[-1][0],
                                    ordered_yellow[0][1] - ordered_yellow[-1][1]) > 1e-6):
        ordered_yellow.append(ordered_yellow[0])

    return ordered_blue, ordered_yellow

def read_csv_data(file_path):
    """
    Reads CSV data and returns a dict of numpy arrays.
    """
    if not os.path.exists(file_path):
        logging.error(f"CSV file not found: {file_path}")
        return None

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        times, x_pos, z_pos, yaw_angle = [], [], [], []
        long_vel, lat_vel, yaw_rate  = [], [], []
        steering, throttle, brake = [], [], []

        for row in reader:
            try:
                times.append(float(row["time"]))
                x_pos.append(float(row["x_pos"]))
                z_pos.append(float(row["z_pos"]))
                yaw_angle.append(float(row["yaw_angle"]))
                long_vel.append(float(row["long_vel"]))
                lat_vel.append(float(row["lat_vel"]))
                yaw_rate.append(float(row["yaw_rate"]))
                steering.append(float(row["steering"]))
                throttle.append(float(row["throttle"]))
                brake.append(float(row["brake"]))
            except Exception as e:
                logging.warning(f"Error parsing row: {row} - {e}")

    if not times:
        logging.error("No data loaded from CSV.")
        return None

    return {
        "time":      np.array(times),
        "x_pos":     np.array(x_pos),
        "z_pos":     np.array(z_pos),
        "yaw_angle": np.array(yaw_angle),
        "long_vel":  np.array(long_vel),
        "lat_vel":   np.array(lat_vel),
        "yaw_rate":  np.array(yaw_rate),
        "steering":  np.array(steering),
        "throttle":  np.array(throttle),
        "brake":     np.array(brake),
    }

def shift_car_position(data, shift_distance=1.5):
    if data is None:
        return None
    for i in range(len(data["x_pos"])):
        yaw = math.radians(data["yaw_angle"][i])
        offset_x = shift_distance * math.sin(yaw)
        offset_z = -shift_distance * math.cos(yaw)
        data["x_pos"][i] += offset_x
        data["z_pos"][i] += offset_z
    return data

def resample_centerline(centerline_x, centerline_z, resolution=1.0):
    cum_dist = compute_centerline_cumulative_distance(centerline_x, centerline_z)
    total_length = cum_dist[-1]
    new_dists = np.arange(0, total_length + resolution, resolution)
    new_x = np.interp(new_dists, cum_dist, centerline_x)
    new_z = np.interp(new_dists, cum_dist, centerline_z)
    return new_x.tolist(), new_z.tolist()

def cross2D(a, b):
    return a[0]*b[1] - a[1]*b[0]

def ray_segment_intersection(ray_origin, ray_direction, seg_start, seg_end):
    p = ray_origin
    r = ray_direction
    q = seg_start
    s = (seg_end[0]-seg_start[0], seg_end[1]-seg_start[1])
    rxs = cross2D(r, s)
    if abs(rxs) < 1e-6:
        return None
    qp = (q[0]-p[0], q[1]-p[1])
    t = cross2D(qp, s) / rxs
    u = cross2D(qp, r) / rxs
    if t >= 0 and 0 <= u <= 1:
        return t
    return None

def compute_ray_edge_intersection_distance(ray_origin, ray_direction, edge_points, max_distance=10.0):
    best_t = max_distance
    found = False
    for i in range(len(edge_points) - 1):
        seg_start = edge_points[i]
        seg_end   = edge_points[i+1]
        t_val = ray_segment_intersection(ray_origin, ray_direction, seg_start, seg_end)
        if t_val is not None and t_val < best_t:
            best_t = t_val
            found = True
    return best_t if found else None

def raycast_for_state(car_x, car_z, car_heading, blue_edge, yellow_edge, max_distance=20):
    yellow_angles_deg = np.arange(-20, 111, 10)
    blue_angles_deg   = np.arange( 20, -111, -10)

    yellow_ray_distances = []
    blue_ray_distances   = []

    for rel_angle_deg in yellow_angles_deg:
        rel_angle = math.radians(rel_angle_deg)
        ray_angle = car_heading + rel_angle
        ray_dir   = (math.cos(ray_angle), math.sin(ray_angle))
        closest_distance = max_distance
        for i in range(len(yellow_edge)-1):
            seg_start = yellow_edge[i]
            seg_end   = yellow_edge[i+1]
            t_val = ray_segment_intersection((car_x, car_z), ray_dir, seg_start, seg_end)
            if t_val is not None and t_val < closest_distance:
                closest_distance = t_val
        yellow_ray_distances.append(closest_distance)

    for rel_angle_deg in blue_angles_deg:
        rel_angle = math.radians(rel_angle_deg)
        ray_angle = car_heading + rel_angle
        ray_dir   = (math.cos(ray_angle), math.sin(ray_angle))
        closest_distance = max_distance
        for i in range(len(blue_edge)-1):
            seg_start = blue_edge[i]
            seg_end   = blue_edge[i+1]
            t_val = ray_segment_intersection((car_x, car_z), ray_dir, seg_start, seg_end)
            if t_val is not None and t_val < closest_distance:
                closest_distance = t_val
        blue_ray_distances.append(closest_distance)

    return yellow_ray_distances, blue_ray_distances

def compute_local_curvature(centerline_x, centerline_z, window_size=5):
    N = len(centerline_x)
    curvatures = [0.0] * N
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2

    for i in range(N):
        start = max(0, i - half_window)
        end   = min(N, i + half_window + 1)
        x_local = np.array(centerline_x[start:end])
        y_local = np.array(centerline_z[start:end])
        if len(x_local) < 3:
            curvatures[i] = 0.0
            continue
        A = np.column_stack((x_local, y_local, np.ones_like(x_local)))
        b_vec = -(x_local**2 + y_local**2)
        sol, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        D, E, F = sol
        center_x = -D/2
        center_y = -E/2
        R_sq = center_x**2 + center_y**2 - F
        if R_sq <= 1e-6:
            curvatures[i] = 0.0
        else:
            R = math.sqrt(R_sq)
            curvature = 1.0 / R
            if i > 0 and i < N-1:
                vec1 = np.array([centerline_x[i - 1] - center_x, centerline_z[i - 1] - center_y])
                vec2 = np.array([centerline_x[i + 1] - center_x, centerline_z[i + 1] - center_y])
                cross_val = vec1[0]*vec2[1] - vec1[1]*vec2[0]
                if cross_val < 0:
                    curvature = -curvature
            curvatures[i] = curvature
    return curvatures

def compute_local_track_widths(resampled_clx, resampled_clz, ordered_blue, ordered_yellow, max_width=10.0):
    results = []
    pts = list(zip(resampled_clx, resampled_clz))
    N = len(pts)

    for i in range(N):
        if i == 0:
            dx = pts[i+1][0] - pts[i][0]
            dz = pts[i+1][1] - pts[i][1]
        elif i == N-1:
            dx = pts[i][0] - pts[i-1][0]
            dz = pts[i][1] - pts[i-1][1]
        else:
            dx = pts[i+1][0] - pts[i-1][0]
            dz = pts[i+1][1] - pts[i-1][1]

        norm = math.hypot(dx, dz)
        if norm < 1e-9:
            T = (1.0, 0.0)
        else:
            T = (dx/norm, dz/norm)

        left_normal  = (-T[1], T[0])
        right_normal = ( T[1],-T[0])

        center = pts[i]
        d_yellow = compute_ray_edge_intersection_distance(center, left_normal,  ordered_yellow, max_distance=max_width)
        d_blue   = compute_ray_edge_intersection_distance(center, right_normal, ordered_blue,   max_distance=max_width)
        if d_yellow is None: d_yellow = max_width
        if d_blue   is None: d_blue   = max_width
        width = d_yellow + d_blue
        results.append({"center": center, "width": width})
    return results

def compute_heading_difference(car_x, car_z, car_heading, centerline_x, centerline_z):
    N = len(centerline_x)
    track_headings = np.zeros(N)
    for i in range(N):
        if i == 0:
            dx = centerline_x[1] - centerline_x[0]
            dz = centerline_z[1] - centerline_z[0]
        elif i == N - 1:
            dx = centerline_x[-1] - centerline_x[-2]
            dz = centerline_z[-1] - centerline_z[-2]
        else:
            dx = centerline_x[i+1] - centerline_x[i-1]
            dz = centerline_z[i+1] - centerline_z[i-1]
        track_headings[i] = math.atan2(dz, dx)
    track_headings_unwrapped = np.unwrap(track_headings)

    dists = np.hypot(np.array(centerline_x) - car_x, np.array(centerline_z) - car_z)
    i_min = int(np.argmin(dists))
    track_heading_closest = track_headings_unwrapped[i_min]

    car_heading_normalized = (car_heading + math.pi) % (2*math.pi) - math.pi
    heading_diff = (car_heading_normalized - track_heading_closest + math.pi) % (2*math.pi) - math.pi

    if heading_diff > math.pi/2:
        heading_diff -= math.pi
    elif heading_diff < -math.pi/2:
        heading_diff += math.pi

    return heading_diff

def compute_signed_distance_to_centerline(car_x, car_z, centerline_x, centerline_z):
    pts = list(zip(centerline_x, centerline_z))
    best_distance = float('inf')
    best_signed_distance = 0.0
    for i in range(len(pts)-1):
        a = pts[i]
        b = pts[i+1]
        vx, vz = (b[0] - a[0], b[1] - a[1])
        v_dot_v = vx*vx + vz*vz
        if v_dot_v == 0:
            proj = a
        else:
            t = ((car_x - a[0])*vx + (car_z - a[1])*vz)/v_dot_v
            if t < 0:
                proj = a
            elif t > 1:
                proj = b
            else:
                proj = (a[0] + t*vx, a[1] + t*vz)
        dist = math.hypot(car_x - proj[0], car_z - proj[1])
        if dist < best_distance:
            best_distance = dist
            norm_v = math.sqrt(v_dot_v) if v_dot_v !=0 else 1e-9
            tangent = (vx/norm_v, vz/norm_v)
            left_normal = (-tangent[1], tangent[0])
            diff_vec = (car_x - proj[0], car_z - proj[1])
            sign = 1 if (diff_vec[0]*left_normal[0] + diff_vec[1]*left_normal[1]) >= 0 else -1
            best_signed_distance = sign*dist
    return best_signed_distance

def compute_accelerations(time, vx, vy):
    ax = np.zeros_like(vx)
    ay = np.zeros_like(vy)
    if len(time) < 2:
        return ax, ay

    ax[0] = (vx[1] - vx[0]) / (time[1] - time[0])
    ay[0] = (vy[1] - vy[0]) / (time[1] - time[0])
    for i in range(1, len(time)-1):
        dt = time[i+1] - time[i-1]
        if abs(dt) < 1e-9:
            ax[i] = 0.0
            ay[i] = 0.0
        else:
            ax[i] = (vx[i+1] - vx[i-1]) / dt
            ay[i] = (vy[i+1] - vy[i-1]) / dt
    ax[-1] = (vx[-1] - vx[-2]) / (time[-1] - time[-2])
    ay[-1] = (vy[-1] - vy[-2]) / (time[-1] - time[-2])
    return ax, ay

def run_tcp_server(realtime_driver, host='127.0.0.1', port=65432):
    """
    Sets up a simple TCP server that:
      1) Waits for a single Unity client to connect.
      2) Repeatedly receives the car state as a comma-separated string:
         "time,x_pos,z_pos,yaw_deg,long_vel,lat_vel,yaw_rate,steer_in,throttle_in,brake_in"
      3) Uses `realtime_driver.process_single_step(...)` to compute predictions.
      4) Sends the predicted steering, throttle, and brake values back to Unity.
    
    The `realtime_driver` argument must be an instance of your RealtimeDriver
    (with a `process_single_step` method).
    """
    print(f"Setting up server on {host}:{port}...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print("Server listening... Waiting for connection.")
    
    client_socket, addr = server_socket.accept()
    print(f"Connection from {addr}")

    try:
        while True:
            # Receive data (blocking call). Expect comma-separated values.
            raw_data = client_socket.recv(4096).decode('utf-8').strip()
            if not raw_data:
                time.sleep(0.01)
                continue  # No new data yet

            # Example CSV format: 10 fields -> time, x, z, yaw_deg, vx, vy, yaw_rate, steer_in, throttle_in, brake_in
            fields = raw_data.split(',')
            if len(fields) < 6:
                print(raw_data)
                # Not enough values, skip
                continue
            
            # Parse incoming fields
            sim_time   = time.time()
            car_x      = float(fields[0])
            car_z      = float(fields[1])
            yaw_deg    = float(fields[2])
            vx         = float(fields[3])
            vy         = float(fields[4])
            yaw_rate   = float(fields[5])

            # Call the real-time driver for predictions
            st_pred, th_pred, br_pred = realtime_driver.process_single_step(
                time=sim_time,
                x_pos=car_x,
                z_pos=car_z,
                yaw_angle_deg=yaw_deg,
                long_vel=vx,
                lat_vel=vy,
                yaw_rate=yaw_rate,
                steering_in=0,
                throttle_in=0,
                brake_in=0)

            # Send predictions back
            message = f"{st_pred},{th_pred},{0}\n"
            print(f"Sending: {message.strip()}")
            client_socket.sendall(message.encode())

    except Exception as e:
        print(f"Error in server loop: {e}")
    finally:
        client_socket.close()
        server_socket.close()
        print("Server closed.")

# ---------------------------------------------------------------------
# ------------------ FEATURE TRANSFORMER (SCALER + PCA) ---------------
# ---------------------------------------------------------------------

class FeatureTransformer:
    """
    Encapsulates StandardScaler + PCA. 
    During training:
       1) fit_transform -> learns standard scaling, then PCA
    During inference:
       1) transform -> uses the stored scalers to transform new data
    """

    def __init__(self):
        self.scaler = None
        self.pca    = None

    def fit_transform(self, df, exclude_cols=None, pca_variance=0.99):
        if exclude_cols is None:
            # Exclude steering, throttle, brake from the transform
            exclude_cols = ["steering", "throttle", "brake"]
        
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                        if c not in exclude_cols]

        self.numeric_cols = numeric_cols
        self.exclude_cols = exclude_cols

        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

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
        if (self.scaler is None) or (self.pca is None):
            raise RuntimeError("FeatureTransformer not fitted yet!")

        numeric_cols = self.numeric_cols
        exclude_cols = self.exclude_cols

        # If some numeric_cols don't exist in df, you might need default=0
        for c in numeric_cols:
            if c not in df.columns:
                df[c] = 0.0

        scaled_data = self.scaler.transform(df[numeric_cols])
        pca_data    = self.pca.transform(scaled_data)

        pc_columns = [f"PC{i+1}" for i in range(pca_data.shape[1])]
        df_pca = pd.DataFrame(pca_data, columns=pc_columns)

        # Keep the excluded columns if present
        df_excl = pd.DataFrame()
        for c in exclude_cols:
            if c in df.columns:
                df_excl[c] = df[c].values

        df_excl = df_excl.reset_index(drop=True)
        df_out = pd.concat([df_pca, df_excl], axis=1)
        return df_out

    def save(self, path="transformer.joblib"):
        joblib.dump({
            "scaler": self.scaler,
            "pca":    self.pca,
            "numeric_cols": self.numeric_cols,
            "exclude_cols": self.exclude_cols
        }, path)

    def load(self, path="transformer.joblib"):
        data = joblib.load(path)
        self.scaler      = data["scaler"]
        self.pca         = data["pca"]
        self.numeric_cols= data["numeric_cols"]
        self.exclude_cols= data["exclude_cols"]


# ---------------------------------------------------------------------
# ------------------  NNDriver -------
# ---------------------------------------------------------------------

class NNDriver:
    """
    A class to encapsulate an MLPRegressor neural network.

    By default:
      - We assume the last three columns are the 'targets' (steering, throttle, brake).
      - The rest are input features.
    You can override by specifying custom input/output columns.
    """

    def __init__(self,
                 hidden_layer_sizes=(64, 56, 48, 40, 32, 24, 16, 8),
                 alpha_value=0.01,
                 learning_rate='adaptive',
                 learning_rate_init=0.001,
                 max_iter=1000,
                 tol=1e-6,
                 random_state=42,
                 verbose=True,
                 early_stopping=False):
        """
        Initialize the neural network model (MLPRegressor).
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='sgd',
            alpha=alpha_value,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            shuffle=True,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            early_stopping=early_stopping
        )
        self.input_cols  = None
        self.output_cols = None

    def train(self, df, input_cols=None, output_cols=None):
        """
        Train on the columns of df. If no columns are specified, 
        use all except last 3 as inputs, and last 3 as outputs.
        """
        if input_cols is None:
            # Assume all columns except last three are inputs
            input_cols = df.columns[:-3]
        if output_cols is None:
            # Assume last three columns are the targets
            output_cols = df.columns[-3:]
        self.input_cols = list(input_cols)
        self.output_cols= list(output_cols)

        X = df[self.input_cols].values
        y = df[self.output_cols].values
        self.model.fit(X, y)

    def predict(self, df):
        """
        df: DataFrame with the same input columns as used in training.
        Returns predictions for each row.
        """
        if (self.input_cols is None):
            raise RuntimeError("NNDriver not trained yet: input_cols is None.")
        X = df[self.input_cols].values
        return self.model.predict(X)

    def evaluate(self, df):
        """
        Evaluate MSE on a given df that includes both input & output columns.
        """
        X = df[self.input_cols].values
        y = df[self.output_cols].values
        y_pred = self.model.predict(X)
        mse_value = mean_squared_error(y, y_pred)
        return mse_value

    def get_loss(self):
        """
        Return the final loss from MLPRegressor.
        """
        return self.model.loss_

class SingleStepPostprocessor:
    """
    Computes the same features as your offline pipeline, 
    but for a single time-step at a time.
    """

    def __init__(self, 
                 resampled_centerline_x, 
                 resampled_centerline_z,
                 track_width_data,
                 curvature_data,
                 ordered_blue,
                 ordered_yellow,
                 shift_distance=1.5,
                 max_ray_distance=20.0):
        """
        Store all static track data and configuration needed 
        for single-step feature computation.
        
        Parameters
        ----------
        resampled_centerline_x, resampled_centerline_z : arrays
            The track's centerline points (already reversed & resampled).
        track_width_data : list of dict
            Precomputed local track widths from the offline pipeline 
            (each dict has {"center": (x,z), "width": w}).
        curvature_data : list or array
            Precomputed curvature for each centerline point.
        ordered_blue, ordered_yellow : lists of (x,z)
            The track edges for raycasting.
        shift_distance : float
            Same shift you did offline, if needed.
        max_ray_distance : float
            The maximum distance for ray intersection.
        """
        self.clx = resampled_centerline_x
        self.clz = resampled_centerline_z
        self.track_widths_all = track_width_data
        self.curvatures_all   = curvature_data
        self.ordered_blue     = ordered_blue
        self.ordered_yellow   = ordered_yellow
        self.shift_distance   = shift_distance
        self.max_ray_distance = max_ray_distance

        # We keep them as list of tuples for quick usage
        self.centerline_pts   = list(zip(self.clx, self.clz))

        # We also prepare to store the last time, last vx, last vy 
        # if we want to compute accelerations:
        self.last_time = None
        self.last_vx   = None
        self.last_vy   = None

    def _shift_car_position(self, x, z, yaw_deg):
        """
        Shift the car position if you used the same offset offline.
        """
        yaw = math.radians(yaw_deg)
        offset_x = self.shift_distance * math.sin(yaw)
        offset_z = -self.shift_distance * math.cos(yaw)
        return x + offset_x, z + offset_z

    def _compute_acceleration(self, time, vx, vy):
        """
        Compute finite-difference acceleration from the previous step.
        """
        if self.last_time is None:
            # First step => no previous data
            ax = 0.0
            ay = 0.0
        else:
            dt = time - self.last_time
            if abs(dt) < 1e-9:
                ax, ay = 0.0, 0.0
            else:
                ax = (vx - self.last_vx) / dt
                ay = (vy - self.last_vy) / dt

        # Update stored values
        self.last_time = time
        self.last_vx   = vx
        self.last_vy   = vy
        return ax, ay

    def _ray_segment_intersection(self, ray_origin, ray_direction, seg_start, seg_end):
        """
        Same as your ray_segment_intersection. 
        (Omitted here for brevity.)
        """
        # ...
        pass  # you can paste your existing code

    def _raycast_for_state(self, car_x, car_z, car_heading):
        """
        Single-step version to compute ray distances to track edges
        (ordered_blue, ordered_yellow).
        """
        import numpy as np
        yellow_angles_deg = np.arange(-20, 111, 10)
        blue_angles_deg   = np.arange( 20, -111, -10)

        def cast_rays(edge, angles_deg):
            dists = []
            for rel_angle_deg in angles_deg:
                rel_angle = math.radians(rel_angle_deg)
                ray_angle = car_heading + rel_angle
                ray_dir = (math.cos(ray_angle), math.sin(ray_angle))
                closest_distance = self.max_ray_distance
                # check each segment
                for i in range(len(edge)-1):
                    seg_start = edge[i]
                    seg_end   = edge[i+1]
                    t_val = self._ray_segment_intersection((car_x, car_z), ray_dir, seg_start, seg_end)
                    if t_val is not None and t_val < closest_distance:
                        closest_distance = t_val
                dists.append(closest_distance)
            return dists

        yellow_ray_distances = cast_rays(self.ordered_yellow, yellow_angles_deg)
        blue_ray_distances   = cast_rays(self.ordered_blue,   blue_angles_deg)
        return yellow_ray_distances, blue_ray_distances

    def _find_projection_index(self, car_x, car_z):
        """
        Find the centerline point closest to (car_x, car_z).
        """
        dists = [math.hypot(px - car_x, pz - car_z) for (px, pz) in self.centerline_pts]
        i_min = int(np.argmin(dists))
        return i_min

    def compute_features_for_single_step(self, 
                                         time, 
                                         x_pos, 
                                         z_pos, 
                                         yaw_angle_deg, 
                                         long_vel, 
                                         lat_vel, 
                                         yaw_rate, 
                                         steering, 
                                         throttle, 
                                         brake):
        """
        Returns a dict of the same features you used offline, but for this single step.

        1) Shifts car position if needed.
        2) Compute acceleration from last step (if you want).
        3) Raycast to edges.
        4) Dist to centerline, heading difference, track width, curvature, etc.
        5) Return as a single-row DataFrame for further transformation.
        """
        # 1) Shift
        x_shifted, z_shifted = self._shift_car_position(x_pos, z_pos, yaw_angle_deg)

        # 2) Accel
        ax, ay = self._compute_acceleration(time, long_vel, lat_vel)

        # 3) Raycast
        yaw_rad = math.radians(yaw_angle_deg)
        yr_dists, br_dists = self._raycast_for_state(x_shifted, z_shifted, yaw_rad)

        # 4) Dist to centerline, heading difference
        i_proj = self._find_projection_index(x_shifted, z_shifted)
        # e.g., track width & curvature at i_proj
        if 0 <= i_proj < len(self.track_widths_all):
            tw0 = self.track_widths_all[i_proj]["width"]
            c0  = self.curvatures_all[i_proj]
        else:
            tw0 = float('nan')
            c0  = float('nan')

        # Signed distance
        dc = compute_signed_distance_to_centerline(x_shifted, z_shifted, self.clx, self.clz)
        # Heading diff
        dh = compute_heading_difference(x_shifted, z_shifted, yaw_rad, self.clx, self.clz)

        # 5) Collect into a dict
        row_dict = {
            "time": time,
            "x_pos": x_shifted,
            "z_pos": z_shifted,
            "yaw_deg": yaw_angle_deg,
            "long_vel": long_vel,
            "lat_vel": lat_vel,
            "yaw_rate": yaw_rate,
            "steering": steering,
            "throttle": throttle,
            "brake": brake,
            "ax": ax,
            "ay": ay,
            "dist_center": -dc,    # or dc
            "head_diff": dh,
            "track_width": tw0,
            "curvature": c0,
        }

        # Insert ray distances
        for idx, dist_val in enumerate(yr_dists, start=1):
            row_dict[f"yr{idx}"] = dist_val
        for idx, dist_val in enumerate(br_dists, start=1):
            row_dict[f"br{idx}"] = dist_val

        # Return as a single-row DataFrame
        df_single = pd.DataFrame([row_dict])
        return df_single


# ---------------------------------------------------------------------
# ------------------ MAIN ORCHESTRATOR CLASS --------------------------
# ---------------------------------------------------------------------

class NNDriverFramework:
    """
    High-level class that orchestrates:
      - Reading CSV & track JSON
      - Postprocessing to generate features
      - Fitting/applying StandardScaler + PCA
      - Training or real-time inference with an MLPRegressor-based NNDriver
    """

    def __init__(self, transformer_path="transformer.joblib", model_path="nn_model.joblib"):
        self.transformer_path = transformer_path
        self.model_path       = model_path

        # The Scaler + PCA pipeline
        self.transformer      = FeatureTransformer()

        # The MLPRegressor-based driver
        self.nn_model         = NNDriver()

    def postprocess_csv(self, data_dict, blue_cones, yellow_cones, clx, clz):
        """
        Emulates postprocessing for each CSV row -> returns DataFrame with relevant features.
        """
        # SHIFT if needed
        data_dict = shift_car_position(data_dict)

        # Reverse centerline
        clx_rev = clx[::-1]
        clz_rev = clz[::-1]

        # Resample
        r_clx, r_clz = resample_centerline(clx_rev, clz_rev, resolution=1.0)
        centerline_pts = list(zip(r_clx, r_clz))

        # Edges
        ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx_rev, clz_rev)

        # Precompute curvature + track width
        curvatures_all = compute_local_curvature(r_clx, r_clz, window_size=5)
        track_widths_all = compute_local_track_widths(r_clx, r_clz, ordered_blue, ordered_yellow, max_width=10.0)

        # Build output row by row
        rows = []
        time_arr = data_dict["time"]
        vx_arr   = data_dict["long_vel"]
        vy_arr   = data_dict["lat_vel"]
        ax_arr, ay_arr = compute_accelerations(time_arr, vx_arr, vy_arr)

        for i in range(len(time_arr)):
            car_x   = data_dict["x_pos"][i]
            car_z   = data_dict["z_pos"][i]
            yaw_deg = data_dict["yaw_angle"][i]
            yaw_rad = math.radians(yaw_deg)

            # Raycasting
            yrd, brd = raycast_for_state(car_x, car_z, yaw_rad, ordered_blue, ordered_yellow, max_distance=20.0)

            # Dist to centerline
            dc = compute_signed_distance_to_centerline(car_x, car_z, r_clx, r_clz)
            # Heading difference
            dh = compute_heading_difference(car_x, car_z, yaw_rad, r_clx, r_clz)

            # Track width + curvature for the projection
            dists = [math.hypot(px - car_x, pz - car_z) for px,pz in centerline_pts]
            i_proj = int(np.argmin(dists))
            tw0 = track_widths_all[i_proj]["width"] if i_proj < len(track_widths_all) else np.nan
            c0  = curvatures_all[i_proj] if i_proj < len(curvatures_all) else np.nan

            row_dict = {
                "time":      time_arr[i],
                "x_pos":     car_x,
                "z_pos":     car_z,
                "yaw_deg":   yaw_deg,
                "long_vel":  vx_arr[i],
                "lat_vel":   vy_arr[i],
                "yaw_rate":  data_dict["yaw_rate"][i],
                "steering":  data_dict["steering"][i],
                "throttle":  data_dict["throttle"][i],
                "brake":     data_dict["brake"][i],
                "ax":        ax_arr[i],
                "ay":        ay_arr[i],
                "dist_center": -dc,  # or dc, depending on sign preference
                "head_diff": dh,
                "track_width": tw0,
                "curvature":   c0,
            }
            # Add ray distances
            for idx, dist_val in enumerate(yrd, start=1):
                row_dict[f"yr{idx}"] = dist_val
            for idx, dist_val in enumerate(brd, start=1):
                row_dict[f"br{idx}"] = dist_val

            rows.append(row_dict)

        df_out = pd.DataFrame(rows)
        return df_out

    def train_mode(self, csv_path, json_path, output_csv_path=None, pca_variance=0.99, test_split=0.2):
        """
        Reads CSV & JSON, builds a DataFrame with features,
        then fits StandardScaler + PCA, and trains the MLPRegressor.
        Saves the fitted pipeline and the model.
        """
        print("[NNDriverFramework] Training mode...")

        # 1) Read raw data
        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return

        # 2) Parse track
        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)

        # 3) Postprocess -> features
        df_features = self.postprocess_csv(data_dict, blue_cones, yellow_cones, clx, clz)
        if output_csv_path:
            df_features.to_csv(output_csv_path, index=False)
            print(f"[NNDriverFramework] Postprocessed CSV saved: {output_csv_path}")

        # 4) Fit StandardScaler + PCA
        df_trans = self.transformer.fit_transform(df_features, 
                                                  exclude_cols=["steering","throttle","brake"],
                                                  pca_variance=pca_variance)

        self.transformer.save(self.transformer_path)
        print(f"[NNDriverFramework] Transformer saved to {self.transformer_path}")

        # 5) We want the last 3 columns to be the outputs: (steering, throttle, brake)
        #    The DF from transformer is: [PC1..PCn, steering, throttle, brake]
        #    Train the NN on that.
        #    But let's do a train/test split first, for example:

        # Input columns = all columns that start with "PC"
        pc_cols = [c for c in df_trans.columns if c.startswith("PC")]
        # Output columns = "steering","throttle","brake"
        out_cols = ["steering","throttle","brake"]

        # Make sure out_cols actually exist in df_trans
        X = df_trans[pc_cols].values
        y = df_trans[out_cols].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)

        # Put them in a small df for the training method:
        train_df = pd.DataFrame(X_train, columns=pc_cols)
        train_df["steering"] = y_train[:,0]
        train_df["throttle"] = y_train[:,1]
        train_df["brake"]    = y_train[:,2]

        test_df = pd.DataFrame(X_test, columns=pc_cols)
        test_df["steering"] = y_test[:,0]
        test_df["throttle"] = y_test[:,1]
        test_df["brake"]    = y_test[:,2]

        # 6) Train the NNDriver
        self.nn_model.train(train_df)

        # Evaluate on test set
        mse_value = self.nn_model.evaluate(test_df)
        print("[NNDriverFramework] Test MSE:", mse_value)
        print("[NNDriverFramework] Final Loss:", self.nn_model.get_loss())

        # 7) Save the NN model (via joblib or pickle):
        joblib.dump(self.nn_model, self.model_path)
        print(f"[NNDriverFramework] NN model saved to {self.model_path}")

    def inference_mode(self, csv_path, json_path):
        """
        Inference mode:
           1) Load the pre-fitted StandardScaler & PCA
           2) Load the trained NNDriver
           3) Postprocess new CSV data, transform, predict
        """
        print("[NNDriverFramework] Inference mode...")

        # Load pipeline
        self.transformer.load(self.transformer_path)
        self.nn_model = joblib.load(self.model_path)  # we get the entire NNDriver object

        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return

        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)

        df_features = self.postprocess_csv(data_dict, blue_cones, yellow_cones, clx, clz)

        # Transform the features
        df_trans = self.transformer.transform(df_features)

        # Predict
        predictions = self.nn_model.predict(df_trans)  # shape (N,3)
        
        # For demonstration, print them out
        times = df_features["time"].values
        for i in range(len(df_features)):
            st, th, br = predictions[i]
            t = times[i]
            print(f"[Inference] t={t:.2f}s => steering={st:.3f}, throttle={th:.3f}, brake={br:.3f}")

        print("[NNDriverFramework] Inference complete.")

class RealtimeDriver:
    def __init__(self, 
                 transformer_path="transformer.joblib", 
                 model_path="nn_model.joblib", 
                 track_json="default.json", 
                 shift_distance=1.5, 
                 max_ray_distance=20.0):
        """
        1) Load the pre-fitted StandardScaler+PCA (FeatureTransformer).
        2) Load the trained MLPRegressor (NNDriver).
        3) Parse track geometry, resample centerline, compute track edges, 
           curvature, track width.
        4) Create a SingleStepPostprocessor to do the real-time feature extraction.
        """
        # 1) Load pipeline
        self.transformer = FeatureTransformer()
        self.transformer.load(transformer_path)
        print(f"[RealtimeDriver] Loaded transformer from {transformer_path}")

        # 2) Load model
        self.nn_model = joblib.load(model_path)  # This is an instance of NNDriver
        print(f"[RealtimeDriver] Loaded NN model from {model_path}")

        # 3) Parse track geometry
        from pathlib import Path
        track_json = Path(track_json)
        with open(track_json, 'r') as f:
            import json
            data = json.load(f)
        x_values = data.get("x", [])
        y_values = data.get("y", [])
        colors   = data.get("color", [])
        clx      = data.get("centerline_x", [])
        clz      = data.get("centerline_y", [])
        
        # Build cones
        blue_cones = [(x,z) for x,z,c in zip(x_values, y_values, colors) if c.lower()=="blue"]
        yellow_cones= [(x,z) for x,z,c in zip(x_values, y_values, colors) if c.lower()=="yellow"]

        # Reverse centerline to match your offline
        clx_rev = clx[::-1]
        clz_rev = clz[::-1]

        # Resample
        r_clx, r_clz = resample_centerline(clx_rev, clz_rev, resolution=1.0)

        # Create edges
        ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx_rev, clz_rev)
        
        # Curvature + track width
        curvature_data   = compute_local_curvature(r_clx, r_clz, window_size=5)
        track_width_data = compute_local_track_widths(r_clx, r_clz, ordered_blue, ordered_yellow, max_width=10.0)

        # 4) Make a single-step postprocessor
        self.single_step_processor = SingleStepPostprocessor(
            resampled_centerline_x=r_clx,
            resampled_centerline_z=r_clz,
            track_width_data=track_width_data,
            curvature_data=curvature_data,
            ordered_blue=ordered_blue,
            ordered_yellow=ordered_yellow,
            shift_distance=shift_distance,
            max_ray_distance=max_ray_distance
        )
        print("[RealtimeDriver] Initialized SingleStepPostprocessor.")

    def process_single_step(self, 
                            time, 
                            x_pos, 
                            z_pos, 
                            yaw_angle_deg, 
                            long_vel, 
                            lat_vel, 
                            yaw_rate, 
                            steering_in, 
                            throttle_in, 
                            brake_in):
        """
        Given the real-time sensor/vehicle data at one time step, 
        do the same postprocessing -> transform -> NN predict -> return commands.
        
        Returns: (pred_steering, pred_throttle, pred_brake)
        """
        # 1) Build the feature row
        df_single = self.single_step_processor.compute_features_for_single_step(
            time=time,
            x_pos=x_pos,
            z_pos=z_pos,
            yaw_angle_deg=yaw_angle_deg,
            long_vel=long_vel,
            lat_vel=lat_vel,
            yaw_rate=yaw_rate,
            steering=steering_in,
            throttle=throttle_in,
            brake=brake_in
        )

        # 2) Apply the same StandardScaler + PCA
        df_trans = self.transformer.transform(df_single)

        # 3) NN predict
        predictions = self.nn_model.predict(df_trans)  # shape (1,3) 
        st_pred, th_pred, br_pred = predictions[0]

        return (st_pred, th_pred, br_pred)

# ---------------------------------------------------------------------
# ------------------ MAIN CLI SCRIPT ----------------------------------
# ---------------------------------------------------------------------

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", help="train, infer, or realtime")
    parser.add_argument("--csv",  type=str, help="Path to CSV file (for train or infer)")
    parser.add_argument("--json", type=str, default="default.json", help="Path to track JSON")
    parser.add_argument("--transformer", type=str, default="transformer.joblib",
                        help="Path to save/load the fitted scaler/PCA")
    parser.add_argument("--model", type=str, default="nn_model.joblib",
                        help="Path to save/load the trained model")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional path to save postprocessed CSV in train mode")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="TCP server host")
    parser.add_argument("--port", type=int, default=65432, help="TCP server port")
    args = parser.parse_args()

    if args.mode.lower() == "train":
        if not args.csv:
            print("Must provide --csv for training.")
            return
        framework = NNDriverFramework(transformer_path=args.transformer, model_path=args.model)
        framework.train_mode(args.csv, args.json, output_csv_path=args.output_csv)
    elif args.mode.lower() == "infer":
        if not args.csv:
            print("Must provide --csv for inference.")
            return
        framework = NNDriverFramework(transformer_path=args.transformer, model_path=args.model)
        framework.inference_mode(args.csv, args.json)
    elif args.mode.lower() == "realtime":
        # Create RealtimeDriver
        rt_driver = RealtimeDriver(
            transformer_path=args.transformer,
            model_path=args.model,
            track_json=args.json
        )
        # Start the TCP server
        run_tcp_server(rt_driver, host=args.host, port=args.port)
    else:
        print("Unknown mode. Use --mode train, infer, or realtime.")


if __name__ == "__main__":
    main()
