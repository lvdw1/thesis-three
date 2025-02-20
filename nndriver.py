#!/usr/bin/env python3
"""
nndriver.py

Unified codebase for:
  - Reading and post-processing data
  - Applying StandardScaler + PCA
  - Training and/or running inference with an MLPRegressor-based NN.
  - Realtime driving via TCP socket.
  - Visualizing postprocessed data (to confirm the NN’s inputs look correct).
  - Visualizing data in "realtime" fashion, calling process_realtime_frame per timestep.
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
    clz      = data.get("centerline_y", [])
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
    
    if ordered_blue:
        ordered_blue.append(ordered_blue[0])
    if ordered_yellow:
        ordered_yellow.append(ordered_yellow[0])
    
    return ordered_blue, ordered_yellow

def read_csv_data(file_path):
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
        "time": np.array(times),
        "x_pos": np.array(x_pos),
        "z_pos": np.array(z_pos),
        "yaw_angle": np.array(yaw_angle),
        "long_vel": np.array(long_vel),
        "lat_vel": np.array(lat_vel),
        "yaw_rate": np.array(yaw_rate),
        "steering": np.array(steering),
        "throttle": np.array(throttle),
        "brake": np.array(brake),
    }

def shift_position_single(x, z, yaw_deg, shift_distance=-1.5):
    yaw = np.radians(yaw_deg)
    offset_x = shift_distance * np.sin(yaw)
    offset_z = -shift_distance * np.cos(yaw)
    return x + offset_x, z + offset_z

def find_projection_index(car_x, car_z, centerline_pts):
    dists = [math.hypot(px - car_x, pz - car_z) for (px, pz) in centerline_pts]
    return int(np.argmin(dists))

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
        seg_end = edge_points[i+1]
        t_val = ray_segment_intersection(ray_origin, ray_direction, seg_start, seg_end)
        if t_val is not None and t_val < best_t:
            best_t = t_val
            found = True
    return best_t if found else None

def raycast_for_state(car_x, car_z, car_heading, blue_edge, yellow_edge, max_distance=20):
    yellow_angles_deg = np.arange(-20, 111, 10)
    blue_angles_deg = np.arange(20, -111, -10)
    yellow_ray_distances = []
    blue_ray_distances = []
    for rel_angle_deg in yellow_angles_deg:
        rel_angle = math.radians(rel_angle_deg)
        ray_angle = car_heading + rel_angle
        ray_dir = (math.cos(ray_angle), math.sin(ray_angle))
        closest_distance = max_distance
        for i in range(len(yellow_edge)-1):
            seg_start = yellow_edge[i]
            seg_end = yellow_edge[i+1]
            t_val = ray_segment_intersection((car_x, car_z), ray_dir, seg_start, seg_end)
            if t_val is not None and t_val < closest_distance:
                closest_distance = t_val
        yellow_ray_distances.append(closest_distance)
    for rel_angle_deg in blue_angles_deg:
        rel_angle = math.radians(rel_angle_deg)
        ray_angle = car_heading + rel_angle
        ray_dir = (math.cos(ray_angle), math.sin(ray_angle))
        closest_distance = max_distance
        for i in range(len(blue_edge)-1):
            seg_start = blue_edge[i]
            seg_end = blue_edge[i+1]
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
        end = min(N, i + half_window + 1)
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
            # sign the curvature based on cross product
            if i > 0 and i < N-1:
                vec1 = np.array([centerline_x[i-1] - center_x, centerline_z[i-1] - center_y])
                vec2 = np.array([centerline_x[i+1] - center_x, centerline_z[i+1] - center_y])
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
        left_normal = (-T[1], T[0])
        right_normal = (T[1], -T[0])
        center = pts[i]
        d_yellow = compute_ray_edge_intersection_distance(center, left_normal, ordered_yellow, max_distance=max_width)
        d_blue = compute_ray_edge_intersection_distance(center, right_normal, ordered_blue, max_distance=max_width)
        if d_yellow is None: d_yellow = max_width
        if d_blue is None: d_blue = max_width
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
        elif i == N-1:
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
            norm_v = math.sqrt(v_dot_v) if v_dot_v != 0 else 1e-9
            tangent = (vx/norm_v, vz/norm_v)
            left_normal = (-tangent[1], tangent[0])
            diff_vec = (car_x - proj[0], car_z - proj[1])
            sign = 1 if (diff_vec[0]*left_normal[0] + diff_vec[1]*left_normal[1]) >= 0 else -1
            best_signed_distance = sign * dist
    return best_signed_distance

def get_local_centerline_points_by_distance(car_x, car_z, car_yaw,
                                           centerline_points,
                                           front_distance=20.0,
                                           behind_distance=5.0):
    pts = np.array(centerline_points)  # shape (N,2)
    cum_dist = np.array(compute_centerline_cumulative_distance(pts[:,0].tolist(),
                                                               pts[:,1].tolist()))
    N = len(pts)
    dists = np.hypot(pts[:,0] - car_x, pts[:,1] - car_z)
    i_proj = int(np.argmin(dists))
    L_proj = cum_dist[i_proj]
    L_front_target = L_proj + front_distance
    L_behind_target = L_proj - behind_distance

    def interpolate_point(target_L):
        if target_L <= cum_dist[0]:
            return pts[0]
        if target_L >= cum_dist[-1]:
            return pts[-1]
        idx = np.searchsorted(cum_dist, target_L)
        L1, L2 = cum_dist[idx-1], cum_dist[idx]
        p1, p2 = pts[idx-1], pts[idx]
        ratio = (target_L - L1) / (L2 - L1)
        return p1 + ratio * (p2 - p1)
    
    global_front_pt = interpolate_point(L_front_target)
    global_behind_pt = interpolate_point(L_behind_target)
    
    front_mask = (cum_dist >= L_proj) & (cum_dist <= L_front_target)
    behind_mask = (cum_dist >= L_behind_target) & (cum_dist <= L_proj)
    
    global_front = pts[front_mask].tolist()
    global_behind = pts[behind_mask].tolist()
    
    if len(global_front) == 0 or np.hypot(*(np.array(global_front[-1]) - global_front_pt)) > 1e-3:
        global_front.append(global_front_pt.tolist())
    if len(global_behind) == 0 or np.hypot(*(np.array(global_behind[0]) - global_behind_pt)) > 1e-3:
        global_behind.insert(0, global_behind_pt.tolist())
    
    front_local = []
    indices_front = np.where(front_mask)[0]
    for idx in indices_front:
        p = pts[idx]
        arc_offset = cum_dist[idx] - L_proj
        dx = p[0] - car_x
        dz = p[1] - car_z
        lateral = -dx * math.sin(car_yaw) + dz * math.cos(car_yaw)
        front_local.append((arc_offset, lateral, 0))
    if len(global_front) > len(indices_front):
        p = np.array(global_front_pt)
        dx = p[0] - car_x
        dz = p[1] - car_z
        lateral = -dx * math.sin(car_yaw) + dz * math.cos(car_yaw)
        front_local.append((front_distance, lateral, 0))
    
    behind_local = []
    indices_behind = np.where(behind_mask)[0]
    for idx in indices_behind:
        p = pts[idx]
        arc_offset = cum_dist[idx] - L_proj
        dx = p[0] - car_x
        dz = p[1] - car_z
        lateral = -dx * math.sin(car_yaw) + dz * math.cos(car_yaw)
        behind_local.append((arc_offset, lateral, 0))
    if len(global_behind) > len(indices_behind):
        p = np.array(global_behind_pt)
        dx = p[0] - car_x
        dz = p[1] - car_z
        lateral = -dx * math.sin(car_yaw) + dz * math.cos(car_yaw)
        behind_local.insert(0, (-behind_distance, lateral, 0))
    
    return front_local, behind_local, global_front, global_behind

def compute_acceleration(time, vx, vy):
    time = np.asarray(time)
    vx = np.asarray(vx)
    vy = np.asarray(vy)
    dt = np.diff(time)
    dt[dt < 1e-9] = 1e-9
    dvx = np.diff(vx)
    dvy = np.diff(vy)
    ax = dvx / dt
    ay = dvy / dt
    ax = np.concatenate([ax, [ax[-1]]])
    ay = np.concatenate([ay, [ay[-1]]])
    return ax, ay

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
# ------------------  NNDriver -------
# ---------------------------------------------------------------------

class NNDriver:
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
        self.input_cols = None
        self.output_cols = None

    def train(self, df, input_cols=None, output_cols=None):
        if input_cols is None:
            input_cols = df.columns[:-3]
        if output_cols is None:
            output_cols = df.columns[-3:]
        self.input_cols = list(input_cols)
        self.output_cols = list(output_cols)
        X = df[self.input_cols].values
        y = df[self.output_cols].values
        self.model.fit(X, y)

    def predict(self, df):
        if self.input_cols is None:
            raise RuntimeError("NNDriver not trained yet: input_cols is None.")
        X = df[self.input_cols].values
        return self.model.predict(X)

    def evaluate(self, df):
        X = df[self.input_cols].values
        y = df[self.output_cols].values
        y_pred = self.model.predict(X)
        mse_value = mean_squared_error(y, y_pred)
        return mse_value

    def get_loss(self):
        return self.model.loss_

# ---------------------------------------------------------------------
# ------------------  NNDriverFramework (Centralized Realtime + Visual) ---------------
# ---------------------------------------------------------------------

class NNDriverFramework:
    def __init__(self, transformer_path="transformer.joblib", model_path="nn_model.joblib"):
        self.transformer_path = transformer_path
        self.model_path = model_path
        self.transformer = FeatureTransformer()
        self.nn_model = NNDriver()
        # For realtime acceleration state
        self.last_time = None
        self.last_vx = None
        self.last_vy = None

    def process_csv(self, data_dict, track_data):
        """Process an entire CSV file frame‐by‐frame using process_realtime_frame."""
        # Reset realtime state for consistent processing
        self.last_time = None
        self.last_vx = None
        self.last_vy = None
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
            frame = self.process_realtime_frame(sensor_data, track_data)
            frames.append(frame)
        return pd.DataFrame(frames)


    def process_realtime_frame(self, sensor_data, track_data):
        """
        For each new sensor reading (car_x, car_z, yaw, velocities,...),
        compute the same features used for training.
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

        x_shifted, z_shifted = shift_position_single(car_x, car_z, yaw_deg, shift_distance=2.5)
        # compute local acceleration from last frame
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
        yrd, brd = raycast_for_state(x_shifted, z_shifted, yaw_rad,
                                     track_data["ordered_blue"],
                                     track_data["ordered_yellow"],
                                     max_distance=20.0)
        dc = compute_signed_distance_to_centerline(x_shifted, z_shifted,
                                                   track_data["r_clx"],
                                                   track_data["r_clz"])
        dh = compute_heading_difference(x_shifted, z_shifted, yaw_rad,
                                        track_data["r_clx"], track_data["r_clz"])
        dists = [math.hypot(px - car_x, pz - car_z) for (px, pz) in track_data["centerline_pts"]]
        i_proj = int(np.argmin(dists))
        if i_proj < len(track_data["r_clx"]):
            c0 = track_data["curvatures_all"][i_proj]
            tw0 = track_data["track_widths_all"][i_proj]["width"]
        else:
            c0 = 0.0
            tw0 = 4

        front_local, behind_local, _, _ = get_local_centerline_points_by_distance(
            x_shifted, z_shifted, yaw_rad, track_data["centerline_pts"],
            front_distance=20.0, behind_distance=5.0
        )
        if len(front_local) > 0:
            fl = np.array(front_local)
            x_front = fl[:,0]
            z_front = fl[:,1]
            target_x = np.arange(1,21)
            target_z = np.interp(target_x, x_front, z_front, left=z_front[0], right=z_front[-1])
        else:
            target_x = np.arange(1,21)
            target_z = np.full(20, 0)
        if len(behind_local) > 0:
            bl = np.array(behind_local)
            x_behind = bl[:,0]
            z_behind = bl[:,1]
            target_x_b = np.arange(-5,0)
            target_z_b = np.interp(target_x_b, x_behind, z_behind, left=z_behind[0], right=z_behind[-1])
        else:
            target_x_b = np.arange(-5,0)
            target_z_b = np.full(5, 0)

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
            row_dict[f"rel_z{j}"] = target_z[j-1]
            idx_front = i_proj + int(round(d))
            if idx_front < len(track_data["r_clx"]):
                row_dict[f"c{j}"] = track_data["curvatures_all"][idx_front]
                row_dict[f"tw{j}"] = track_data["track_widths_all"][idx_front]["width"]
            else:
                row_dict[f"c{j}"] = 0.0
                row_dict[f"tw{j}"] = 4.0
        for j, d in enumerate(target_x_b, start=1):
            row_dict[f"b_rel_z{j}"] = target_z_b[j-1]
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

    def train_mode(self, csv_path, json_path, output_csv_path=None,
                   pca_variance=0.99, test_split=0.2):
        print("[NNDriverFramework] Training mode...")
        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return
        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)

        # 1) Postprocess the CSV to produce raw features
        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)
        # Build track_data (same as in realtime_mode)
        clx_rev = clx[::-1]
        clz_rev = clz[::-1]
        r_clx, r_clz = resample_centerline(clx_rev, clz_rev, resolution=1.0)
        centerline_pts = list(zip(r_clx, r_clz))
        ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx_rev, clz_rev)
        curvatures_all = compute_local_curvature(r_clx, r_clz, window_size=5)
        track_widths_all = compute_local_track_widths(r_clx, r_clz, ordered_blue, ordered_yellow, max_width=10.0)
        track_data = {
            "r_clx": r_clx,
            "r_clz": r_clz,
            "centerline_pts": centerline_pts,
            "ordered_blue": ordered_blue,
            "ordered_yellow": ordered_yellow,
            "curvatures_all": curvatures_all,
            "track_widths_all": track_widths_all
        }
        df_features = self.process_csv(data_dict, track_data)
        if output_csv_path:
            df_features.to_csv(output_csv_path, index=False)
            print(f"[NNDriverFramework] Postprocessed CSV saved to {output_csv_path}")

        # 2) Fit the transformer (scaler + PCA) on the raw features
        df_trans = self.transformer.fit_transform(df_features,
                                                  exclude_cols=["steering","throttle","brake"],
                                                  pca_variance=pca_variance)
        self.transformer.save(self.transformer_path)
        print(f"[NNDriverFramework] Transformer saved to {self.transformer_path}")

        # 3) Train the MLP model
        pc_cols = [c for c in df_trans.columns if c.startswith("PC")]
        out_cols = ["steering", "throttle", "brake"]
        X = df_trans[pc_cols].values
        y = df_trans[out_cols].values
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_split,
                                                            random_state=42)

        train_df = pd.DataFrame(X_train, columns=pc_cols)
        train_df["steering"] = y_train[:, 0]
        train_df["throttle"] = y_train[:, 1]
        train_df["brake"] = y_train[:, 2]

        test_df = pd.DataFrame(X_test, columns=pc_cols)
        test_df["steering"] = y_test[:, 0]
        test_df["throttle"] = y_test[:, 1]
        test_df["brake"] = y_test[:, 2]

        self.nn_model.train(train_df)
        mse_value = self.nn_model.evaluate(test_df)
        print("[NNDriverFramework] Test MSE:", mse_value)
        print("[NNDriverFramework] Final Loss:", self.nn_model.get_loss())

        joblib.dump(self.nn_model, self.model_path)
        print(f"[NNDriverFramework] NN model saved to {self.model_path}")

    def inference_mode(self, csv_path, json_path):
        print("[NNDriverFramework] Inference mode...")
        self.transformer.load(self.transformer_path)
        self.nn_model = joblib.load(self.model_path)
        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return
        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)

        # postprocess to produce same columns
        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)
        clx_rev = clx[::-1]
        clz_rev = clz[::-1]
        r_clx, r_clz = resample_centerline(clx_rev, clz_rev, resolution=1.0)
        centerline_pts = list(zip(r_clx, r_clz))
        ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx_rev, clz_rev)
        curvatures_all = compute_local_curvature(r_clx, r_clz, window_size=5)
        track_widths_all = compute_local_track_widths(r_clx, r_clz, ordered_blue, ordered_yellow, max_width=10.0)
        track_data = {
            "r_clx": r_clx,
            "r_clz": r_clz,
            "centerline_pts": centerline_pts,
            "ordered_blue": ordered_blue,
            "ordered_yellow": ordered_yellow,
            "curvatures_all": curvatures_all,
            "track_widths_all": track_widths_all
        }
        df_features = self.process_csv(data_dict, track_data)
        df_trans = self.transformer.transform(df_features)

        # run predictions
        predictions = self.nn_model.predict(df_trans)
        times = df_features["time"].values
        for i in range(len(df_features)):
            st, th, br = predictions[i]
            t = times[i]
            print(f"[Inference] t={t:.2f}s => steering={st:.3f}, throttle={th:.3f}, brake={br:.3f}")
        print("[NNDriverFramework] Inference complete.")

    def realtime_mode(self, track_json, host='127.0.0.1', port=65432):
        print("[NNDriverFramework] Realtime mode...")
        self.transformer.load(self.transformer_path)
        self.nn_model = joblib.load(self.model_path)

        with open(track_json, 'r') as f:
            data = json.load(f)
        x_values = data.get("x", [])
        y_values = data.get("y", [])
        colors = data.get("color", [])
        clx = data.get("centerline_x", [])
        clz = data.get("centerline_y", [])
        blue_cones = [(x, z) for x, z, c in zip(x_values, y_values, colors) if c.lower() == "blue"]
        yellow_cones = [(x, z) for x, z, c in zip(x_values, y_values, colors) if c.lower() == "yellow"]

        # build track data for repeated usage
        clx_rev = clx[::-1]
        clz_rev = clz[::-1]
        r_clx, r_clz = resample_centerline(clx_rev, clz_rev, resolution=1.0)
        centerline_pts = list(zip(r_clx, r_clz))
        ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx_rev, clz_rev)
        curvatures_all = compute_local_curvature(r_clx, r_clz, window_size=5)
        track_widths_all = compute_local_track_widths(r_clx, r_clz, ordered_blue, ordered_yellow, max_width=10.0)

        track_data = {
            "r_clx": r_clx,
            "r_clz": r_clz,
            "centerline_pts": centerline_pts,
            "ordered_blue": ordered_blue,
            "ordered_yellow": ordered_yellow,
            "curvatures_all": curvatures_all,
            "track_widths_all": track_widths_all
        }

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
                if len(fields) < 6:
                    print(raw_data)
                    continue
                sensor_data = {
                    "time": time.time(),
                    "x_pos": float(fields[0]),
                    "z_pos": float(fields[1]),
                    "yaw_deg": float(fields[2]),
                    "long_vel": float(fields[3]),
                    "lat_vel": float(fields[4]),
                    "yaw_rate": float(fields[5]),
                    "steering": 0.0,
                    "throttle": 0.0,
                    "brake": 0.0
                }
                # produce single-row features
                row_dict = self.process_realtime_frame(sensor_data, track_data)
                df_single = pd.DataFrame([row_dict])
                df_trans = self.transformer.transform(df_single)
                predictions = self.nn_model.predict(df_trans)
                st_pred, th_pred, br_pred = predictions[0]
                message = f"{st_pred},{th_pred},{0}\n"
                print(f"[Realtime] Sending: {message.strip()}")
                client_socket.sendall(message.encode())
        except Exception as e:
            print(f"[Realtime] Error: {e}")
        finally:
            client_socket.close()
            server_socket.close()
            print("[Realtime] Server closed.")


    def visual_realtime_mode(self, csv_path, json_path, heading_length=3.0):
        """
        Processes each CSV row one at a time (via process_realtime_frame)
        and updates the plot immediately before moving on.
        """
        print("[NNDriverFramework] Visualize-Realtime mode (immediate plotting)...")
        # Reset any 'last_*' state so each run starts fresh
        self.last_time = None
        self.last_vx = None
        self.last_vy = None

        data_dict = read_csv_data(csv_path)
        if data_dict is None:
            print("Could not load CSV data.")
            return

        # Build the same 'track_data' used in realtime_mode
        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)
        clx_rev = clx[::-1]
        clz_rev = clz[::-1]
        r_clx, r_clz = resample_centerline(clx_rev, clz_rev, resolution=1.0)
        centerline_pts = list(zip(r_clx, r_clz))
        ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx_rev, clz_rev)
        curvatures_all = compute_local_curvature(r_clx, r_clz, window_size=5)
        track_widths_all = compute_local_track_widths(r_clx, r_clz, ordered_blue, ordered_yellow, max_width=10.0)

        track_data = {
            "r_clx": r_clx,
            "r_clz": r_clz,
            "centerline_pts": centerline_pts,
            "ordered_blue": ordered_blue,
            "ordered_yellow": ordered_yellow,
            "curvatures_all": curvatures_all,
            "track_widths_all": track_widths_all
        }

        # --- Prepare Matplotlib figure & axes (similar to animate_features) ---
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10,10), gridspec_kw={'height_ratios':[3,1]})
        fig.subplots_adjust(right=0.7)
        ax_top.set_xlim(-10, 30)
        ax_top.set_ylim(-10, 10)
        ax_top.set_aspect('equal', adjustable='box')
        ax_top.set_title("Local Frame - RealTime Visualization")
        ax_top.set_xlabel("Local X (m)")
        ax_top.set_ylabel("Local Z (m)")

        ax_heading = fig.add_axes([0.75, 0.1, 0.1, 0.19])
        ax_heading.set_title("Heading Diff (rad)")
        ax_heading.set_ylim(-1,1)
        ax_heading.set_xticks([])
        heading_bar = ax_heading.bar(0, 0, width=0.5, color='purple')

        car_point, = ax_top.plot([], [], 'ko', ms=8, label='Car')
        heading_line, = ax_top.plot([], [], 'r-', lw=2, label='Heading')

        front_scatter = ax_top.scatter([], [], c='magenta', s=25, label='Front Centerline')
        behind_scatter = ax_top.scatter([], [], c='green', s=25, label='Behind Centerline')
        centerline_line, = ax_top.plot([], [], 'k-', lw=1, label='Centerline')
        centerline_bline, = ax_top.plot([], [], 'k-', lw=1)

        # Ray angles
        yellow_angles_deg = np.arange(-20, 111, 10)
        blue_angles_deg   = np.arange(20, -111, -10)
        yellow_angles = np.deg2rad(yellow_angles_deg)
        blue_angles   = np.deg2rad(blue_angles_deg)
        yellow_ray_lines = [ax_top.plot([], [], color='yellow', linestyle='--', lw=1)[0]
                            for _ in range(len(yellow_angles))]
        blue_ray_lines = [ax_top.plot([], [], color='cyan', linestyle='--', lw=1)[0]
                          for _ in range(len(blue_angles))]

        # Bottom axes for track width & curvature
        ax_bottom.set_title("Track Width and Centerline Curvature")
        ax_bottom.set_xlabel("Local X (m)")
        ax_bottom.set_ylabel("Track Width (m)")
        ax_bottom.set_xlim(-5, 20)
        ax_bottom.set_ylim(0, 10)

        track_width_line, = ax_bottom.plot([], [], 'bo-', label='Fwd Track Width')
        track_width_line_back, = ax_bottom.plot([], [], 'go-', label='Back Track Width')
        ax_bottom.legend(loc='upper left')

        ax_curv = ax_bottom.twinx()
        curvature_line, = ax_curv.plot([], [], 'r.-', label='Fwd Curvature')
        curvature_line_back, = ax_curv.plot([], [], 'm.-', label='Back Curvature')
        ax_curv.set_ylim(-0.5, 0.5)
        ax_curv.legend(loc='upper right')
        # ----------------------------------------------------------------------

        # Iterate through each row in the CSV, process it, then update
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
            # 1) Build the sensor_data row
            sensor_data = {
                "time":     t_arr[i],
                "x_pos":    x_arr[i],
                "z_pos":    z_arr[i],
                "yaw_deg":  yaw_arr[i],
                "long_vel": vx_arr[i],
                "lat_vel":  vy_arr[i],
                "yaw_rate": yr_arr[i],
                "steering": st_arr[i],
                "throttle": th_arr[i],
                "brake":    br_arr[i],
            }

            # 2) Use process_realtime_frame for the actual feature extraction
            frame = self.process_realtime_frame(sensor_data, track_data)

            # 3) Immediately update the plot with the newly computed frame
            #    (Essentially the same block you'd do in your FuncAnimation update)
            # -----------------------------------------------------------------
            car_point.set_data([0], [0])
            heading_line.set_data([0, heading_length], [0, 0])

            # front local centerline points
            front_pts = []
            for j in range(1, 21):
                x_val = float(j)
                z_val = frame.get(f"rel_z{j}", 0.0)
                front_pts.append([x_val, z_val])
            front_scatter.set_offsets(np.array(front_pts))

            # behind local centerline
            behind_pts = []
            for j, x_val in enumerate(range(-5, 0), start=1):
                z_val = frame.get(f"b_rel_z{j}", 0.0)
                behind_pts.append([float(x_val), z_val])
            behind_scatter.set_offsets(np.array(behind_pts))

            # lines for front & behind
            cl_x_fwd = np.arange(1, 21)
            cl_z_fwd = [frame.get(f"rel_z{k}", 0.0) for k in range(1,21)]
            centerline_line.set_data(cl_x_fwd, cl_z_fwd)

            cl_x_b = np.arange(-5, 0)
            cl_z_b = [frame.get(f"b_rel_z{k}", 0.0) for k in range(1,6)]
            centerline_bline.set_data(cl_x_b, cl_z_b)

            # rays
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

            # track width & curvature (front)
            tws, curvs = [], []
            for j in range(0, 21):
                tws.append(frame.get(f"tw{j}", 0.0))
                curvs.append(frame.get(f"c{j}", 0.0))
            track_width_line.set_data(range(21), tws)
            curvature_line.set_data(range(21), curvs)

            # behind
            btws, bcurvs = [], []
            for j in range(1, 6):
                btws.append(frame.get(f"b_tw{j}", 0.0))
                bcurvs.append(frame.get(f"b_c{j}", 0.0))
            local_xs_b = list(range(-5, 0))
            track_width_line_back.set_data(local_xs_b, btws)
            curvature_line_back.set_data(local_xs_b, bcurvs)

            # heading diff bar
            dh = frame.get("head_diff", 0)
            if dh >= 0:
                heading_bar[0].set_y(0)
                heading_bar[0].set_height(dh)
            else:
                heading_bar[0].set_y(dh)
                heading_bar[0].set_height(-dh)
            # -----------------------------------------------------------------

            # 4) Force a draw and short pause so it updates in real time
            plt.draw()
            plt.pause(0.001)
        print("[NNDriverFramework] Finished frame-by-frame realtime visualization.")

    def visualize_realtime_absolute_mode(self, csv_path, json_path, heading_length=3.0):
        """
        Visualize in absolute coordinates with realtime animation:
          - Top panel: Plot cones, track edges, and centerline (from JSON) and animate the car (shifted position)
            with cast rays (transformed to absolute coordinates).
          - Bottom panel: Plot the local metrics (local centerline points, track width & curvature) as in the relative plot.
        """
        # --- Build track data from JSON ---
        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)
        clx_rev = clx[::-1]
        clz_rev = clz[::-1]
        r_clx, r_clz = resample_centerline(clx_rev, clz_rev, resolution=1.0)
        centerline_pts = list(zip(r_clx, r_clz))
        ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx_rev, clz_rev)
        curvatures_all = compute_local_curvature(r_clx, r_clz, window_size=5)
        track_widths_all = compute_local_track_widths(r_clx, r_clz, ordered_blue, ordered_yellow, max_width=10.0)
        track_data = {
             "r_clx": r_clx,
             "r_clz": r_clz,
             "centerline_pts": centerline_pts,
             "ordered_blue": ordered_blue,
             "ordered_yellow": ordered_yellow,
             "curvatures_all": curvatures_all,
             "track_widths_all": track_widths_all
        }
        
        # --- Read CSV absolute car data ---
        data = read_csv_data(csv_path)
        if data is None:
            print("Could not load CSV data.")
            return

        # --- Create a two-panel figure: top for absolute view, bottom for relative metrics ---
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12,10), gridspec_kw={'height_ratios':[3,1]})
        fig.subplots_adjust(right=0.7)
        
        # Top panel: Plot background (cones, track edges, centerline)
        if blue_cones:
            bx, bz = zip(*blue_cones)
            ax_top.scatter(bx, bz, c='blue', marker='o', label="Blue Cones")
        if yellow_cones:
            yx, yz = zip(*yellow_cones)
            ax_top.scatter(yx, yz, c='gold', marker='o', label="Yellow Cones")
        if ordered_blue:
            blue_x, blue_z = zip(*ordered_blue)
            ax_top.plot(blue_x, blue_z, 'b-', label="Blue Edge")
        if ordered_yellow:
            yellow_x, yellow_z = zip(*ordered_yellow)
            ax_top.plot(yellow_x, yellow_z, 'y-', label="Yellow Edge")
        if clx and clz:
            ax_top.plot(clx, clz, 'm--', label="Centerline")
        ax_top.set_xlabel("X (m)")
        ax_top.set_ylabel("Z (m)")
        ax_top.set_title("Absolute Scene with Cast Rays")
        ax_top.legend()

        # Prepare markers for the car and its heading in top panel.
        car_marker, = ax_top.plot([], [], 'ro', markersize=10, label="Car")
        heading_line, = ax_top.plot([], [], 'r-', lw=2, label="Heading")
        # Create line objects for the rays.
        yellow_angles = np.deg2rad(np.arange(-20, 111, 10))
        blue_angles = np.deg2rad(np.arange(20, -111, -10))
        yellow_ray_lines = [ax_top.plot([], [], color='yellow', linestyle='--', lw=1)[0] for _ in range(len(yellow_angles))]
        blue_ray_lines = [ax_top.plot([], [], color='cyan', linestyle='--', lw=1)[0] for _ in range(len(blue_angles))]
        
        # Bottom panel: Set up for relative metrics (local centerline, track width & curvature)
        ax_bottom.set_title("Local Metrics: Centerline, Track Width & Curvature")
        ax_bottom.set_xlabel("Local X (m)")
        ax_bottom.set_ylabel("Track Width (m)")
        ax_bottom.set_xlim(-5, 21)
        ax_bottom.set_ylim(0, 10)
        # We'll update these plots in the loop.

                # Create persistent line objects for front centerline and track width
        front_centerline_line, = ax_bottom.plot([], [], 'm.-', label="Front Centerline")
        f_track_line, = ax_bottom.plot([], [], 'bo-', label="Fwd Track Width")
        back_centerline_line, = ax_bottom.plot([], [], 'g.-', label="Back Centerline")
        b_track_line, = ax_bottom.plot([], [], 'go-', label="Back Track Width")
        ax_bottom.legend(loc='upper left')

# Create twin axis for curvature metrics
        ax_curv_bottom = ax_bottom.twinx()
        ax_curv_bottom.set_ylim(-0.5, 0.5)
        f_curv_line, = ax_curv_bottom.plot([], [], 'r.-', label="Fwd Curvature")
        b_curv_line, = ax_curv_bottom.plot([], [], 'm.-', label="Back Curvature")
        ax_curv_bottom.legend(loc='upper right')
        
        # --- Reset realtime state ---
        self.last_time = None
        self.last_vx = None
        self.last_vy = None

        # --- Iterate through CSV rows to animate ---
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
            # Process the frame (this applies the usual shift)
            frame = self.process_realtime_frame(sensor_data, track_data)
            
            # --- Top Panel Update: Absolute view ---
            car_x = frame["x_pos"]
            car_z = frame["z_pos"]
            heading_deg = frame["yaw_deg"]
            heading_rad = math.radians(heading_deg)
            car_marker.set_data([car_x], [car_z])
            hx = car_x + heading_length * math.cos(heading_rad)
            hz = car_z + heading_length * math.sin(heading_rad)
            heading_line.set_data([car_x, hx], [car_z, hz])
            # Update cast rays (yellow)
            for idx, angle in enumerate(yellow_angles, start=1):
                ray_dist = frame.get(f"yr{idx}", 0)
                local_x = ray_dist * math.cos(angle)
                local_y = ray_dist * math.sin(angle)
                abs_x = car_x + local_x * math.cos(heading_rad) - local_y * math.sin(heading_rad)
                abs_y = car_z + local_x * math.sin(heading_rad) + local_y * math.cos(heading_rad)
                yellow_ray_lines[idx-1].set_data([car_x, abs_x], [car_z, abs_y])
            # Update cast rays (blue)
            for idx, angle in enumerate(blue_angles, start=1):
                ray_dist = frame.get(f"br{idx}", 0)
                local_x = ray_dist * math.cos(angle)
                local_y = ray_dist * math.sin(angle)
                abs_x = car_x + local_x * math.cos(heading_rad) - local_y * math.sin(heading_rad)
                abs_y = car_z + local_x * math.sin(heading_rad) + local_y * math.cos(heading_rad)
                blue_ray_lines[idx-1].set_data([car_x, abs_x], [car_z, abs_y])
            
            # --- Bottom Panel Update: Relative metrics ---
            # Clear the bottom panel for each frame
            # Compute forward relative metrics
            front_local_vals = [frame.get(f"rel_z{j}", 0.0) for j in range(1, 21)]
            f_tw = [frame.get(f"tw{j}", 0.0) for j in range(21)]
            f_curv = [frame.get(f"c{j}", 0.0) for j in range(21)]

# Compute behind relative metrics
            back_local_vals = [frame.get(f"b_rel_z{j}", 0.0) for j in range(1, 6)]
            b_tw = [frame.get(f"b_tw{j}", 0.0) for j in range(1, 6)]
            b_curv = [frame.get(f"b_c{j}", 0.0) for j in range(1, 6)]

# Update the persistent line objects:
            f_track_line.set_data(np.arange(21), f_tw)
            f_curv_line.set_data(np.arange(21), f_curv)

            b_track_line.set_data(np.array(list(range(-5,0))), b_tw)
            b_curv_line.set_data(np.array(list(range(-5,0))), b_curv)
            
            plt.pause(0.01)
            plt.draw()
        print("[NNDriverFramework] Finished realtime absolute visualization.")
        plt.show()
        # ---------------------------------------------------------------------
# ------------------ MAIN ENTRY POINT ---------------------------------
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        help="train, infer, realtime, or visualize-realtime")
    parser.add_argument("--csv", type=str,
                        help="Path to CSV file (for train, infer, visual, or visualize-realtime or visualize-absolute)")
    parser.add_argument("--json", type=str, default="default.json",
                        help="Path to track JSON")
    parser.add_argument("--transformer", type=str, default="transformer.joblib",
                        help="Path to save/load the fitted scaler/PCA")
    parser.add_argument("--model", type=str, default="nn_model.joblib",
                        help="Path to save/load the trained model")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional path to save postprocessed CSV in train mode")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="TCP server host")
    parser.add_argument("--port", type=int, default=65432,
                        help="TCP server port")
    args = parser.parse_args()

    framework = NNDriverFramework(transformer_path=args.transformer,
                                  model_path=args.model)

    mode = args.mode.lower()
    if mode == "train":
        if not args.csv:
            print("Must provide --csv for training.")
            return
        framework.train_mode(csv_path=args.csv,
                             json_path=args.json,
                             output_csv_path=args.output_csv)
    elif mode == "infer":
        if not args.csv:
            print("Must provide --csv for inference.")
            return
        framework.inference_mode(csv_path=args.csv, json_path=args.json)
    elif mode == "realtime":
        framework.realtime_mode(track_json=args.json, host=args.host, port=args.port)
    elif mode == "visualize-realtime":
        if not args.csv:
            print("Must provide --csv for 'visualize-realtime'.")
            return
        framework.visual_realtime_mode(csv_path=args.csv, json_path=args.json)
    elif mode == "visualize-absolute":
        framework.visualize_absolute_mode(csv_path=args.csv, json_path=args.json)
    elif mode == 'visualize-realtime-absolute':
        framework.visualize_realtime_absolute_mode(csv_path=args.csv, json_path=args.json)
    else:
        print("Unknown mode. Use --mode train, infer, realtime, or visualize-realtime.")

if __name__ == "__main__":
    main()
