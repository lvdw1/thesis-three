import socket
import time
import csv
import math
import json
import torch
import numpy as np
import torch.nn as nn

########################
# Neural Network Model Definition
########################

# Updated input size:
# 10 blue cones * 2 coords + 10 yellow cones * 2 coords = 20 cones *2 = 40
# + 3 velocities (long_vel, slide_vel, yaw_rate) = 43
# + 10 curvature values = 53
# + 1 distance to centerline = 54
INPUT_SIZE = 54

# Load normalization parameters
means = np.load("input_means.npy")
stds = np.load("input_stds.npy")

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
        # out has 3 values: [steering, throttle, brake]
        # Apply bounding:
        # steering: tanh to get [-1, 1]
        steering = torch.tanh(out[:, 0:1])
        # throttle: sigmoid to get [0, 1]
        throttle = torch.sigmoid(out[:, 1:2])
        # brake: sigmoid to get [0, 1]
        brake = torch.sigmoid(out[:, 2:3])

        return torch.cat([steering, throttle, brake], dim=1)

########################
# Load the trained model
########################

print("Loading model...")
model = SimpleNN()
model.load_state_dict(torch.load("car_controller_model.pth", map_location='cpu'))
model.eval()  # Set model to evaluation mode
print("Model loaded successfully.")

########################
# Helper Functions
########################

def rotate_point(x, z, angle_deg):
    rad = math.radians(angle_deg)
    cos_angle = math.cos(rad)
    sin_angle = math.sin(rad)
    x_rot = cos_angle * x + sin_angle * z
    z_rot = -sin_angle * x + cos_angle * z
    return x_rot, z_rot

def transform_cones(cones, car_position, yaw):
    transformed = []
    car_x, car_z = car_position
    for (cx, cz) in cones:
        dx = cx - car_x
        dz = cz - car_z
        x_rel, z_rel = rotate_point(dx, dz, -yaw)
        transformed.append((x_rel, z_rel))
    return transformed

def pad_cones(cones, desired_count=10):
    """
    Pad cones list to have exactly desired_count entries.
    Use (20.0,20.0) as the placeholder, as done in training.
    """
    if len(cones) < desired_count:
        cones = cones + [(20.0,20.0)]*(desired_count - len(cones))
    elif len(cones) > desired_count:
        cones = cones[:desired_count]
    return cones
NUM_BLUE = 10
NUM_YELLOW = 10
def parse_cone_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    x_values = data.get("x", [])
    y_values = data.get("y", [])
    colors = data.get("color", [])
    centerline_x = data.get("centerline_x", [])
    centerline_y = data.get("centerline_y", [])

    if not (len(x_values) == len(y_values) == len(colors)):
        raise ValueError("JSON file data lengths for 'x', 'y', and 'color' must be equal.")

    blue_cones = [(x, y) for x, y, color in zip(x_values, y_values, colors) if color.lower() == "blue"]
    yellow_cones = [(x, y) for x, y, color in zip(x_values, y_values, colors) if color.lower() == "yellow"]

    return blue_cones, yellow_cones, centerline_x, centerline_y

blue_cones, yellow_cones, centerline_x, centerline_y = parse_cone_data("cones.json")

def find_next_cones(car_pos, heading_angle, cones, cone_colors, num_next=10, fov_angle=150):
    heading_angle = (heading_angle + 360) % 360
    heading_rad = math.radians(heading_angle)
    heading_vector = (math.cos(heading_rad), math.sin(heading_rad))

    def cone_distance_and_angle(cone):
        cone_vector = (cone[0] - car_pos[0], cone[1] - car_pos[1])
        distance = math.sqrt(cone_vector[0] ** 2 + cone_vector[1] ** 2)
        if distance > 0:
            norm_cone_vector = (cone_vector[0] / distance, cone_vector[1] / distance)
        else:
            norm_cone_vector = (0, 0)
        alignment = norm_cone_vector[0]*heading_vector[0] + norm_cone_vector[1]*heading_vector[1]
        alignment = max(min(alignment, 1), -1)
        angle_to_cone = math.degrees(math.acos(alignment))
        return distance, alignment, angle_to_cone

    blue_cones_list = []
    yellow_cones_list = []

    for cone, color in zip(cones, cone_colors):
        distance, alignment, angle_to_cone = cone_distance_and_angle(cone)
        if alignment > 0 and -fov_angle / 2 <= angle_to_cone <= fov_angle / 2:
            if color.lower() == 'blue':
                blue_cones_list.append((cone, distance))
            elif color.lower() == 'yellow':
                yellow_cones_list.append((cone, distance))

    blue_cones_list.sort(key=lambda x: x[1])
    yellow_cones_list.sort(key=lambda x: x[1])

    return [c[0] for c in blue_cones_list[:num_next]], [c[0] for c in yellow_cones_list[:num_next]]

def find_next_points(car_pos, heading_angle, x_values, y_values, num_next=50, fov_angle=150):
    heading_angle = (heading_angle + 360) % 360
    heading_rad = math.radians(heading_angle)
    heading_vector = (math.cos(heading_rad), math.sin(heading_rad))

    distances_and_points = []

    for (x, y) in zip(x_values, y_values):
        dx = x - car_pos[0]
        dy = y - car_pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        if distance == 0:
            continue
        point_vector = (dx/distance, dy/distance)
        alignment = point_vector[0]*heading_vector[0] + point_vector[1]*heading_vector[1]
        alignment = max(min(alignment, 1), -1)
        angle_to_point = math.degrees(math.acos(alignment))

        if alignment > 0 and -fov_angle/2 <= angle_to_point <= fov_angle/2:
            distances_and_points.append(((x, y), distance))

    distances_and_points.sort(key=lambda x: x[1])
    selected_points = [p[0] for p in distances_and_points[:num_next]]
    return selected_points

def compute_curvature(points):
    n = len(points)
    if n < 3:
        return [0.0]*10

    curvatures_all = []
    for i in range(1, n-1):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        x3, y3 = points[i+1]

        a = math.dist((x2, y2), (x3, y3))
        b = math.dist((x1, y1), (x3, y3))
        c = math.dist((x1, y1), (x2, y2))

        area = abs(x1*(y2 - y3)+x2*(y3 - y1)+x3*(y1 - y2))/2.0

        if area == 0:
            curvature = 0.0
        else:
            radius = (a*b*c)/(4.0*area) if area != 0 else float('inf')
            curvature = 1.0/radius if radius != 0 else 0.0

        curvatures_all.append(curvature)

    c = len(curvatures_all)
    if c >= 48:
        indices = [0,5,10,15,20,25,30,35,40,45]
        return [curvatures_all[i] for i in indices]
    else:
        if c == 0:
            return [0.0]*10
        elif c < 10:
            return curvatures_all + [0.0]*(10-c)
        else:
            step = c//10
            if step == 0:
                return (curvatures_all[:10] + [0.0]*10)[:10]
            indices = [min(i*step, c-1) for i in range(10)]
            return [curvatures_all[i] for i in indices]

def compute_distance_to_centerline(car_pos, centerline_points):
    if not centerline_points:
        return float('nan')
    cx, cz = car_pos
    min_dist = float('inf')
    for (px, pz) in centerline_points:
        dist = math.sqrt((px - cx)**2 + (pz - cz)**2)
        if dist < min_dist:
            min_dist = dist
    return min_dist

########################
# Server Setup
########################

host = '127.0.0.1'
port = 65432
print(f"Setting up server on {host}:{port}...")
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(5)
print("Server listening... Waiting for connection.")

client_socket, addr = server_socket.accept()
print(f"Connection from {addr}")

# Build final CSV header: original columns plus the final numeric features
header = ["Time", "X Position", "Z Position", "Yaw", "Longitudinal Velocity", "Sliding Velocity", "Yaw Rate",
          "Steering Angle", "Throttle", "Brake"]

# Add cone columns
for i in range(NUM_BLUE):
    header.append(f"Bx{i+1}")
    header.append(f"Bz{i+1}")
for i in range(NUM_YELLOW):
    header.append(f"Yx{i+1}")
    header.append(f"Yz{i+1}")

# Curvature columns
for i in range(10):
    header.append(f"C{i+1}")

# Distance to centerline
header.append("DistCenterline")

csv_file = open("car_data_run_nn.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(header)

try:
    while True:
        # Attempt to receive car state from Unity
        raw_data = client_socket.recv(4096).decode('utf-8').strip()
        if not raw_data:
            time.sleep(0.01)
            continue

        fields = raw_data.split(',')
        if len(fields) < 6:
            time.sleep(0.01)
            continue

        # Parse the car state EXACTLY like in humandriver
        try:
            x_pos = float(fields[0])
            z_pos = float(fields[1])
            yaw = -float(fields[2]) + 90
            long_vel = float(fields[3])
            slide_vel = float(fields[4])
            yaw_rate = float(fields[5])
        except Exception as e:
            print(f"Error parsing car state: {e}")
            continue

        # Apply the same translation as humandriver
        right_x_offset = -1.8 * math.cos(math.radians(yaw + 90))
        right_z_offset = -1.8 * math.sin(math.radians(yaw + 90))
        translated_x_pos = x_pos + right_x_offset
        translated_z_pos = z_pos + right_z_offset

        all_cones = blue_cones + yellow_cones
        all_colors = ['blue'] * len(blue_cones) + ['yellow'] * len(yellow_cones)

        # Find cones
        next_blue_cones, next_yellow_cones = find_next_cones((translated_x_pos, translated_z_pos), yaw,
                                                             all_cones, all_colors,
                                                             num_next=NUM_BLUE, fov_angle=150)

        # Transform and pad cones
        # We must transform them relative to car
        rel_blue_cones = next_blue_cones  # Already chosen 10
        rel_yellow_cones = next_yellow_cones
        # Actually transform them if needed:
        rel_blue_cones = pad_cones(rel_blue_cones, NUM_BLUE)
        rel_yellow_cones = pad_cones(rel_yellow_cones, NUM_YELLOW)

        # transform
        rel_blue_cones = transform_cones(rel_blue_cones, (translated_x_pos, translated_z_pos), yaw)
        rel_yellow_cones = transform_cones(rel_yellow_cones, (translated_x_pos, translated_z_pos), yaw)

        # Find up to 50 centerline points
        up_to_50_points = find_next_points((translated_x_pos, translated_z_pos), yaw, centerline_x, centerline_y, num_next=50, fov_angle=150)

        # Select 10 equidistant points
        num_selected_points = 10
        if len(up_to_50_points) > num_selected_points:
            indices = np.linspace(0, len(up_to_50_points)-1, num_selected_points).astype(int)
            selected_points = [up_to_50_points[i] for i in indices]
        else:
            selected_points = up_to_50_points

        # Compute curvature of these 10 points
        centerline_curvatures = compute_curvature(selected_points)

        # Flatten 10 centerline points (or fewer) for CSV
        centerline_flat = []
        for (cx, cy) in selected_points:
            centerline_flat.extend([cx, cy])

        # If fewer than 10 points found, pad with NaN
        while len(centerline_flat) < 20:
            centerline_flat.extend([float('nan'), float('nan')])

        # Compute distance to centerline
        distance_to_centerline = compute_distance_to_centerline((translated_x_pos, translated_z_pos), selected_points)

        # Build input vector
        input_vector = []
        for (bx, bz) in rel_blue_cones:
            input_vector.extend([bx, bz])
        for (yx, yz) in rel_yellow_cones:
            input_vector.extend([yx, yz])
        input_vector.extend([long_vel, slide_vel, yaw_rate])
        input_vector.extend(centerline_curvatures)
        input_vector.append(distance_to_centerline)

        # Normalize using the same means and stds as training
        input_array = np.array(input_vector, dtype=np.float32).reshape(1, -1)
        input_array = (input_array - means) / stds
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        # NN prediction
        with torch.no_grad():
            output = model(input_tensor)
        steering_pred, throttle_pred, brake_pred = output[0].tolist()

        # Send predicted controls back to Unity
        if brake_pred < 0.2:
            brake_pred = 0

        message = f"{steering_pred},{throttle_pred},{brake_pred}\n"
        print(message)
        client_socket.sendall(message.encode())

        cone_values = []
        for (bx, bz) in rel_blue_cones:
            cone_values.append(bx)
            cone_values.append(bz)
        for (yx, yz) in rel_yellow_cones:
            cone_values.append(yx)
            cone_values.append(yz)

        row = [time.time(), translated_x_pos, translated_z_pos, yaw, long_vel, slide_vel, yaw_rate,
               steering_pred, throttle_pred, brake_pred]
        row.extend(cone_values)  # Cone positions as a flat list
        row.extend(centerline_curvatures)  # Curvature values as a flat list
        row.append(distance_to_centerline)  # Distance to centerline
        csv_writer.writerow(row)

        time.sleep(0.01)

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    client_socket.close()
    server_socket.close()
    csv_file.close()
    print("Server and connection closed.")
