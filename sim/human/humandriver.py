import socket
import time
import hid
import json
import math
import csv
import numpy as np
from scipy.interpolate import splprep, splev

# HID Device Configuration
VENDOR_ID = 0x044f
PRODUCT_ID = 0xb66d

# Socket Configuration
host = '127.0.0.1'
port = 65432

# Number of Cones to Log
NUM_BLUE = 10
NUM_YELLOW = 10

# Number of Track Edge Points to Sample
NUM_TRACK_EDGE_POINTS = 10

def compute_distance_to_track_edge(point, edge_type, cones):
    """
    Computes the shortest distance from a given point to the specified track edge represented by cones.
    """
    if edge_type not in ["left", "right"]:
        raise ValueError(f"Invalid edge_type: {edge_type}. Must be 'left' or 'right'.")

    edge_cones = cones.get(edge_type, [])
    if not edge_cones:
        raise ValueError(f"No cones found for edge_type: {edge_type}.")

    x, z = point
    min_distance = float('inf')
    for cone_x, cone_z in edge_cones:
        distance = math.sqrt((cone_x - x)**2 + (cone_z - z)**2)
        if distance < min_distance:
            min_distance = distance
    return min_distance

def fit_spline_to_cones(cones):
    """
    Fits a spline to a list of (x, z) cone positions.
    """
    if len(cones) < 3:
        return None
    x, z = zip(*cones)
    tck, _ = splprep([x, z], s=0)
    def spline_fn(t):
        return splev(t, tck)
    return spline_fn

def parse_cone_data(json_file_path):
    """
    Parses the JSON file containing cone and centerline data.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    x_values = data.get("x", [])
    y_values = data.get("y", [])
    colors = data.get("color", [])
    centerline_x = data.get("centerline_x", [])
    centerline_z = data.get("centerline_y", [])  # Treated as z-coordinates

    if not (len(x_values) == len(y_values) == len(colors)):
        raise ValueError("JSON file data lengths for 'x', 'z', and 'color' must be equal.")

    # Interpret y_values as z coordinates
    blue_cones = [(x, z) for x, z, c in zip(x_values, y_values, colors) if c.lower() == "blue"]
    yellow_cones = [(x, z) for x, z, c in zip(x_values, y_values, colors) if c.lower() == "yellow"]

    return blue_cones, yellow_cones, centerline_x, centerline_z

# Parse initial cone and centerline data
blue_cones, yellow_cones, centerline_x_vals, centerline_z_vals = parse_cone_data("../tracks/cones.json")

def decode_angle(data, calibration_offset, scaling):
    """
    Decodes raw angle data from the HID device.
    """
    angle = int.from_bytes(data, byteorder='little', signed=True)
    calibrated_angle = angle - calibration_offset
    scaled_angle = calibrated_angle / scaling
    return scaled_angle

def calibrate_angle(device):
    """
    Calibrates initial angles for steering, brake, and accelerator.
    """
    data = device.read(64)
    if data:
        initial_steering = data[42:45]
        initial_steering_angle = int.from_bytes(initial_steering, byteorder='little', signed=True)

        initial_brake = data[48:51]
        initial_brake_angle = int.from_bytes(initial_brake, byteorder='little', signed=True)

        initial_accelerator = data[45:48]
        initial_accelerator_angle = int.from_bytes(initial_accelerator, byteorder='little', signed=True)
        return initial_steering_angle, initial_brake_angle, initial_accelerator_angle
    return 0, 0, 0

def rotate_point(x, z, angle_deg):
    """
    Rotates a point (x, z) by a given angle in degrees.
    """
    rad = math.radians(angle_deg)
    cos_angle = math.cos(rad)
    sin_angle = math.sin(rad)
    x_rot = cos_angle * x + sin_angle * z
    z_rot = -sin_angle * x + cos_angle * z
    return x_rot, z_rot

def transform_cones(cones, car_position, yaw):
    """
    Transforms cone positions to be relative to the vehicle's position and orientation.
    """
    transformed = []
    car_x, car_z = car_position
    for (cx, cz) in cones:
        dx = cx - car_x
        dz = cz - car_z
        x_rel, z_rel = rotate_point(dx, dz, -yaw)
        transformed.append((x_rel, z_rel))
    return transformed

def find_next_points(car_pos, heading_angle, x_values, z_values, num_next=50, fov_angle=360, max_ahead=20, max_behind=5):
    """
    Finds the next centerline points relative to the vehicle's position and heading.
    """
    heading_angle = (heading_angle + 360) % 360
    heading_rad = math.radians(heading_angle)
    heading_vector = (math.cos(heading_rad), math.sin(heading_rad))

    distances_and_points = []

    for (x, z) in zip(x_values, z_values):
        dx = x - car_pos[0]
        dz = z - car_pos[1]
        distance = math.sqrt(dx*dx + dz*dz)
        if distance == 0:
            continue
        point_vector = (dx/distance, dz/distance)
        alignment = point_vector[0]*heading_vector[0] + point_vector[1]*heading_vector[1]
        if -1 <= alignment <= 1:
            angle_to_point = math.degrees(math.acos(alignment))
        else:
            angle_to_point = 180

        # Determine if the point is within the specified range
        if alignment > 0 and distance <= max_ahead:
            # In front
            if -fov_angle/2 <= angle_to_point <= fov_angle/2:
                distances_and_points.append(((x, z), distance))
        elif alignment < 0 and distance <= max_behind:
            # Behind
            distances_and_points.append(((x, z), distance))

    distances_and_points.sort(key=lambda x: x[1])
    selected_points = [p[0] for p in distances_and_points[:num_next]]
    return selected_points

def compute_curvature(points):
    """
    Computes the curvature at each point based on three consecutive points.
    """
    n = len(points)
    if n < 3:
        return [0.0] * n
    curvatures = []
    for i in range(1, n - 1):
        x1, z1 = points[i - 1]
        x2, z2 = points[i]
        x3, z3 = points[i + 1]
        a = math.dist((x2, z2), (x3, z3))
        b = math.dist((x1, z1), (x3, z3))
        c = math.dist((x1, z1), (x2, z2))

        area = abs(x1*(z2 - z3) + x2*(z3 - z1) + x3*(z1 - z2)) / 2.0
        if area == 0:
            curvature = 0.0
        else:
            radius = (a*b*c) / (4.0 * area)
            curvature = 1.0 / radius if radius != 0 else 0.0

        vector1 = (x2 - x1, z2 - z1)
        vector2 = (x3 - x2, z3 - z2)
        cross_product = vector1[0]*vector2[1] - vector1[1]*vector2[0]
        signed_curvature = curvature if cross_product > 0 else -curvature
        curvatures.append(signed_curvature)

    # Pad to have one curvature per point
    curvatures = [0.0] + curvatures + [0.0]
    return curvatures

def compute_signed_distance_to_centerline(car_pos, centerline_points):
    """
    Computes the signed distance from the vehicle to the closest point on the centerline.
    """
    if not centerline_points:
        return float('nan'), -1
    cx, cz = car_pos
    min_dist = float('inf')
    closest_idx = -1
    for i, (px, pz) in enumerate(centerline_points):
        dist = math.sqrt((px - cx)**2 + (pz - cz)**2)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    if min_dist == float('inf'):
        return float('nan'), -1
    if len(centerline_points) < 2:
        return min_dist, closest_idx

    # Determine sign using local direction of centerline
    if closest_idx == len(centerline_points)-1 and closest_idx > 0:
        p1 = centerline_points[closest_idx-1]
        p2 = centerline_points[closest_idx]
    else:
        p1 = centerline_points[closest_idx]
        p2 = centerline_points[min(closest_idx+1, len(centerline_points)-1)]

    dx = p2[0] - p1[0]
    dz = p2[1] - p1[1]

    tx = cx - p1[0]
    tz = cz - p1[1]

    cross = dx*tz - dz*tx
    signed_distance = min_dist if cross >= 0 else -min_dist
    return signed_distance, closest_idx

def compute_relative_heading_angle(yaw, centerline_points):
    """
    Computes the relative heading angle between the vehicle's orientation and the centerline's direction.
    """
    if len(centerline_points) < 2:
        return 0.0
    (x1, z1) = centerline_points[0]
    (x2, z2) = centerline_points[1]
    centerline_angle = math.degrees(math.atan2(z2 - z1, x2 - x1))
    relative_heading = yaw - centerline_angle
    relative_heading = (relative_heading + 180) % 360 - 180
    return relative_heading

def cast_rays(vehicle_position, yaw, cones, max_distance=20, num_spline_points=100):
    """
    Casts rays from the vehicle's position at specified angles to measure distances to track edges.
    """
    spline = fit_spline_to_cones(cones)
    if spline:
        t_vals = np.linspace(0, 1, num_spline_points)
        edge_points = list(zip(*spline(t_vals)))
    else:
        edge_points = cones

    def ray_distance(ray_angle):
        absolute_angle = math.radians(yaw + ray_angle)
        vehicle_x, vehicle_z = vehicle_position
        min_d = max_distance
        for ex, ez in edge_points:
            dx = ex - vehicle_x
            dz = ez - vehicle_z
            projection = dx*math.cos(absolute_angle) + dz*math.sin(absolute_angle)
            if projection < 0 or projection > max_distance:
                continue
            perp_dist = abs(-dx*math.sin(absolute_angle) + dz*math.cos(absolute_angle))
            if perp_dist < 0.5:
                dist_to_edge = math.sqrt(dx*dx + dz*dz)
                if dist_to_edge < min_d:
                    min_d = dist_to_edge
        return min_d

    return ray_distance

def pad_cones(cones, desired_count=10):
    """
    Pads the list of cones with placeholder values if there are fewer than desired.
    """
    if len(cones) < desired_count:
        cones = cones + [(20.0, 20.0)]*(desired_count - len(cones))
    elif len(cones) > desired_count:
        cones = cones[:desired_count]
    return cones

# Ray angle sets:
left_ray_angles = list(range(-20, 111, 10))   # -20° to 110° by 10°
right_ray_angles = list(range(-110, 21, 10))  # -110° to 20° by 10°

# Initialize server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(5)
print(f"Server listening on {host}:{port}")

# Initialize HID device
device = hid.Device(VENDOR_ID, PRODUCT_ID)
print(f"Connected to {device.manufacturer} - {device.product}")

# Calibrate initial angles
calibration_offset_steering, calibration_offset_brake, calibration_offset_accelerator = calibrate_angle(device)

# Accept client connection
client_socket, addr = server_socket.accept()
print(f"Connection from {addr}")

# Build CSV header with all required fields
header = [
    "time",
    # Vehicle State:
    "long_vel", "lat_vel", "yaw_rate", "long_acc", "lat_acc",
    # Relative Position:
    "dist_centerline", "rel_heading_angle",
]

# Centerline Points (10 x,z pairs)
for i in range(1, 11):
    header.append(f"centerline_x{i}")
    header.append(f"centerline_z{i}")

# Track Widths
header.append("left_track_width")
header.append("right_track_width")

# Curvatures (10)
for i in range(1, 11):
    header.append(f"curvature_{i}")

# Ray Distances - Left and Right
for angle in left_ray_angles:
    header.append(f"left_ray_{angle}deg")
for angle in right_ray_angles:
    header.append(f"right_ray_{angle}deg")

# Track Edge Points (Left and Right, 10 x,z pairs each)
for i in range(1, NUM_TRACK_EDGE_POINTS + 1):
    header.append(f"left_edge_x{i}")
    header.append(f"left_edge_z{i}")
for i in range(1, NUM_TRACK_EDGE_POINTS + 1):
    header.append(f"right_edge_x{i}")
    header.append(f"right_edge_z{i}")

# Initialize CSV file and write header
csv_file = open("../../runs/session2/car_data_run_8.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(header)

try:
    while True:
        # Read data from HID device
        data = device.read(64)
        if data:
            # Decode steering, brake, and throttle angles
            steering_angle = decode_angle(data[42:45], calibration_offset_steering, 20000)
            if steering_angle < -600:
                steering_angle += 838.8
            if steering_angle > 90:
                steering_angle = 90
            if steering_angle < -90:
                steering_angle = -90
            steering_angle = steering_angle / 90

            brake_angle = decode_angle(data[48:51], calibration_offset_brake, 10)
            brake = -brake_angle / 25.5

            accelerator_angle = decode_angle(data[45:48], calibration_offset_accelerator, 100)
            if accelerator_angle > 0:
                accelerator_angle -= 327.5 * 2
            throttle = -accelerator_angle / 655

        # Prepare and send control inputs to the client
        message = f"{steering_angle},{throttle},{brake}\n"
        client_socket.sendall(message.encode())

        # Receive vehicle state data from the client
        raw_data = client_socket.recv(1024).decode('utf-8').strip()
        try:
            fields = raw_data.split(',')
            if len(fields) < 6:
                raise ValueError(f"Incomplete data received: {fields}")

            # Extract vehicle state (x_pos, z_pos, yaw, long_vel, lat_vel, yaw_rate)
            x_pos = float(fields[0])
            z_pos = float(fields[1])
            yaw = -float(fields[2]) + 90
            long_vel = float(fields[3])
            lat_vel = float(fields[4])  # lateral velocity
            yaw_rate = float(fields[5])

            # Placeholder for accelerations (to be replaced if available)
            long_acc = 0.0
            lat_acc = 0.0

            # Compute vehicle's relative position and orientation
            car_pos = (x_pos, z_pos)

            # Find next centerline points relative to the vehicle
            centerline_points = find_next_points(
                car_pos, yaw, centerline_x_vals, centerline_z_vals,
                num_next=10, fov_angle=360, max_ahead=20, max_behind=5
            )
            rel_centerline_points = []
            for (cx, cz) in centerline_points:
                dx = cx - x_pos
                dz = cz - z_pos
                rx, rz = rotate_point(dx, dz, -yaw)
                rel_centerline_points.append((rx, rz))

            # Pad centerline points if fewer than 10
            while len(rel_centerline_points) < 10:
                rel_centerline_points.append((0.0, 0.0))
            rel_centerline_points = rel_centerline_points[:10]

            # Compute curvature based on absolute centerline points
            curvatures = compute_curvature(centerline_points)
            # Pad curvatures if fewer than 10
            while len(curvatures) < 10:
                curvatures.append(0.0)
            curvatures = curvatures[:10]

            # Compute distance to centerline and relative heading angle
            dist_centerline, closest_idx = compute_signed_distance_to_centerline(car_pos, centerline_points)
            rel_heading_angle = compute_relative_heading_angle(yaw, centerline_points)

            # Compute track widths at the closest centerline point
            if 0 <= closest_idx < len(centerline_points):
                cpx, cpz = centerline_points[closest_idx]
            else:
                cpx, cpz = (x_pos, z_pos)

            # Compute distances to left and right track edges
            cones = {"left": blue_cones, "right": yellow_cones}
            left_track_width = compute_distance_to_track_edge((cpx, cpz), "left", cones)
            right_track_width = compute_distance_to_track_edge((cpx, cpz), "right", cones)

            # Cast rays to left and right track edges
            left_cast = cast_rays((x_pos, z_pos), yaw, blue_cones)
            right_cast = cast_rays((x_pos, z_pos), yaw, yellow_cones)

            left_ray_distances = [left_cast(angle) for angle in left_ray_angles]
            right_ray_distances = [right_cast(angle) for angle in right_ray_angles]

            # Compute track edges by fitting splines and sampling points
            left_spline = fit_spline_to_cones(blue_cones)
            right_spline = fit_spline_to_cones(yellow_cones)

            left_edge_points = []
            if left_spline:
                t_vals = np.linspace(0, 1, NUM_TRACK_EDGE_POINTS)
                left_edge_sampled = list(zip(*left_spline(t_vals)))
                left_edge_points = list(zip(left_edge_sampled[0], left_edge_sampled[1]))
            else:
                left_edge_points = blue_cones[:NUM_TRACK_EDGE_POINTS]
            left_edge_points = transform_cones(left_edge_points, car_pos, yaw)
            left_edge_points = pad_cones(left_edge_points, NUM_TRACK_EDGE_POINTS)

            right_edge_points = []
            if right_spline:
                t_vals = np.linspace(0, 1, NUM_TRACK_EDGE_POINTS)
                right_edge_sampled = list(zip(*right_spline(t_vals)))
                right_edge_points = list(zip(*right_spline(t_vals)))
                right_edge_points = list(zip(right_edge_sampled[0], right_edge_sampled[1]))
            else:
                right_edge_points = yellow_cones[:NUM_TRACK_EDGE_POINTS]
            right_edge_points = transform_cones(right_edge_points, car_pos, yaw)
            right_edge_points = pad_cones(right_edge_points, NUM_TRACK_EDGE_POINTS)

            # Build CSV row with all required fields
            row = [
                time.time(),
                long_vel, lat_vel, yaw_rate, long_acc, lat_acc,
                dist_centerline, rel_heading_angle
            ]
            for (rx, rz) in rel_centerline_points:
                row.append(rx)
                row.append(rz)
            row.append(left_track_width)
            row.append(right_track_width)
            row.extend(curvatures)
            row.extend(left_ray_distances)
            row.extend(right_ray_distances)
            for (lx, lz) in left_edge_points:
                row.append(lx)
                row.append(lz)
            for (rx, rz) in right_edge_points:
                row.append(rx)
                row.append(rz)

            # Write row to CSV
            csv_writer.writerow(row)
            csv_file.flush()

        except ValueError as ve:
            print(f"Data parsing error: {ve}. Raw data: '{raw_data}'")
        except Exception as e:
            print(f"Unexpected error: {e}")

        time.sleep(0.01)

except Exception as e:
    print(f"Error occurred: {e}")

finally:
    client_socket.close()
    device.close()
    server_socket.close()
    csv_file.close()
    print("Server and connection closed.")
