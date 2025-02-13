import os
import json
import logging
import math
import csv
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


def read_csv_data(file_path):
    """
    Reads CSV data and organizes it into structured numpy arrays.
    """
    if not os.path.exists(file_path):
        logging.error(f"CSV file not found at path: {file_path}")
        return None

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Initialize lists to store data
        times = []
        x_pos = []
        z_pos = []
        yaw_angle = []
        long_vel = []
        lat_vel = []
        yaw_rate = []
        long_acc = []
        lat_acc = []

        for row in reader:
            try:
                # Parse basic scalar values
                times.append(float(row["time"]))
                x_pos.append(float(row["x_pos"]))
                z_pos.append(float(row["z_pos"]))
                yaw_angle.append(float(row["yaw_angle"]))
                long_vel.append(float(row["long_vel"]))
                lat_vel.append(float(row["lat_vel"]))
                yaw_rate.append(float(row["yaw_rate"]))
                long_acc.append(float(row["long_acc"]))
                lat_acc.append(float(row["lat_acc"]))

            except Exception as e:
                logging.warning(f"Error parsing row: {row} - {e}")

        # Check if data was loaded
        if not times:
            logging.error("No data loaded from CSV.")
            return None

        # Convert lists to numpy arrays for efficiency
        return {
            "time": np.array(times),
            "x_pos": np.array(x_pos),
            "z_pos": np.array(z_pos),
            "yaw_angle": np.array(yaw_angle),
            "long_vel": np.array(long_vel),
            "lat_vel": np.array(lat_vel),
            "yaw_rate": np.array(yaw_rate),
            "long_acc": np.array(long_acc),
            "lat_acc": np.array(lat_acc),
                    }

def find_centerpoints_in_front_and_behind(car_x, car_z, car_yaw, centerpoints,
                                          n_front=20, n_behind=5, fov_deg=180):
    """
    Classifies centerpoints into 'front' or 'behind' relative to the car's heading
    within a given Field Of View (FOV). Then picks up to n_front front points and
    n_behind behind points (closest by distance) and transforms them into the car's
    local coordinate frame:
    
       - local x-axis: forward in the direction of car_yaw
       - local z-axis: to the car's right
    
    Parameters
    ----------
    car_x, car_z : float
        Car's position in global coordinates.
    car_yaw : float
        Car's heading in radians (0 rad means facing +X globally).
    centerpoints : list of (X, Z) or np.array of shape (N, 2)
        Global coordinates of centerpoints to classify and transform.
    n_front : int
        Maximum number of closest front centerpoints to keep.
    n_behind : int
        Maximum number of closest behind centerpoints to keep.
    fov_deg : float
        Total field of view in degrees for “front” classification. 180 means ±90°.
    
    Returns
    -------
    front_local : list of (x_local, z_local)
        Up to n_front centerpoints in local coords, classified as front, sorted by distance.
    behind_local : list of (x_local, z_local)
        Up to n_behind centerpoints in local coords, classified as behind, sorted by distance.
    """
    half_fov_rad = math.radians(fov_deg / 2.0)

    # We'll hold points along with their distance for sorting
    front_points = []
    behind_points = []

    for (cx, cz) in centerpoints:
        # Vector from car to centerpoint
        dx = cx - car_x
        dz = cz - car_z

        # Distance for sorting
        dist = math.hypot(dx, dz)

        # Angle of centerpoint relative to the car's heading
        # The global angle of centerpoint is atan2(dz, dx), then we subtract car_yaw
        angle = math.atan2(dz, dx) - car_yaw
        
        # Normalize angle to [-pi, +pi]
        angle = math.atan2(math.sin(angle), math.cos(angle))

        # If the absolute angle is within half_fov_rad => "front"
        # else => "behind"
        if abs(angle) <= half_fov_rad:
            front_points.append((cx, cz, dist))
        else:
            behind_points.append((cx, cz, dist))

    # Sort by distance ascending, then pick up to n_front or n_behind
    front_points.sort(key=lambda p: p[2])
    front_points = front_points[:n_front]

    behind_points.sort(key=lambda p: p[2])
    behind_points = behind_points[:n_behind]

    # Transform each point into the car's local coordinate frame
    # local x = dx*cos(yaw) + dz*sin(yaw)
    # local z = -dx*sin(yaw) + dz*cos(yaw)
    front_local = []
    for (cx, cz, dist) in front_points:
        dx = cx - car_x
        dz = cz - car_z
        x_local = dx*math.cos(car_yaw) + dz*math.sin(car_yaw)
        z_local = -dx*math.sin(car_yaw) + dz*math.cos(car_yaw)
        front_local.append((x_local, z_local))

    behind_local = []
    for (cx, cz, dist) in behind_points:
        dx = cx - car_x
        dz = cz - car_z
        x_local = dx*math.cos(car_yaw) + dz*math.sin(car_yaw)
        z_local = -dx*math.sin(car_yaw) + dz*math.cos(car_yaw)
        behind_local.append((x_local, z_local))

    return front_local, behind_local

def animate_run(blue_cones, yellow_cones, centerline_x, centerline_z, car_data,
                heading_length=3.0,
                front_distance_thresh=20.0,
                behind_distance_thresh=5.0):
    """
    Creates an animation showing:
      - Blue and yellow cones as static scatter points.
      - A car (blue dot) moving according to the CSV data.
      - A heading line (red) using the yaw_angle from the CSV data.
      - Only those centerline points that are within:
          - 'front_distance_thresh' (e.g. 20m) in front
          - 'behind_distance_thresh' (e.g. 5m) behind
        color-coded in magenta (front) and green (behind).

    Parameters
    ----------
    blue_cones : list of tuples (x, z)
    yellow_cones : list of tuples (x, z)
    centerline_x : list or np.array of x-coordinates for the centerline
    centerline_z : list or np.array of z-coordinates for the centerline
    car_data : dict with at least the following keys:
        {
            'time': np.array([...]),
            'x_pos': np.array([...]),
            'z_pos': np.array([...]),
            'yaw_angle': np.array([...]),  # in degrees
        }
    heading_length : float
        The length of the red heading line from the car marker.
    front_distance_thresh : float
        Only show centerline points in front if their local distance < this value.
    behind_distance_thresh : float
        Only show centerline points behind if their local distance < this value.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object.
    """

    # Extract relevant data
    t = car_data["time"]
    x_car = car_data["x_pos"]
    z_car = car_data["z_pos"]
    yaw_deg = car_data["yaw_angle"]
    # Convert yaw from degrees to radians
    yaw = np.deg2rad(yaw_deg)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Car + Centerline Points (Thresholded Distances)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    # --- Plot static elements ---
    # 1. Cones
    if blue_cones:
        bx, bz = zip(*blue_cones)
        ax.scatter(bx, bz, c='blue', s=10, label='Blue Cones')
    if yellow_cones:
        yx, yz = zip(*yellow_cones)
        ax.scatter(yx, yz, c='gold', s=10, label='Yellow Cones')

    # (Optional) If you want to see the full centerline as a dashed line:
    # ax.plot(centerline_x, centerline_z, 'k--', label='Centerline (global)')

    # --- Initialize animated elements ---
    # Car marker: single point
    car_point, = ax.plot([], [], 'bo', ms=8, label='Car')
    # Heading line: from car_position to car_position + heading_length
    heading_line, = ax.plot([], [], 'r-', lw=2, label='Heading')

    # We'll show centerline points color-coded each frame
    front_scatter = ax.scatter([], [], c='magenta', s=25, label='Front (<= 20m)')
    behind_scatter = ax.scatter([], [], c='green',   s=25, label='Behind (<= 5m)')

    # Combine for auto-limits
    all_x = list(x_car) + [pt[0] for pt in blue_cones] + [pt[0] for pt in yellow_cones]
    all_z = list(z_car) + [pt[1] for pt in blue_cones] + [pt[1] for pt in yellow_cones]
    # Add the centerline, if any
    all_x += list(centerline_x)
    all_z += list(centerline_z)

    if all_x and all_z:
        margin = 2.0
        min_x, max_x = min(all_x) - margin, max(all_x) + margin
        min_z, max_z = min(all_z) - margin, max(all_z) + margin
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_z, max_z)

    ax.legend()

    def update(frame_idx):
        # Current car position & heading
        x_curr = x_car[frame_idx]
        z_curr = z_car[frame_idx]
        yaw_curr = yaw[frame_idx]

        # Car marker
        car_point.set_data([x_curr], [z_curr])

        # Heading line
        x_heading = x_curr + heading_length * math.cos(yaw_curr)
        z_heading = z_curr + heading_length * math.sin(yaw_curr)
        heading_line.set_data([x_curr, x_heading], [z_curr, z_heading])

        # Classify centerline points (front vs behind) with distance thresholds
        xf_front = []
        zf_front = []
        xf_behind = []
        zf_behind = []

        for cx, cz in zip(centerline_x, centerline_z):
            dx = cx - x_curr
            dz = cz - z_curr
            # local coords
            x_local = dx * math.cos(yaw_curr) + dz * math.sin(yaw_curr)
            z_local = -dx * math.sin(yaw_curr) + dz * math.cos(yaw_curr)
            dist_local = math.hypot(x_local, z_local)

            # If in front (x_local >= 0) AND within front_distance_thresh
            if x_local >= 0 and dist_local <= front_distance_thresh:
                xf_front.append(cx)
                zf_front.append(cz)
            # If behind (x_local < 0) AND within behind_distance_thresh
            elif x_local < 0 and dist_local <= behind_distance_thresh:
                xf_behind.append(cx)
                zf_behind.append(cz)

        # Convert to Nx2 arrays for scatter
        front_coords = np.column_stack((xf_front, zf_front)) if xf_front else np.empty((0,2))
        behind_coords = np.column_stack((xf_behind, zf_behind)) if xf_behind else np.empty((0,2))

        front_scatter.set_offsets(front_coords)
        behind_scatter.set_offsets(behind_coords)

        # Return updated artists
        return car_point, heading_line, front_scatter, behind_scatter

    # Create animation
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(t),
        interval=20,   # ms between frames
        blit=True
    )

    plt.show()
data = read_csv_data("session3/run1.csv")
blue_cones, yellow_cones, centerline_x_vals, centerline_z_vals = parse_cone_data("../../sim/tracks/default.json")
animate_run(blue_cones, yellow_cones, centerline_x_vals, centerline_z_vals, data)
