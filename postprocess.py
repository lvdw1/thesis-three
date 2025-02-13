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

def compute_centerline_cumulative_distance(centerline_x, centerline_z):
    """
    Computes cumulative distances along the centerline.
    Returns a list where each entry corresponds to the arc-length from the start.
    """
    cum_dist = [0.0]
    for i in range(1, len(centerline_x)):
        dx = centerline_x[i] - centerline_x[i - 1]
        dz = centerline_z[i] - centerline_z[i - 1]
        dist = np.hypot(dx, dz)
        cum_dist.append(cum_dist[-1] + dist)
    return cum_dist

def project_cone_onto_centerline(cone, centerline_x, centerline_z, cum_dist):
    """
    Projects a cone onto the centerline by finding the closest centerline point.
    Returns the cumulative distance at that point.
    """
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
    """
    Orders a list of cones (each a tuple (x, z)) by their projection distance onto the centerline.
    """
    if not cones:
        return []
    cum_dist = compute_centerline_cumulative_distance(centerline_x, centerline_z)
    cones_with_projection = []
    for cone in cones:
        proj_distance = project_cone_onto_centerline(cone, centerline_x, centerline_z, cum_dist)
        cones_with_projection.append((proj_distance, cone))
    # Sort by the projection distance
    cones_with_projection.sort(key=lambda item: item[0])
    ordered_cones = [cone for _, cone in cones_with_projection]
    return ordered_cones

def create_track_edges(blue_cones, yellow_cones, centerline_x, centerline_z):
    """
    Creates track edges by ordering the blue and yellow cones along the centerline.
    Returns two lists: one for the blue edge and one for the yellow edge.
    """
    ordered_blue = order_cones_by_centerline(blue_cones, centerline_x, centerline_z)
    ordered_yellow = order_cones_by_centerline(yellow_cones, centerline_x, centerline_z)
    return ordered_blue, ordered_yellow

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

def cross2D(a, b):
    """
    Computes the 2D cross product (scalar) between vectors a and b.
    """
    return a[0]*b[1] - a[1]*b[0]

def ray_segment_intersection(ray_origin, ray_direction, seg_start, seg_end):
    """
    Computes the intersection of a ray with a line segment (if any).

    The ray is defined by:
        R(t) = ray_origin + t * ray_direction,   t >= 0

    The segment is defined by its endpoints seg_start and seg_end.

    Returns:
        t (float): the distance along the ray at which the intersection occurs,
                   or None if there is no intersection (or if the intersection occurs
                   beyond the segment limits).
    """
    p = ray_origin
    r = ray_direction
    q = seg_start
    # Compute the segment vector
    s = (seg_end[0] - seg_start[0], seg_end[1] - seg_start[1])
    
    rxs = cross2D(r, s)
    if abs(rxs) < 1e-6:
        # The ray and segment are parallel (or nearly so)
        return None
    
    qp = (q[0] - p[0], q[1] - p[1])
    t = cross2D(qp, s) / rxs
    u = cross2D(qp, r) / rxs

    if t >= 0 and (0 <= u <= 1):
        return t
    return None

def raycast_for_state(car_x, car_z, car_heading, blue_edge, yellow_edge, max_distance=20):
    """
    Casts two bundles of rays from the current car position and heading:
      - Yellow rays: relative angles from -20° to 110° (in 10° steps), checking for 
        intersections with the yellow track edge.
      - Blue rays: relative angles from 20° down to -110° (in 10° steps), checking for 
        intersections with the blue track edge.

    Parameters
    ----------
    car_x : float
        X-coordinate of the car.
    car_z : float
        Z-coordinate of the car.
    car_heading : float
        Car heading in radians (global frame).
    blue_edge : list of (x, z)
        Ordered points defining the blue track edge.
    yellow_edge : list of (x, z)
        Ordered points defining the yellow track edge.
    max_distance : float, optional
        Maximum distance for a ray (default is 20 m). If no intersection is found within
        this range, the returned distance for that ray is max_distance.

    Returns
    -------
    yellow_ray_distances : list of float
        Intersection distances for each yellow ray.
    blue_ray_distances : list of float
        Intersection distances for each blue ray.
    """
    # Define ray bundles in degrees (relative to the car's heading)
    # Yellow rays: from -20° to 110° in 10° increments
    yellow_angles_deg = np.arange(-20, 111, 10)
    # Blue rays: from 20° down to -110° in 10° increments
    blue_angles_deg = np.arange(20, -111, -10)
    
    yellow_ray_distances = []
    blue_ray_distances = []
    
    # Process yellow rays
    for rel_angle_deg in yellow_angles_deg:
        rel_angle = math.radians(rel_angle_deg)
        ray_angle = car_heading + rel_angle
        ray_dir = (math.cos(ray_angle), math.sin(ray_angle))
        
        closest_distance = max_distance  # initialize to max
        # Iterate over consecutive segments in the yellow track edge
        for i in range(len(yellow_edge) - 1):
            seg_start = yellow_edge[i]
            seg_end = yellow_edge[i+1]
            t = ray_segment_intersection((car_x, car_z), ray_dir, seg_start, seg_end)
            if t is not None and t < closest_distance:
                closest_distance = t
        yellow_ray_distances.append(closest_distance)
    
    # Process blue rays
    for rel_angle_deg in blue_angles_deg:
        rel_angle = math.radians(rel_angle_deg)
        ray_angle = car_heading + rel_angle
        ray_dir = (math.cos(ray_angle), math.sin(ray_angle))
        
        closest_distance = max_distance  # initialize to max
        # Iterate over consecutive segments in the blue track edge
        for i in range(len(blue_edge) - 1):
            seg_start = blue_edge[i]
            seg_end = blue_edge[i+1]
            t = ray_segment_intersection((car_x, car_z), ray_dir, seg_start, seg_end)
            if t is not None and t < closest_distance:
                closest_distance = t
        blue_ray_distances.append(closest_distance)
    
    return yellow_ray_distances, blue_ray_distances

def raycast_over_time(x_car, z_car, yaw, blue_edge, yellow_edge, max_distance=20):
    """
    Loops over a time series of car positions and headings, performing raycasting
    for each time step.

    Parameters
    ----------
    x_car : array-like of float
        Array of car x-positions.
    z_car : array-like of float
        Array of car z-positions.
    yaw : array-like of float
        Array of car headings (in radians).
    blue_edge : list of (x, z)
        Ordered points defining the blue track edge.
    yellow_edge : list of (x, z)
        Ordered points defining the yellow track edge.
    max_distance : float, optional
        Maximum ray distance (default is 20 m).

    Returns
    -------
    yellow_all : np.ndarray
        Array of shape (n_time, n_yellow_rays) with intersection distances for yellow rays.
    blue_all : np.ndarray
        Array of shape (n_time, n_blue_rays) with intersection distances for blue rays.
    """
    yellow_all = []
    blue_all = []
    for cx, cz, ch in zip(x_car, z_car, yaw):
        yellow_dist, blue_dist = raycast_for_state(cx, cz, ch, blue_edge, yellow_edge, max_distance)
        yellow_all.append(yellow_dist)
        blue_all.append(blue_dist)
    return np.array(yellow_all), np.array(blue_all)

def animate_run(blue_cones, yellow_cones, centerline_x, centerline_z, car_data,
                heading_length=3.0,
                front_distance_thresh=20.0,
                behind_distance_thresh=5.0,
                max_ray_distance=20.0):
    """
    Creates an animation showing:
      - Blue and yellow cones as static scatter points.
      - A car (blue dot) moving according to the CSV data.
      - A heading line (red) using the yaw_angle from the CSV data.
      - Only those centerline points that are within:
          - 'front_distance_thresh' in front
          - 'behind_distance_thresh' behind
        color-coded in magenta (front) and green (behind).
      - Track edges are drawn by connecting the cones (after ordering them
        along the centerline) in their respective colors.
      - A bundle of rays is cast from the car:
            * Yellow rays: from -20° to 110° (in 10° steps) that check intersections
              with the yellow track edge.
            * Blue rays: from 20° to -110° (in 10° steps) that check intersections
              with the blue track edge.
        Rays extend up to max_ray_distance.

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
        Only show centerline points in front if within this distance.
    behind_distance_thresh : float
        Only show centerline points behind if within this distance.
    max_ray_distance : float
        Maximum distance for raycasting (default 20 m).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The animation object.
    """
    # Extract car data
    t = car_data["time"]
    x_car = car_data["x_pos"]
    z_car = car_data["z_pos"]
    yaw_deg = car_data["yaw_angle"]
    yaw = np.deg2rad(yaw_deg)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Car Run with Rays & Track Edges")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")

    # --- Plot static elements ---
    # Cones
    if blue_cones:
        bx, bz = zip(*blue_cones)
        ax.scatter(bx, bz, c='blue', s=10, label='Blue Cones')
    if yellow_cones:
        yx, yz = zip(*yellow_cones)
        ax.scatter(yx, yz, c='gold', s=10, label='Yellow Cones')

    # Order cones along the centerline to form track edges
    ordered_blue = order_cones_by_centerline(blue_cones, centerline_x, centerline_z)
    ordered_yellow = order_cones_by_centerline(yellow_cones, centerline_x, centerline_z)
    if ordered_blue:
        bx_order, bz_order = zip(*ordered_blue)
        ax.plot(bx_order, bz_order, 'b-', lw=2, label='Blue Track Edge')
    if ordered_yellow:
        yx_order, yz_order = zip(*ordered_yellow)
        ax.plot(yx_order, yz_order, color='gold', lw=2, label='Yellow Track Edge')

    # (Optional) Plot the full centerline
    # ax.plot(centerline_x, centerline_z, 'k--', label='Centerline (global)')

    # --- Initialize animated elements ---
    # Car marker and heading line
    car_point, = ax.plot([], [], 'bo', ms=8, label='Car')
    heading_line, = ax.plot([], [], 'r-', lw=2, label='Heading')

    # Centerline points for threshold display
    front_scatter = ax.scatter([], [], c='magenta', s=25, label='Front (<= 20m)')
    behind_scatter = ax.scatter([], [], c='green', s=25, label='Behind (<= 5m)')

    # Predefine ray bundles (relative angles)
    yellow_angles_deg = np.arange(-20, 111, 10)  # 14 rays
    blue_angles_deg = np.arange(20, -111, -10)     # 14 rays
    yellow_angles = np.deg2rad(yellow_angles_deg)
    blue_angles = np.deg2rad(blue_angles_deg)

    # Create line objects for the rays (using dashed lines)
    yellow_ray_lines = [ax.plot([], [], color='yellow', linestyle='--', lw=1)[0]
                        for _ in range(len(yellow_angles))]
    blue_ray_lines = [ax.plot([], [], color='cyan', linestyle='--', lw=1)[0]
                      for _ in range(len(blue_angles))]

    # Set axis limits based on all points
    all_x = list(x_car) + [pt[0] for pt in blue_cones] + [pt[0] for pt in yellow_cones] + list(centerline_x)
    all_z = list(z_car) + [pt[1] for pt in blue_cones] + [pt[1] for pt in yellow_cones] + list(centerline_z)
    if all_x and all_z:
        margin = 2.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_z) - margin, max(all_z) + margin)

    ax.legend()

    def update(frame_idx):
        # Current car state
        x_curr = x_car[frame_idx]
        z_curr = z_car[frame_idx]
        yaw_curr = yaw[frame_idx]

        # Update car marker and heading line
        car_point.set_data([x_curr], [z_curr])
        x_heading = x_curr + heading_length * math.cos(yaw_curr)
        z_heading = z_curr + heading_length * math.sin(yaw_curr)
        heading_line.set_data([x_curr, x_heading], [z_curr, z_heading])

        # Classify centerline points (for front and behind display)
        xf_front, zf_front = [], []
        xf_behind, zf_behind = [], []
        for cx, cz in zip(centerline_x, centerline_z):
            dx = cx - x_curr
            dz = cz - z_curr
            # Rotate to local car coordinates:
            x_local = dx * math.cos(yaw_curr) + dz * math.sin(yaw_curr)
            z_local = -dx * math.sin(yaw_curr) + dz * math.cos(yaw_curr)
            dist_local = math.hypot(x_local, z_local)
            if x_local >= 0 and dist_local <= front_distance_thresh:
                xf_front.append(cx)
                zf_front.append(cz)
            elif x_local < 0 and dist_local <= behind_distance_thresh:
                xf_behind.append(cx)
                zf_behind.append(cz)
        front_coords = np.column_stack((xf_front, zf_front)) if xf_front else np.empty((0, 2))
        behind_coords = np.column_stack((xf_behind, zf_behind)) if xf_behind else np.empty((0, 2))
        front_scatter.set_offsets(front_coords)
        behind_scatter.set_offsets(behind_coords)

        # --------- Raycasting and Ray Updates ----------
        # Obtain intersection distances for current state:
        # (Pass ordered_blue as blue_edge and ordered_yellow as yellow_edge)
        yellow_ray_dists, blue_ray_dists = raycast_for_state(
            x_curr, z_curr, yaw_curr, ordered_blue, ordered_yellow, max_distance=max_ray_distance)

        # Update yellow ray lines:
        for i, d in enumerate(yellow_ray_dists):
            ray_angle = yaw_curr + yellow_angles[i]
            end_x = x_curr + d * math.cos(ray_angle)
            end_z = z_curr + d * math.sin(ray_angle)
            yellow_ray_lines[i].set_data([x_curr, end_x], [z_curr, end_z])

        # Update blue ray lines:
        for i, d in enumerate(blue_ray_dists):
            ray_angle = yaw_curr + blue_angles[i]
            end_x = x_curr + d * math.cos(ray_angle)
            end_z = z_curr + d * math.sin(ray_angle)
            blue_ray_lines[i].set_data([x_curr, end_x], [z_curr, end_z])

        # Return all animated artists
        return (car_point, heading_line, front_scatter, behind_scatter,
                *yellow_ray_lines, *blue_ray_lines)

    # Create and return the animation
    anim = animation.FuncAnimation(
        fig, update, frames=len(t), interval=20, blit=True)
    plt.show()

data = read_csv_data("session3/run1.csv")
blue_cones, yellow_cones, centerline_x_vals, centerline_z_vals = parse_cone_data("../../sim/tracks/default.json")
animate_run(blue_cones, yellow_cones, centerline_x_vals, centerline_z_vals, data)
