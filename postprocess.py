import os
import json
import logging
import math
import csv
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --------------------- Cone and Track Functions ---------------------

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
    Orders cones by their projection distance onto the centerline.
    """
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
    """
    Creates track edges by ordering the blue and yellow cones along the centerline.
    """
    ordered_blue = order_cones_by_centerline(blue_cones, centerline_x, centerline_z)
    ordered_yellow = order_cones_by_centerline(yellow_cones, centerline_x, centerline_z)
    return ordered_blue, ordered_yellow

# --------------------- CSV Data Function ---------------------

def shift_car_position(data, shift_distance=1.5):
    """
    Shifts the car's global (x,z) coordinates 1.5 units to its right.
    """
    if data is None:
        return None

    yaw = np.deg2rad(data["yaw_angle"])
    offset_x = shift_distance * np.sin(yaw)
    offset_z = -shift_distance * np.cos(yaw)

    data["x_pos"] = data["x_pos"] + offset_x
    data["z_pos"] = data["z_pos"] + offset_z
    return data

def read_csv_data(file_path):
    """
    Reads CSV data and organizes it into structured numpy arrays.
    """
    if not os.path.exists(file_path):
        logging.error(f"CSV file not found at path: {file_path}")
        return None

    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        times, x_pos, z_pos, yaw_angle = [], [], [], []
        long_vel, lat_vel, yaw_rate, long_acc, lat_acc = [], [], [], [], []

        for row in reader:
            try:
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
            "long_acc": np.array(long_acc),
            "lat_acc": np.array(lat_acc),
        }

# --------------------- Resampling and Local Extraction ---------------------

def resample_centerline(centerline_x, centerline_z, resolution=1.0):
    """
    Resamples the centerline so that points are spaced by `resolution` meters.
    """
    cum_dist = compute_centerline_cumulative_distance(centerline_x, centerline_z)
    total_length = cum_dist[-1]
    new_dists = np.arange(0, total_length + resolution, resolution)
    new_x = np.interp(new_dists, cum_dist, centerline_x)
    new_z = np.interp(new_dists, cum_dist, centerline_z)
    return new_x.tolist(), new_z.tolist()

def get_local_centerline_points_by_distance(car_x, car_z, car_yaw, centerline_points,
                                              front_distance=20.0, behind_distance=5.0):
    """
    Given a list of resampled (global) centerline points (as (x,z) tuples),
    this function computes the cumulative arc length along the centerline, finds
    the projection point (the one closest to the car), and then selects (or
    interpolates) points exactly front_distance ahead and behind_distance behind
    along the centerline.
    
    It returns:
      - front_local: list of points (x_local, z_local, curvature) for points ahead,
                     where x_local is forced to be the arc-length offset (so the
                     20th point appears at x = 20).
      - behind_local: list of points (x_local, z_local, curvature) for points behind,
      - global_front: corresponding global (x,z) points ahead (for plotting)
      - global_behind: corresponding global (x,z) points behind (for plotting)
    
    The local coordinates are computed such that:
      x_local = (arc length from projection)   and 
      z_local = signed lateral offset (computed as usual).
    """
    pts = np.array(centerline_points)  # shape (N,2)
    cum_dist = np.array(compute_centerline_cumulative_distance(pts[:, 0].tolist(),
                                                               pts[:, 1].tolist()))
    N = len(pts)
    
    # Find the index of the centerline point closest to the car.
    dists = np.hypot(pts[:, 0] - car_x, pts[:, 1] - car_z)
    i_proj = int(np.argmin(dists))
    L_proj = cum_dist[i_proj]
    
    # Define target arc-lengths for front and behind.
    L_front_target = L_proj + front_distance
    L_behind_target = L_proj - behind_distance

    def interpolate_point(target_L):
        # If target_L is outside the available range, clamp it.
        if target_L <= cum_dist[0]:
            return pts[0]
        if target_L >= cum_dist[-1]:
            return pts[-1]
        idx = np.searchsorted(cum_dist, target_L)
        L1, L2 = cum_dist[idx - 1], cum_dist[idx]
        p1, p2 = pts[idx - 1], pts[idx]
        ratio = (target_L - L1) / (L2 - L1)
        return p1 + ratio * (p2 - p1)
    
    global_front_pt = interpolate_point(L_front_target)
    global_behind_pt = interpolate_point(L_behind_target)
    
    # Select points between the projection point and the target arc lengths.
    front_mask = (cum_dist >= L_proj) & (cum_dist <= L_front_target)
    behind_mask = (cum_dist >= L_behind_target) & (cum_dist <= L_proj)
    
    global_front = pts[front_mask].tolist()
    global_behind = pts[behind_mask].tolist()
    
    # Append (or insert) the interpolated endpoint(s) if needed.
    if len(global_front) == 0 or np.hypot(*(np.array(global_front[-1]) - global_front_pt)) > 1e-3:
        global_front.append(global_front_pt.tolist())
    if len(global_behind) == 0 or np.hypot(*(np.array(global_behind[0]) - global_behind_pt)) > 1e-3:
        global_behind.insert(0, global_behind_pt.tolist())
    
    # Now, instead of using the standard global-to-local transform, we force the x-coordinate
    # (longitudinal direction) to be the arc-length offset (i.e. 0, 1, 2, ... m).
    # The lateral (z_local) coordinate is still computed as the signed perpendicular distance.
    front_local = []
    indices_front = np.where(front_mask)[0]
    for idx in indices_front:
        p = pts[idx]
        arc_offset = cum_dist[idx] - L_proj  # this will be in meters
        dx = p[0] - car_x
        dz = p[1] - car_z
        lateral = -dx * math.sin(car_yaw) + dz * math.cos(car_yaw)
        front_local.append((arc_offset, lateral, 0))
    # Append the interpolated endpoint if it wasn’t in the original mask.
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
        arc_offset = cum_dist[idx] - L_proj  # will be negative
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
# --------------------- Centerpoints Classification with Curvature ---------------------

# --------------------- Raycasting Functions ---------------------

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
    if t >= 0 and (0 <= u <= 1):
        return t
    return None

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

def compute_ray_edge_intersection_distance(ray_origin, ray_direction, edge_points, max_distance=10.0):
    """
    Computes the intersection distance from a ray (origin, direction) with a given edge polyline.
    The edge is given as a list of (x,z) points forming consecutive segments.
    
    Returns the smallest positive distance along the ray at which an intersection occurs,
    or None if no intersection is found within max_distance.
    """
    best_t = max_distance
    found = False
    for i in range(len(edge_points) - 1):
        seg_start = edge_points[i]
        seg_end = edge_points[i + 1]
        t_val = ray_segment_intersection(ray_origin, ray_direction, seg_start, seg_end)
        if t_val is not None and t_val < best_t:
            best_t = t_val
            found = True
    return best_t if found else None


def compute_local_track_widths(resampled_clx, resampled_clz, ordered_blue, ordered_yellow, max_width=10.0):
    """
    For each resampled centerline point (spaced at 1 m intervals), compute the local track width.
    
    The width is determined by:
      - Computing the local tangent of the centerline.
      - Defining a left normal (pointing toward the yellow edge) and a right normal (toward the blue edge).
      - Casting a ray from the centerline point in the left normal direction to find the yellow edge intersection,
        and in the right normal direction to find the blue edge intersection.
      - The local width is the sum of the distances from the centerline to the yellow and blue edge intersections.
    
    Parameters:
      - resampled_clx, resampled_clz: lists of x and z coordinates (resampled at every meter).
      - ordered_blue: list of (x,z) points for the blue (right) track edge (ordered along the centerline).
      - ordered_yellow: list of (x,z) points for the yellow (left) track edge.
      - max_width: maximum distance to search for an intersection (used as a fallback if no intersection is found).
    
    Returns:
      A list of dictionaries, each containing:
         - "center": the centerline point (x, z)
         - "width": the computed local track width (in meters)
         - "blue_pt": the intersection point on the blue edge
         - "yellow_pt": the intersection point on the yellow edge
    """
    results = []
    pts = list(zip(resampled_clx, resampled_clz))
    N = len(pts)
    
    for i in range(N):
        # Compute a local tangent using central differences (with forward/backward for boundaries)
        if i == 0:
            dx = pts[i + 1][0] - pts[i][0]
            dz = pts[i + 1][1] - pts[i][1]
        elif i == N - 1:
            dx = pts[i][0] - pts[i - 1][0]
            dz = pts[i][1] - pts[i - 1][1]
        else:
            dx = pts[i + 1][0] - pts[i - 1][0]
            dz = pts[i + 1][1] - pts[i - 1][1]
        T_norm = math.hypot(dx, dz)
        if T_norm == 0:
            T = (1.0, 0.0)
        else:
            T = (dx / T_norm, dz / T_norm)
        
        # Define normals:
        # Left normal (points toward the yellow edge) = (-T_y, T_x)
        right_normal = (-T[1], T[0])
        # Right normal (points toward the blue edge) = (T[1], -T[0])
        left_normal = (T[1], -T[0])
        
        center = pts[i]
        
        # Cast rays to find intersections with the edges
        d_yellow = compute_ray_edge_intersection_distance(center, left_normal, ordered_yellow, max_distance=max_width)
        d_blue = compute_ray_edge_intersection_distance(center, right_normal, ordered_blue, max_distance=max_width)
        
        # Use max_width if no intersection is found (this is a fallback)
        if d_yellow is None:
            d_yellow = max_width
        if d_blue is None:
            d_blue = max_width
        
        # Compute the intersection points on each edge
        yellow_pt = (center[0] + d_yellow * left_normal[0], center[1] + d_yellow * left_normal[1])
        blue_pt = (center[0] + d_blue * right_normal[0], center[1] + d_blue * right_normal[1])
        
        width = d_yellow + d_blue
        
        results.append({
            "center": center,
            "width": width,
            "yellow_pt": yellow_pt,
            "blue_pt": blue_pt,
        })
        
    return results

# --------------------- Cubic Polynomial Fit ---------------------

def fit_cubic_polynomial(front_points, domain=(-5, 20), num_samples=100):
    front_points = np.array(front_points)
    if len(front_points) < 4:
        return front_points
    xs = front_points[:, 0]
    zs = front_points[:, 1]
    coeffs = np.polyfit(xs, zs, 3)
    poly = np.poly1d(coeffs)
    x_sample = np.linspace(domain[0], domain[1], num_samples)
    z_sample = poly(x_sample)
    return np.vstack((x_sample, z_sample)).T

# --------------------- Animation Function ---------------------
def animate_run(blue_cones, yellow_cones, centerline_x, centerline_z, car_data,
                heading_length=3.0,
                front_distance_thresh=20.0,
                behind_distance_thresh=5.0,
                max_ray_distance=20.0):
    """
    Creates an animation showing:
      - The track with cones and track edges.
      - A moving car with heading and raycasting.
      - Local centerline points (visualized on the track).
      - A bottom subplot showing the track width (distance between yellow and blue edges)
        for each resampled centerline point that lies within 5 m behind to 20 m ahead
        of the car's current projection on the centerline.
    """
    # Extract car data
    t = car_data["time"]
    x_car = car_data["x_pos"]
    z_car = car_data["z_pos"]
    yaw_deg = car_data["yaw_angle"]
    yaw = np.deg2rad(yaw_deg)

    # Create figure with two subplots.
    fig, (ax_track, ax_width) = plt.subplots(2, 1, figsize=(8, 10),
                                             gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Setup track (top) axes ---
    ax_track.set_aspect('equal', adjustable='box')
    ax_track.set_title("Car Run with Rays & Track Edges")
    ax_track.set_xlabel("X")
    ax_track.set_ylabel("Z")
    
    if blue_cones:
        bx, bz = zip(*blue_cones)
        ax_track.scatter(bx, bz, c='blue', s=10, label='Blue Cones')
    if yellow_cones:
        yx, yz = zip(*yellow_cones)
        ax_track.scatter(yx, yz, c='gold', s=10, label='Yellow Cones')
    
    ordered_blue = order_cones_by_centerline(blue_cones, centerline_x, centerline_z)
    ordered_yellow = order_cones_by_centerline(yellow_cones, centerline_x, centerline_z)
    if ordered_blue:
        bx_order, bz_order = zip(*ordered_blue)
        ax_track.plot(bx_order, bz_order, 'b-', lw=2, label='Blue Track Edge')
    if ordered_yellow:
        yx_order, yz_order = zip(*ordered_yellow)
        ax_track.plot(yx_order, yz_order, color='gold', lw=2, label='Yellow Track Edge')
    
    # Initialize animated elements on the track.
    car_point, = ax_track.plot([], [], 'bo', ms=8, label='Car')
    heading_line, = ax_track.plot([], [], 'r-', lw=2, label='Heading')
    front_scatter = ax_track.scatter([], [], c='magenta', s=25, label='Front Centerline')
    behind_scatter = ax_track.scatter([], [], c='green', s=25, label='Behind Centerline')
    
    yellow_angles_deg = np.arange(-20, 111, 10)
    blue_angles_deg = np.arange(20, -111, -10)
    yellow_angles = np.deg2rad(yellow_angles_deg)
    blue_angles = np.deg2rad(blue_angles_deg)
    yellow_ray_lines = [ax_track.plot([], [], color='yellow', linestyle='--', lw=1)[0]
                        for _ in range(len(yellow_angles))]
    blue_ray_lines = [ax_track.plot([], [], color='cyan', linestyle='--', lw=1)[0]
                      for _ in range(len(blue_angles))]
    
    # Set track axis limits.
    all_x = list(x_car) + [pt[0] for pt in blue_cones] + [pt[0] for pt in yellow_cones] + list(centerline_x)
    all_z = list(z_car) + [pt[1] for pt in blue_cones] + [pt[1] for pt in yellow_cones] + list(centerline_z)
    if all_x and all_z:
        margin = 2.0
        ax_track.set_xlim(min(all_x)-margin, max(all_x)+margin)
        ax_track.set_ylim(min(all_z)-margin, max(all_z)+margin)
    ax_track.legend()
    
    # --- Setup track width (bottom) axes ---
    ax_width.set_title("Track Width vs. Local X")
    ax_width.set_xlabel("Local X (m)")
    ax_width.set_ylabel("Track Width (m)")
    ax_width.set_xlim(-5, 20)
    ax_width.set_ylim(0, 10)  # y-axis between 0 and 5 meters
    track_width_line, = ax_width.plot([], [], 'bo-', label='Track Width')
    ax_width.legend()
    
    # Precompute cumulative distances along the resampled centerline and the track widths for each centerpoint.
    cum_dist = compute_centerline_cumulative_distance(centerline_x, centerline_z)
    # Use max_width=2.5 so that if no intersection is found, each side falls back to 2.5 m.
    track_widths_all = compute_local_track_widths(centerline_x, centerline_z,
                                                  ordered_blue, ordered_yellow,
                                                  max_width=10)
    centerline_pts = list(zip(centerline_x, centerline_z))
    pts_array = np.array(centerline_pts)
    
    def update(frame_idx):
        x_curr = x_car[frame_idx]
        z_curr = z_car[frame_idx]
        yaw_curr = yaw[frame_idx]

        # Update car marker and heading.
        car_point.set_data([x_curr], [z_curr])
        x_heading = x_curr + heading_length * math.cos(yaw_curr)
        z_heading = z_curr + heading_length * math.sin(yaw_curr)
        heading_line.set_data([x_curr, x_heading], [z_curr, z_heading])
        
        # Visualize local centerline points on the track.
        resampled_pts = centerline_pts  # already resampled at every meter
        front_local, behind_local, global_front, global_behind = get_local_centerline_points_by_distance(
            x_curr, z_curr, yaw_curr, resampled_pts,
            front_distance=5.0, behind_distance=20.0)  # 20 m ahead and 5 m behind (note the parameter order)
        if global_front:
            front_scatter.set_offsets(np.array(global_front))
        else:
            front_scatter.set_offsets(np.empty((0,2)))
        if global_behind:
            behind_scatter.set_offsets(np.array(global_behind))
        else:
            behind_scatter.set_offsets(np.empty((0,2)))
        
        # --------- Raycasting for visualization ----------
        yellow_ray_dists, blue_ray_dists = raycast_for_state(
            x_curr, z_curr, yaw_curr, ordered_blue, ordered_yellow, max_distance=max_ray_distance)
        for i, d in enumerate(yellow_ray_dists):
            ray_angle = yaw_curr + yellow_angles[i]
            end_x = x_curr + d * math.cos(ray_angle)
            end_z = z_curr + d * math.sin(ray_angle)
            yellow_ray_lines[i].set_data([x_curr, end_x], [z_curr, end_z])
        for i, d in enumerate(blue_ray_dists):
            ray_angle = yaw_curr + blue_angles[i]
            end_x = x_curr + d * math.cos(ray_angle)
            end_z = z_curr + d * math.sin(ray_angle)
            blue_ray_lines[i].set_data([x_curr, end_x], [z_curr, end_z])
        
        # --- Update Track Width Plot (Bottom Subplot) ---
        # Determine the projection of the car onto the centerline.
        dists_to_car = np.hypot(pts_array[:, 0] - x_curr, pts_array[:, 1] - z_curr)
        i_proj = int(np.argmin(dists_to_car))
        L_proj = cum_dist[i_proj]
        
        local_offsets = []
        local_widths = []
        # Loop over every centerline point (precomputed in track_widths_all).
        # (The ordering in track_widths_all matches that of centerline_x/centerline_z.)
        for i, tw in enumerate(track_widths_all):
            # Compute the local arc-length offset relative to the car's projection.
            offset = cum_dist[i] - L_proj
            # Only consider points between -5 m and +20 m.
            if -5 <= offset <= 20:
                local_offsets.append(offset)
                local_widths.append(tw["width"])
        
        track_width_line.set_data(local_offsets, local_widths)
        
        return (car_point, heading_line, front_scatter, behind_scatter,
                *yellow_ray_lines, *blue_ray_lines, track_width_line)

    anim = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)
    plt.tight_layout()
    plt.show()
    return anim

# --------------------- Main Script ---------------------
if __name__ == "__main__":
    data = read_csv_data("session3/run1.csv")
    data = shift_car_position(data)
    blue_cones, yellow_cones, clx, clz = parse_cone_data("../../sim/tracks/default.json")
    # Resample the centerline so that points are every 1 meter.
    resampled_clx, resampled_clz = resample_centerline(clx, clz, resolution=1.0)
    ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx, clz)
    track_widths = compute_local_track_widths(resampled_clx, resampled_clz, ordered_blue, ordered_yellow)
# track_widths is a list of dicts with keys: "center", "width", "yellow_pt", "blue_pt"
    animate_run(blue_cones, yellow_cones, resampled_clx, resampled_clz, data)
