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
    Ensures that the edges form a closed loop.
    """
    ordered_blue = order_cones_by_centerline(blue_cones, centerline_x, centerline_z)
    ordered_yellow = order_cones_by_centerline(yellow_cones, centerline_x, centerline_z)
    
    # Check if the first and last points are the same (or nearly the same).
    if ordered_blue and (np.hypot(ordered_blue[0][0] - ordered_blue[-1][0],
                                  ordered_blue[0][1] - ordered_blue[-1][1]) > 1e-6):
        ordered_blue.append(ordered_blue[0])
    
    if ordered_yellow and (np.hypot(ordered_yellow[0][0] - ordered_yellow[-1][0],
                                    ordered_yellow[0][1] - ordered_yellow[-1][1]) > 1e-6):
        ordered_yellow.append(ordered_yellow[0])
    
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
def compute_local_curvature(centerline_x, centerline_z, window_size=5):
    """
    Computes the local curvature at each point along the centerline by fitting a circle
    to a local window of points. The curvature is defined as the reciprocal of the circle’s
    radius (1/R). It is a signed value: positive for a left turn and negative for a right turn.
    
    Parameters:
      - centerline_x, centerline_z: lists or arrays of centerline coordinates.
      - window_size: number of consecutive points to use in the local circle fit (should be odd; default is 5).
    
    Returns:
      - curvatures: a list of curvature values (1/R) for each centerline point.
    """
    N = len(centerline_x)
    curvatures = [0.0] * N
    
    # Ensure window_size is odd and at least 3
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1
    half_window = window_size // 2

    for i in range(N):
        # Determine the local window indices (clamp at boundaries)
        start = max(0, i - half_window)
        end = min(N, i + half_window + 1)
        
        x_local = np.array(centerline_x[start:end])
        y_local = np.array(centerline_z[start:end])
        
        # Need at least 3 points to fit a circle
        if len(x_local) < 3:
            curvatures[i] = 0.0
            continue
        
        # Set up the linear system for the algebraic circle fit:
        # We model the circle as: x^2 + y^2 + D*x + E*y + F = 0.
        # For each point, we have: D*x + E*y + F = - (x^2 + y^2).
        A = np.column_stack((x_local, y_local, np.ones_like(x_local)))
        b_vec = -(x_local**2 + y_local**2)
        
        # Solve the least-squares problem: [D, E, F]
        sol, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        D, E, F = sol
        
        # The circle's center is (-D/2, -E/2) and its radius is computed by:
        center_x = -D / 2.0
        center_y = -E / 2.0
        R_sq = center_x**2 + center_y**2 - F
        if R_sq <= 1e-6:
            curvatures[i] = 0.0
        else:
            R = math.sqrt(R_sq)
            curvature = 1.0 / R
            
            # Determine the sign of the curvature.
            # Use the vectors from the fitted circle center to the neighboring points.
            if i > 0 and i < N - 1:
                vec1 = np.array([centerline_x[i - 1] - center_x, centerline_z[i - 1] - center_y])
                vec2 = np.array([centerline_x[i + 1] - center_x, centerline_z[i + 1] - center_y])
                cross_val = vec1[0] * vec2[1] - vec1[1] * vec2[0]
                # If the cross product is negative, the turn is to the right (negative curvature)
                if cross_val < 0:
                    curvature = -curvature
            curvatures[i] = curvature
            
    return curvatures

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
        left_normal = (-T[1], T[0])
        # Right normal (points toward the blue edge) = (T[1], -T[0])
        right_normal = (T[1], -T[0])
        
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
# --------------------- Heading differencem ----------------------
def compute_heading_difference(car_x, car_z, car_heading,
                              centerline_x, centerline_z):
    """
    Computes the difference between the car's heading and the local track heading,
    both measured relative to the same "reference" where:
       - heading = 0 means "east"
       - heading = pi/2 means "north"
       - heading = -pi/2 means "south"
       - heading = pi or -pi means "west"
    
    The result is guaranteed not to jump abruptly (e.g., from -pi/2 to +pi/2) 
    because we carefully unwrap both angles and then compute the difference.

    Parameters
    ----------
    car_x, car_z : float
        The car's global (x, z) position.
    car_heading : float
        The car's heading (in radians), which may be in any range (e.g., -3pi/2 
        or +pi/2 for "north"). This will be normalized to [-pi, pi].
    centerline_x, centerline_z : array-like
        The track centerline coordinates.

    Returns
    -------
    heading_diff : float
        The difference (in radians) between car heading and track heading at 
        the closest centerline point, constrained to [-pi/2, pi/2] by "flipping"
        (not hard-clipping). A positive value means the car is to the "left" 
        (counter-clockwise) relative to the track direction.
    """
    # 1. Compute track headings so that 0 = east, +pi/2 = north.
    #    => we must use atan2(dz, dx), NOT atan2(-dz, dx).
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
        # atan2(dz, dx) => 0 = east, +pi/2 = north, etc.
        track_headings[i] = math.atan2(dz, dx)

    # 2. Unwrap the track headings so that they form a continuous function.
    track_headings_unwrapped = np.unwrap(track_headings)

    # 3. Find the closest centerline point to the car and get that track heading.
    dists = [
        np.hypot(px - car_x, pz - car_z)
        for px, pz in zip(centerline_x, centerline_z)
    ]
    i_min = np.argmin(dists)
    track_heading_closest = track_headings_unwrapped[i_min]

    # 4. Normalize the car heading into [-pi, pi].
    #    This ensures that e.g. -3pi/2 also becomes +pi/2, etc.
    car_heading_normalized = (car_heading + math.pi) % (2*math.pi) - math.pi

    # (Optional) If your car heading is jumping around (like from +3.0 rad 
    # to -3.0 rad) frame to frame, you may need to do a "running" unwrapping 
    # across time. But for a single call, normalizing is often enough.

    # 5. Compute raw difference in [-pi, pi] between the two headings.
    heading_diff = (car_heading_normalized - track_heading_closest + math.pi) % (2*math.pi) - math.pi

    # 6. Because you mentioned the difference should never exceed ±90°,
    #    we can "flip" angles outside [-pi/2, pi/2] by ±pi. 
    #    This avoids a hard jump from -pi/2 to +pi/2.
    if heading_diff > math.pi/2:
        heading_diff -= math.pi
    elif heading_diff < -math.pi/2:
        heading_diff += math.pi

    return heading_diff
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

# --------------------- Distance to centerline ----------------
def compute_signed_distance_to_centerline(car_x, car_z, centerline_x, centerline_z):
    """
    Computes the signed perpendicular distance from the car's position (car_x, car_z)
    to the centerline (treated as a polyline). The sign is determined using the leftward
    normal of each segment (defined as (-tangent_y, tangent_x)). The function returns the
    distance (with sign) corresponding to the segment which gives the smallest absolute distance.
    
    Parameters:
      - car_x, car_z: The car's global (x, z) position.
      - centerline_x, centerline_z: Lists (or arrays) of the centerline's x and z coordinates.
    
    Returns:
      - The signed distance from the car to the centerline.
    """
    pts = list(zip(centerline_x, centerline_z))
    best_distance = float('inf')
    best_signed_distance = 0.0
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        # Vector along the segment.
        vx = b[0] - a[0]
        vz = b[1] - a[1]
        v_dot_v = vx * vx + vz * vz
        if v_dot_v == 0:
            proj = a
        else:
            t = ((car_x - a[0]) * vx + (car_z - a[1]) * vz) / v_dot_v
            if t < 0:
                proj = a
            elif t > 1:
                proj = b
            else:
                proj = (a[0] + t * vx, a[1] + t * vz)
        dist = math.hypot(car_x - proj[0], car_z - proj[1])
        if dist < best_distance:
            best_distance = dist
            # Compute unit tangent for the segment.
            if v_dot_v == 0:
                tangent = (1.0, 0.0)
            else:
                norm_v = math.sqrt(v_dot_v)
                tangent = (vx / norm_v, vz / norm_v)
            # Define the leftward normal (consistent with the heading difference convention).
            left_normal = (-tangent[1], tangent[0])
            diff_vec = (car_x - proj[0], car_z - proj[1])
            # The dot product gives the signed magnitude.
            sign = 1 if (diff_vec[0]*left_normal[0] + diff_vec[1]*left_normal[1]) >= 0 else -1
            best_signed_distance = sign * dist
    return best_signed_distance

# --------------------- Accelerations --------------------------
def compute_accelerations(time, vx, vy):
    ax = np.zeros_like(vx)
    ay = np.zeros_like(vy)
    
    # Use forward difference for the first sample.
    ax[0] = (vx[1] - vx[0]) / (time[1] - time[0])
    ay[0] = (vy[1] - vy[0]) / (time[1] - time[0])
    
    # Use central difference for interior points.
    for i in range(1, len(time)-1):
        dt = time[i+1] - time[i-1]
        ax[i] = (vx[i+1] - vx[i-1]) / dt
        ay[i] = (vy[i+1] - vy[i-1]) / dt
    
    # Use backward difference for the last sample.
    ax[-1] = (vx[-1] - vx[-2]) / (time[-1] - time[-2])
    ay[-1] = (vy[-1] - vy[-2]) / (time[-1] - time[-2])
    
    return ax, ay
# --------------------- Animation Function ---------------------
# LEGACY animation, used for debugging the csv writer and served as base for visualizer.py
# def animate_run(blue_cones, yellow_cones, centerline_x, centerline_z, car_data,
#                 heading_length=3.0,
#                 front_distance_thresh=20.0,
#                 behind_distance_thresh=5.0,
#                 max_ray_distance=20.0):
#     """
#     Creates an animation showing:
#       - The track with cones and track edges.
#       - A moving car with heading and raycasting.
#       - Local centerline points (visualized on the track).
#       - A bottom subplot showing both the track width (distance between yellow and blue edges)
#         and the centerline curvature (1/R, with sign) for each resampled centerline point
#         that lies within 5 m behind to 20 m ahead of the car's projection onto the centerline.
#       - Two bar plots in the lower-right corner: one showing the instantaneous heading difference
#         (car's heading minus track heading) and one showing the signed distance from the car to
#         the centerline.
#     """
#     # Extract car data.
#     t = car_data["time"]
#     x_car = car_data["x_pos"]
#     z_car = car_data["z_pos"]
#     yaw_deg = car_data["yaw_angle"]
#     yaw = np.deg2rad(yaw_deg)
#
#     # Create figure with two subplots (top and bottom).
#     fig, (ax_track, ax_width) = plt.subplots(2, 1, figsize=(10, 10),
#                                              gridspec_kw={'height_ratios': [3, 1]})
#     
#     # Adjust main subplots to leave space on the right for the bar plots.
#     fig.subplots_adjust(right=0.7)
#     
#     # Add a new axes in the lower right for heading difference.
#     ax_heading = fig.add_axes([0.75, 0.1, 0.1, 0.19])  # [left, bottom, width, height]
#     ax_heading.set_title("Heading Diff (rad)")
#     ax_heading.set_ylim(-1, 1)
#     ax_heading.set_xticks([])
#     heading_bar = ax_heading.bar(0, 0, width=0.5, color='purple')
#     
#     # Add a new axes above the heading diff for the signed distance.
#     ax_distance = fig.add_axes([0.88, 0.10, 0.1, 0.19])
#     ax_distance.set_title("Distance to Centerline (m)")
#     # Set y-axis limits as appropriate (e.g., -5 to 5 meters).
#     ax_distance.set_ylim(-2, 2)
#     ax_distance.set_xticks([])
#     distance_bar = ax_distance.bar(0, 0, width=0.5, color='blue')
#     
#     # --- Setup track (top) axes ---
#     ax_track.set_aspect('equal', adjustable='box')
#     ax_track.set_title("Car Run with Rays & Track Edges")
#     ax_track.set_xlabel("X")
#     ax_track.set_ylabel("Z")
#     
#     if blue_cones:
#         bx, bz = zip(*blue_cones)
#         ax_track.scatter(bx, bz, c='blue', s=10, label='Blue Cones')
#     if yellow_cones:
#         yx, yz = zip(*yellow_cones)
#         ax_track.scatter(yx, yz, c='gold', s=10, label='Yellow Cones')
#     
#     ordered_blue = order_cones_by_centerline(blue_cones, centerline_x, centerline_z)
#     ordered_yellow = order_cones_by_centerline(yellow_cones, centerline_x, centerline_z)
#     if ordered_blue:
#         bx_order, bz_order = zip(*ordered_blue)
#         ax_track.plot(bx_order, bz_order, 'b-', lw=2, label='Blue Track Edge')
#     if ordered_yellow:
#         yx_order, yz_order = zip(*ordered_yellow)
#         ax_track.plot(yx_order, yz_order, color='gold', lw=2, label='Yellow Track Edge')
#     
#     # Initialize animated elements on the track.
#     car_point, = ax_track.plot([], [], 'bo', ms=8, label='Car')
#     heading_line, = ax_track.plot([], [], 'r-', lw=2, label='Heading')
#     front_scatter = ax_track.scatter([], [], c='magenta', s=25, label='Front Centerline')
#     behind_scatter = ax_track.scatter([], [], c='green', s=25, label='Behind Centerline')
#     
#     yellow_angles_deg = np.arange(-20, 111, 10)
#     blue_angles_deg = np.arange(20, -111, -10)
#     yellow_angles = np.deg2rad(yellow_angles_deg)
#     blue_angles = np.deg2rad(blue_angles_deg)
#     yellow_ray_lines = [ax_track.plot([], [], color='yellow', linestyle='--', lw=1)[0]
#                         for _ in range(len(yellow_angles))]
#     blue_ray_lines = [ax_track.plot([], [], color='cyan', linestyle='--', lw=1)[0]
#                       for _ in range(len(blue_angles))]
#     
#     # --- Setup bottom axes for Track Width and Curvature ---
#     ax_width.set_title("Track Width and Centerline Curvature vs. Local X")
#     ax_width.set_xlabel("Local X (m)")
#     ax_width.set_ylabel("Track Width (m)")
#     ax_width.set_xlim(-5, 20)
#     ax_width.set_ylim(0, 10)
#     track_width_line, = ax_width.plot([], [], 'bo-', label='Track Width')
#     
#     # Create a twin y-axis for curvature.
#     ax_curv = ax_width.twinx()
#     ax_curv.set_ylabel("Curvature (1/m)")
#     ax_curv.set_ylim(-1, 1)
#     curvature_line, = ax_curv.plot([], [], 'r.-', label='Curvature (1/m)')
#     
#     # Combine legends from both axes on the bottom subplot.
#     lines = [track_width_line, curvature_line]
#     labels = [track_width_line.get_label(), curvature_line.get_label()]
#     ax_width.legend(lines, labels, loc='upper right')
#     
#     # Precompute cumulative distances along the resampled centerline.
#     cum_dist = compute_centerline_cumulative_distance(centerline_x, centerline_z)
#     centerline_pts = list(zip(centerline_x, centerline_z))
#     pts_array = np.array(centerline_pts)
#     
#     # Precompute track widths.
#     track_widths_all = compute_local_track_widths(centerline_x, centerline_z,
#                                                   ordered_blue, ordered_yellow,
#                                                   max_width=10.0)
#     # Precompute the local curvature for each centerline point.
#     curvatures_all = compute_local_curvature(centerline_x, centerline_z, window_size=5)
#     
#     def update(frame_idx):
#         x_curr = x_car[frame_idx]
#         z_curr = z_car[frame_idx]
#         yaw_curr = yaw[frame_idx]
#
#         # Update car marker and heading.
#         car_point.set_data([x_curr], [z_curr])
#         x_heading = x_curr + heading_length * math.cos(yaw_curr)
#         z_heading = z_curr + heading_length * math.sin(yaw_curr)
#         heading_line.set_data([x_curr, x_heading], [z_curr, z_heading])
#         
#         # Visualize local centerline points on the track.
#         resampled_pts = centerline_pts
#         front_local, behind_local, global_front, global_behind = get_local_centerline_points_by_distance(
#             x_curr, z_curr, yaw_curr, resampled_pts,
#             front_distance=5.0, behind_distance=20.0)
#         if global_front:
#             front_scatter.set_offsets(np.array(global_front))
#         else:
#             front_scatter.set_offsets(np.empty((0,2)))
#         if global_behind:
#             behind_scatter.set_offsets(np.array(global_behind))
#         else:
#             behind_scatter.set_offsets(np.empty((0,2)))
#         
#         # Raycasting for visualization.
#         yellow_ray_dists, blue_ray_dists = raycast_for_state(
#             x_curr, z_curr, yaw_curr, ordered_blue, ordered_yellow, max_distance=max_ray_distance)
#         for i, d in enumerate(yellow_ray_dists):
#             ray_angle = yaw_curr + yellow_angles[i]
#             end_x = x_curr + d * math.cos(ray_angle)
#             end_z = z_curr + d * math.sin(ray_angle)
#             yellow_ray_lines[i].set_data([x_curr, end_x], [z_curr, end_z])
#         for i, d in enumerate(blue_ray_dists):
#             ray_angle = yaw_curr + blue_angles[i]
#             end_x = x_curr + d * math.cos(ray_angle)
#             end_z = z_curr + d * math.sin(ray_angle)
#             blue_ray_lines[i].set_data([x_curr, end_x], [z_curr, end_z])
#         
#         # Update Track Width and Curvature Plot (Bottom Subplot).
#         dists_to_car = np.hypot(pts_array[:, 0] - x_curr, pts_array[:, 1] - z_curr)
#         i_proj = int(np.argmin(dists_to_car))
#         L_proj = cum_dist[i_proj]
#         
#         local_offsets = []
#         local_widths = []
#         local_curvs = []
#         for i in range(len(centerline_x)):
#             offset = cum_dist[i] - L_proj
#             if -5 <= offset <= 20:
#                 local_offsets.append(offset)
#                 local_widths.append(track_widths_all[i]["width"])
#                 local_curvs.append(curvatures_all[i])
#         
#         track_width_line.set_data(local_offsets, local_widths)
#         curvature_line.set_data(local_offsets, local_curvs)
#         
#         # --- Compute and Update Heading Difference Bar ---
#         heading_diff = compute_heading_difference(x_curr, z_curr, yaw_curr, centerline_x, centerline_z)
#         bar_rect = heading_bar[0]
#         if heading_diff >= 0:
#             bar_rect.set_y(0)
#             bar_rect.set_height(heading_diff)
#         else:
#             bar_rect.set_y(heading_diff)
#             bar_rect.set_height(-heading_diff)
#         
#         # --- Compute and Update Signed Distance Bar ---
#         signed_distance = compute_signed_distance_to_centerline(x_curr, z_curr, centerline_x, centerline_z)
#         dist_rect = distance_bar[0]
#         # For the bar, set its y-position and height so that a positive signed_distance
#         # makes the bar extend upward and a negative signed_distance extends downward.
#         if signed_distance >= 0:
#             dist_rect.set_y(0)
#             dist_rect.set_height(signed_distance)
#         else:
#             dist_rect.set_y(signed_distance)
#             dist_rect.set_height(-signed_distance)
#         
#         return (car_point, heading_line, front_scatter, behind_scatter,
#                 *yellow_ray_lines, *blue_ray_lines, track_width_line, curvature_line, 
#                 bar_rect, dist_rect)
#
#     anim = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)
#     plt.show()
#     return None

# ----------- --------------------- Main Script ---------------------
if __name__ == "__main__":
    # Read and adjust the CSV data.
    data = read_csv_data("session3/run1.csv")
    if data is None:
        exit(1)
    data = shift_car_position(data)

    # Parse cones and centerline from the JSON track file.
    blue_cones, yellow_cones, clx, clz = parse_cone_data("../../../sim/tracks/default.json")
    
    # Reverse the centerline to match the car's driving direction
    clx = clx[::-1]
    clz = clz[::-1]
    
    # Resample the centerline at 1 m intervals.
    resampled_clx, resampled_clz = resample_centerline(clx, clz, resolution=1.0)
    centerline_pts = list(zip(resampled_clx, resampled_clz))

    # Get the ordered track edges (used for raycasting and track width computations).
    ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx, clz)
    
    # Precompute curvature and track width for the resampled centerline.
    curvatures_all = compute_local_curvature(resampled_clx, resampled_clz, window_size=5)
    track_widths_all = compute_local_track_widths(resampled_clx, resampled_clz,
                                                  ordered_blue, ordered_yellow,
                                                  max_width=10.0)
    
    # Prepare the output CSV with the required headers.
    output_filename = "../mid/session3/run1.csv"
    with open(output_filename, "w", newline="") as csvfile:
        fieldnames = []
        # Front 20 points ahead of the car.
        for i in range(1, 21):
            # Removed rel_x columns since they're just 1, 2, 3, etc.
            fieldnames.extend([f"rel_z{i}", f"c{i}", f"tw{i}"])
        # 5 points behind the car.
        for i in range(1, 6):
            fieldnames.extend([f"b_rel_z{i}", f"b_c{i}", f"b_tw{i}"])
        # Add the current (projection) point's curvature and track width.
        fieldnames.extend(["c0", "tw0"])
        # 14 yellow ray distances.
        for i in range(1, 15):
            fieldnames.append(f"yr{i}")
        # 14 blue ray distances.
        for i in range(1, 15):
            fieldnames.append(f"br{i}")
        # Distance to centerline, heading difference, and vehicle dynamics.
        fieldnames.extend(["dc", "dh", "vx", "vy", "psi_dot", "ax", "ay", "steering", "throttle", "brake"])
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        time = data["time"]
        num_frames = len(time)
        for frame in range(num_frames):
            car_x = data["x_pos"][frame]
            car_z = data["z_pos"][frame]
            # Convert yaw from degrees to radians.
            yaw_curr = math.radians(data["yaw_angle"][frame])
            
            # Use the provided function to get local centerline points both ahead and behind.
            front_local, behind_local, _, _ = get_local_centerline_points_by_distance(
                car_x, car_z, yaw_curr, centerline_pts,
                front_distance=20.0, behind_distance=5.0)
            
            # Interpolate front local points so we have one sample per meter (1 to 20).
            if front_local:
                fl = np.array(front_local)  # Each row: (arc_offset, lateral, dummy)
                x_front = fl[:, 0]
                z_front = fl[:, 1]
                target_x = np.arange(1, 21)
                target_z = np.interp(target_x, x_front, z_front, left=z_front[0], right=z_front[-1])
            else:
                target_x = np.arange(1, 21)
                target_z = np.full(20, float("nan"))
            
            # Interpolate behind local points so we have one sample per meter for -5 to -1.
            if behind_local:
                bl = np.array(behind_local)  # Each row: (arc_offset, lateral, dummy)
                x_behind = bl[:, 0]
                z_behind = bl[:, 1]
                # Target behind offsets: -5, -4, -3, -2, -1 (in increasing order).
                target_x_b = np.arange(-5, 0, 1)
                target_z_b = np.interp(target_x_b, x_behind, z_behind, left=z_behind[0], right=z_behind[-1])
            else:
                target_x_b = np.arange(-5, 0, 1)
                target_z_b = np.full(5, float("nan"))
            
            # Determine the projection index on the resampled centerline.
            dists = [math.hypot(pt[0] - car_x, pt[1] - car_z) for pt in centerline_pts]
            i_proj = int(np.argmin(dists))
            
            row = {}
            # Fill in front (ahead) points.
            for i, d in enumerate(target_x, start=1):
                row[f"rel_z{i}"] = target_z[i - 1]
                idx = i_proj + int(round(d))
                if idx < len(resampled_clx):
                    row[f"c{i}"] = curvatures_all[idx]
                    row[f"tw{i}"] = track_widths_all[idx]["width"]
                else:
                    row[f"c{i}"] = float("nan")
                    row[f"tw{i}"] = float("nan")
            
            # Fill in behind points.
            for i, d in enumerate(target_x_b, start=1):
                row[f"b_rel_z{i}"] = target_z_b[i - 1]
                idx_b = i_proj + int(round(d))
                if 0 <= idx_b < len(resampled_clx):
                    row[f"b_c{i}"] = curvatures_all[idx_b]
                    row[f"b_tw{i}"] = track_widths_all[idx_b]["width"]
                else:
                    row[f"b_c{i}"] = float("nan")
                    row[f"b_tw{i}"] = float("nan")
            
            # Write current (projection) point's track width and curvature.
            if i_proj < len(resampled_clx):
                row["c0"] = curvatures_all[i_proj]
                row["tw0"] = track_widths_all[i_proj]["width"]
            else:
                row["c0"] = float("nan")
                row["tw0"] = float("nan")
            
            # Compute raycasting distances for yellow and blue edges.
            yellow_ray_dists, blue_ray_dists = raycast_for_state(
                car_x, car_z, yaw_curr, ordered_blue, ordered_yellow, max_distance=20.0)
            for i, d in enumerate(yellow_ray_dists, start=1):
                row[f"yr{i}"] = d
            for i, d in enumerate(blue_ray_dists, start=1):
                row[f"br{i}"] = d
            
            # Compute the signed distance to the centerline and heading difference.
            dc = compute_signed_distance_to_centerline(car_x, car_z, resampled_clx, resampled_clz)
            dh = compute_heading_difference(car_x, car_z, yaw_curr, resampled_clx, resampled_clz)
            row["dc"] = -dc
            row["dh"] = dh
            
            # Append vehicle dynamics from the input CSV.
            vx = data["long_vel"]
            vy = data["lat_vel"]
            row["vx"] = data["long_vel"][frame]
            row["vy"] = data["lat_vel"][frame]
            row["psi_dot"] = data["yaw_rate"][frame]
            ax, ay = compute_accelerations(time,vx,vy)
            row["ax"] = ax[frame]
            row["ay"] = ay[frame]

            # Append driver inputs from the input CSV
            steering = data["steering"]
            throttle = data["throttle"]
            brake = data["brake"]

            row["steering"] = steering[frame]
            row["throttle"] = throttle[frame]
            row["brake"] = brake[frame]
            
            writer.writerow(row)
    
    print(f"Output CSV saved to {output_filename}")
