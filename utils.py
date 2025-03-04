# utils.py
import os
import csv
import json
import math
import logging
import numpy as np


# ---------------------------------------------------------------------
# ------------------ UTILITY / POSTPROCESSING CODE --------------------
# ---------------------------------------------------------------------

def get_local_centerline_points_by_distance(car_x, car_z, car_yaw, centerline_points,
                                              front_distance=20.0, behind_distance=5.0):
    """
    Given a list of resampled (global) centerline points (as (x,z) tuples) for a circular track,
    compute the cumulative arc length along the centerline and find the projection point (closest
    to the car). Then, select or interpolate points exactly front_distance ahead and behind_distance
    behind along the centerline, wrapping around if necessary.
    
    Returns:
      - front_local: list of tuples (arc_offset, lateral, dummy) for points ahead.
      - behind_local: list of tuples (arc_offset, lateral, dummy) for points behind.
      - global_front: corresponding global (x,z) points for forward local points.
      - global_behind: corresponding global (x,z) points for backward local points.
    
    The local coordinates are defined relative to the projection point on the centerline:
      x_local = (arc length from projection, with wrap-around)
      z_local = signed lateral offset.
    """
    pts = np.array(centerline_points)  # shape (N,2)
    cum_dist = np.array(compute_centerline_cumulative_distance(pts[:, 0].tolist(),
                                                               pts[:, 1].tolist()))
    T = cum_dist[-1]  # total track length
    N = len(pts)
    
    # Find the projection index and its cumulative distance.
    dists = np.hypot(pts[:,0] - car_x, pts[:,1] - car_z)
    i_proj = int(np.argmin(dists))
    L_proj = cum_dist[i_proj]
    
    # Compute target arc distances (wrap if necessary).
    L_front_target = L_proj + front_distance
    if L_front_target > T:
        L_front_target_wrapped = L_front_target - T
    else:
        L_front_target_wrapped = None  # no wrapping needed
    
    L_behind_target = L_proj - behind_distance
    if L_behind_target < 0:
        L_behind_target_wrapped = L_behind_target + T
    else:
        L_behind_target_wrapped = None  # no wrapping needed

    # Function to interpolate a point given a target arc length.
    def interpolate_point(target_L, cum, pts_array):
        if target_L <= cum[0]:
            return pts_array[0]
        if target_L >= cum[-1]:
            return pts_array[-1]
        idx = np.searchsorted(cum, target_L)
        L1, L2 = cum[idx-1], cum[idx]
        p1, p2 = pts_array[idx-1], pts_array[idx]
        ratio = (target_L - L1) / (L2 - L1)
        return p1 + ratio * (p2 - p1)
    
    # Build forward points.
    global_front = []
    front_local = []
    # First, take points from L_proj up to T.
    front_mask1 = (cum_dist >= L_proj) & (cum_dist <= T)
    if np.any(front_mask1):
        pts_front1 = pts[front_mask1]
        cum_front1 = cum_dist[front_mask1]
        for i, p in zip(np.where(front_mask1)[0], pts_front1):
            if cum_dist[i] <= L_front_target if L_front_target_wrapped is None else True:
                arc_offset = cum_dist[i] - L_proj
                dx = p[0] - car_x
                dz = p[1] - car_z
                lateral = -dx * math.sin(car_yaw) + dz * math.cos(car_yaw)
                global_front.append(p.tolist())
                front_local.append((arc_offset, lateral, 0))
    # If wrapping is needed, then take points from the beginning.
    if L_front_target_wrapped is not None:
        front_mask2 = (cum_dist >= 0) & (cum_dist <= L_front_target_wrapped)
        if np.any(front_mask2):
            pts_front2 = pts[front_mask2]
            cum_front2 = cum_dist[front_mask2]
            for i, p in zip(np.where(front_mask2)[0], pts_front2):
                arc_offset = (cum_dist[i] + T) - L_proj  # add T to account for wrap
                dx = p[0] - car_x
                dz = p[1] - car_z
                lateral = -dx * math.sin(car_yaw) + dz * math.cos(car_yaw)
                global_front.append(p.tolist())
                front_local.append((arc_offset, lateral, 0))
    
    # Build backward points.
    global_behind = []
    behind_local = []
    # First, take points from 0 up to L_proj.
    behind_mask1 = (cum_dist >= 0) & (cum_dist <= L_proj)
    if np.any(behind_mask1):
        pts_behind1 = pts[behind_mask1]
        cum_behind1 = cum_dist[behind_mask1]
        for i, p in zip(np.where(behind_mask1)[0], pts_behind1):
            if cum_dist[i] >= L_behind_target if L_behind_target_wrapped is None else True:
                arc_offset = cum_dist[i] - L_proj
                dx = p[0] - car_x
                dz = p[1] - car_z
                lateral = -dx * math.sin(car_yaw) + dz * math.cos(car_yaw)
                global_behind.append(p.tolist())
                behind_local.append((arc_offset, lateral, 0))
    # If wrapping is needed, take points from the end.
    if L_behind_target_wrapped is not None:
        behind_mask2 = (cum_dist >= L_behind_target_wrapped) & (cum_dist <= T)
        if np.any(behind_mask2):
            pts_behind2 = pts[behind_mask2]
            cum_behind2 = cum_dist[behind_mask2]
            for i, p in zip(np.where(behind_mask2)[0], pts_behind2):
                arc_offset = (cum_dist[i] - T) - L_proj  # subtract T for wrap
                dx = p[0] - car_x
                dz = p[1] - car_z
                lateral = -dx * math.sin(car_yaw) + dz * math.cos(car_yaw)
                global_behind.insert(0, p.tolist())  # prepend
                behind_local.insert(0, (arc_offset, lateral, 0))
    
    return front_local, behind_local, global_front, global_behind

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
        "yaw_deg": np.array(yaw_angle),
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
    # SWAP: Previously yellow was left (-20 to 110), blue was right (20 to -110)
    # Now blue is left (-20 to 110), yellow is right (20 to -110)
    blue_angles_deg = np.arange(-20, 111, 10)   # Changed from yellow_angles_deg
    yellow_angles_deg = np.arange(20, -111, -10) # Changed from blue_angles_deg
    
    yellow_ray_distances = []
    blue_ray_distances = []
    
    # Use blue angles for blue rays (left side)
    for rel_angle_deg in blue_angles_deg:
        rel_angle = math.radians(rel_angle_deg)
        ray_angle = car_heading + rel_angle
        ray_dir = (math.cos(ray_angle), math.sin(ray_angle))
        closest_distance = max_distance
        for i in range(len(blue_edge)-1):  # Now using blue_edge instead of yellow_edge
            seg_start = blue_edge[i]
            seg_end = blue_edge[i+1]
            t_val = ray_segment_intersection((car_x, car_z), ray_dir, seg_start, seg_end)
            if t_val is not None and t_val < closest_distance:
                closest_distance = t_val
        blue_ray_distances.append(closest_distance)  # Changed from yellow_ray_distances
    
    # Use yellow angles for yellow rays (right side)
    for rel_angle_deg in yellow_angles_deg:
        rel_angle = math.radians(rel_angle_deg)
        ray_angle = car_heading + rel_angle
        ray_dir = (math.cos(ray_angle), math.sin(ray_angle))
        closest_distance = max_distance
        for i in range(len(yellow_edge)-1):  # Now using yellow_edge instead of blue_edge
            seg_start = yellow_edge[i]
            seg_end = yellow_edge[i+1]
            t_val = ray_segment_intersection((car_x, car_z), ray_dir, seg_start, seg_end)
            if t_val is not None and t_val < closest_distance:
                closest_distance = t_val
        yellow_ray_distances.append(closest_distance)  # Changed from blue_ray_distances
    
    return blue_ray_distances, yellow_ray_distances  # Swapped order to match the naming

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
        right_normal = (-T[1], T[0])
        left_normal = (T[1], -T[0])
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
