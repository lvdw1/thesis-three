import numpy as np
import math
import pandas as pd

from utils import *

class Processor:
    """
    Handles processing of raw sensor data.
    Can process a single frame or an entire CSV file.
    Maintains realtime state for acceleration calculation.
    """
    def __init__(self, buffer_size=20):
        self.reset_realtime_state()
        self.buffer_size = buffer_size

    def reset_realtime_state(self):
        self.time_buffer = []
        self.vx_buffer = []
        self.vy_buffer = []

    def process_frame(self, sensor_data, track_data):
        """
        Process a single sensor reading and compute features.
        """
        t = sensor_data["time"]
        car_x = sensor_data["x_pos"]
        car_z = sensor_data["z_pos"]
        yaw_angle = sensor_data["yaw_angle"]
        long_vel = sensor_data["long_vel"]
        lat_vel = sensor_data["lat_vel"]
        yaw_rate = sensor_data["yaw_rate"]
        steering = sensor_data["steering"]
        throttle = sensor_data["throttle"]
        brake = sensor_data["brake"]

        # Shift car position to a lookahead point
        x_shifted, z_shifted = shift_position_single(car_x, car_z, yaw_angle, shift_distance=-1.7)

        # Compute local acceleration (using previous frame if available)
        self.time_buffer.append(t)
        self.vx_buffer.append(long_vel)
        self.vy_buffer.append(lat_vel)

        if len(self.time_buffer) > self.buffer_size:
            self.time_buffer.pop(0)
            self.vx_buffer.pop(0)
            self.vy_buffer.pop(0)

        if len(self.time_buffer) >= 3:
            ax_arr, ay_arr = compute_acceleration(self.time_buffer, self.vx_buffer, self.vy_buffer, window_length = self.buffer_size)
            ax, ay = ax_arr[-1], ay_arr[-1]
        else:
            ax, ay = 0.0, 0.0

        yaw_rad = math.radians(yaw_angle)
        brd, yrd = raycast_for_state(
            x_shifted, z_shifted, yaw_rad,
            track_data["ordered_blue"], track_data["ordered_yellow"],
            max_distance=20.0
        )
        dc = compute_signed_distance_to_centerline(
            x_shifted, z_shifted, track_data["r_clx"], track_data["r_clz"]
        )
        dh = compute_heading_difference(
            x_shifted, z_shifted, yaw_rad,
            track_data["r_clx"], track_data["r_clz"]
        )

        # Find the projection of the car on the centerline
        dists = [math.hypot(px - car_x, pz - car_z) for (px, pz) in track_data["centerline_pts"]]
        i_proj = int(np.argmin(dists))
        if i_proj < len(track_data["r_clx"]):
            c0 = track_data["curvatures_all"][i_proj]
            tw0 = track_data["track_widths_all"][i_proj]["width"]
        else:
            c0 = 0.0
            tw0 = 4.0

        # Compute local forward points (and interpolation)
        front_local, behind_local, _, _ = get_local_centerline_points_by_distance(
            x_shifted, z_shifted, yaw_rad, track_data["centerline_pts"],
            front_distance=20.0, behind_distance=5.0
        )
        if behind_local:
            fl = np.array(front_local)
            x_front = fl[:, 0]
            z_front = fl[:, 1]
            target_x = np.arange(1, 21)
            target_z = np.interp(target_x, x_front, z_front, left=z_front[0], right=z_front[-1])
        else:
            target_x = np.arange(1, 21)
            target_z = np.zeros(20)

        if front_local:
            bl = np.array(behind_local)
            x_behind = bl[:, 0]
            z_behind = bl[:, 1]
            target_x_b = np.arange(-5, 0)
            target_z_b = np.interp(target_x_b, x_behind, z_behind, left=z_behind[0], right=z_behind[-1])
        else:
            target_x_b = np.arange(-5, 0)
            target_z_b = np.zeros(5)

        # Build the output feature dictionary
        row_dict = {
            "time": t,
            "x_pos": x_shifted,
            "z_pos": z_shifted,
            "yaw_angle": yaw_angle,
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
            row_dict[f"rel_z{j}"] = target_z[j - 1]
            # Wrap-around index calculation
            idx_front = (i_proj + int(round(d))) % len(track_data["r_clx"])
            row_dict[f"c{j}"] = track_data["curvatures_all"][idx_front]
            row_dict[f"tw{j}"] = track_data["track_widths_all"][idx_front]["width"]
        for j, d in enumerate(target_x_b, start=1):
            row_dict[f"b_rel_z{j}"] = target_z_b[j - 1]
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

    def process_csv(self, data_dict, track_data):
        """
        Process an entire CSV (frame-by-frame) using process_frame.
        """
        self.reset_realtime_state()
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
                "yaw_angle": yaw_arr[i],
                "long_vel": vx_arr[i],
                "lat_vel": vy_arr[i],
                "yaw_rate": yr_arr[i],
                "steering": st_arr[i],
                "throttle": th_arr[i],
                "brake": br_arr[i],
            }
            frame = self.process_frame(sensor_data, track_data)
            frames.append(frame)

        return pd.DataFrame(frames)

    def build_track_data(self, json_path):
        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)
        clx_rev = clx
        clz_rev = clz
        r_clx, r_clz = resample_centerline(clx_rev, clz_rev, resolution=1.0)
        centerline_pts = list(zip(r_clx, r_clz))
        ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx_rev, clz_rev)
        curvatures_all = compute_local_curvature(r_clx, r_clz, window_size=5)
        track_widths_all = compute_local_track_widths(r_clx, r_clz, ordered_blue, ordered_yellow, max_width=10.0)
        return {
            "r_clx": r_clx,
            "r_clz": r_clz,
            "centerline_pts": centerline_pts,
            "ordered_blue": ordered_blue,
            "ordered_yellow": ordered_yellow,
            "curvatures_all": curvatures_all,
            "track_widths_all": track_widths_all
           }
