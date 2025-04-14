import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

from utils import *
from processor import Processor

class Augmenter:
    def __init__(self, buffer_size = 20):
        self.buffer_size = buffer_size

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

    def mirror_track(self, track):
        """
        Mirrors the track around the y-axis.
        """
        r_clx = [-x for x in track["r_clx"]]
        r_clz = track["r_clz"]
        centerline_pts = [(-x, z) for x, z in track["centerline_pts"]]
        ordered_yellow = [(-x, z) for x, z in track["ordered_blue"]]
        ordered_blue = [(-x, z) for x, z in track["ordered_yellow"]]
        curvatures_all = [-c for c in track["curvatures_all"]]
        track_widths_all = []
        for i in track["track_widths_all"]:
            track_widths_all.append({'center': (-i['center'][0], i['center'][1]), 'width': i['width']})
        return {
                "r_clx": r_clx,
                "r_clz": r_clz,
                "centerline_pts": centerline_pts,
                "ordered_blue": ordered_blue,
                "ordered_yellow": ordered_yellow,
                "curvatures_all": curvatures_all,
                "track_widths_all": track_widths_all
                }

    def save_track_data(self, track_data, output_path):
        """
        Saves the track data to a JSON file.
        """
        with open(output_path, 'w') as f:
            json.dump(track_data, f, indent=4)

    def mirror_recording(self, csv):
        df = pd.read_csv(csv)
        df.x_pos = -df.x_pos
        df.yaw_angle = -df.yaw_angle - 180
        df.lat_vel = - df.lat_vel
        df.yaw_rate = -df.yaw_rate
        df.steering = -df.steering
        return df

    def reset_realtime_state(self):
        self.time_buffer = []
        self.vx_buffer = []
        self.vy_buffer = []

    def visualize_absolute(self, data, track, heading_length=3.0, use_postprocessed=False):
        """
        Visualize frame-by-frame in absolute (global) coordinates.
        The scene (cones, track edges, centerline) is drawn from the JSON file,
        and the car (and its associated cast rays and local centerline points) is animated.
        """
        print("[Visualizer] Starting absolute visualization...")

        # Build track data from the JSON.
        track_data = track
        # Also, get the raw cone and centerline information.
        blue_cones = track_data["ordered_blue"]
        yellow_cones = track_data["ordered_yellow"]
        clx = track_data["r_clx"]
        clz = track_data["r_clz"]
        clx_rev = clx
        clz_rev = clz
        clx_abs, clz_abs = resample_centerline(clx, clz, resolution=1.0)
        centerline_pts_fw = list(zip(clx_abs, clz_abs))

        df_features = pd.DataFrame(data) 
        frames = df_features.to_dict('records')

        # --- Set up Matplotlib figure and axes ---
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10), 
                                                  gridspec_kw={'height_ratios': [3, 1]})
        fig.subplots_adjust(right=0.7)

        # Top panel: plot cones, track edges, and centerline.
        if blue_cones:
            bx, bz = zip(*blue_cones)
            ax_top.scatter(bx, bz, c='blue', marker='o', label="Blue Cones")
        if yellow_cones:
            yx, yz = zip(*yellow_cones)
            ax_top.scatter(yx, yz, c='gold', marker='o', label="Yellow Cones")
        ordered_blue, ordered_yellow = create_track_edges(blue_cones, yellow_cones, clx_rev, clz_rev)
        if ordered_blue:
            blue_x, blue_z = zip(*ordered_blue)
            ax_top.plot(blue_x, blue_z, 'b-', label="Blue Edge")
        if ordered_yellow:
            yellow_x, yellow_z = zip(*ordered_yellow)
            ax_top.plot(yellow_x, yellow_z, 'y-', label="Yellow Edge")
        if clx_abs and clz_abs:
            ax_top.plot(clx_abs, clz_abs, 'm--', label="Centerline")
        ax_top.set_xlabel("X (m)")
        ax_top.set_ylabel("Z (m)")
        ax_top.set_title("Absolute Scene with Cast Rays & Local Points")
        ax_top.legend()

        # Persistent markers for the car and its heading.
        car_marker, = ax_top.plot([], [], 'ro', markersize=10, label="Car")
        heading_line, = ax_top.plot([], [], 'r-', lw=2, label="Heading")

        # Cast ray lines.
        blue_angles = np.deg2rad(np.arange(-20, 111, 10))
        yellow_angles = np.deg2rad(np.arange(20, -111, -10))
        yellow_ray_lines = [ax_top.plot([], [], color='yellow', linestyle='--', lw=1)[0]
                            for _ in yellow_angles]
        blue_ray_lines = [ax_top.plot([], [], color='cyan', linestyle='--', lw=1)[0]
                          for _ in blue_angles]

        # Scatter objects for local centerline points (transformed to absolute coordinates).
        front_scatter = ax_top.scatter([], [], c='magenta', marker='o', s=50, label="Forward Local Points")
        behind_scatter = ax_top.scatter([], [], c='green', marker='o', s=50, label="Behind Local Points")
        ax_top.legend()

        # Bottom panel: local metrics (track width and curvature)
        ax_bottom.set_title("Local Metrics: Track Width & Curvature")
        ax_bottom.set_xlabel("Local X (m)")
        ax_bottom.set_ylabel("Track Width (m)")
        ax_bottom.set_xlim(-5, 21)
        ax_bottom.set_ylim(0, 10)
        f_track_line, = ax_bottom.plot([], [], 'bo-', label="Fwd Track Width")
        b_track_line, = ax_bottom.plot([], [], 'bo-')
        ax_bottom.legend(loc='upper left')
        ax_curv_bottom = ax_bottom.twinx()
        ax_curv_bottom.set_ylim(-0.5, 0.5)
        f_curv_line, = ax_curv_bottom.plot([], [], 'r.-', label="Curvature")
        b_curv_line, = ax_curv_bottom.plot([], [], 'r.-')
        ax_curv_bottom.legend(loc='upper right')

        # Reset realtime state.
        self.reset_realtime_state()

        for frame in frames:
            # --- Top Panel Update: Absolute View ---
            car_x = frame["x_pos"]
            car_z = frame["z_pos"]
            heading_deg = frame["yaw_angle"]
            heading_rad = math.radians(heading_deg)
            car_marker.set_data([car_x], [car_z])
            hx = car_x + heading_length * math.cos(heading_rad)
            hy = car_z + heading_length * math.sin(heading_rad)
            heading_line.set_data([car_x, hx], [car_z, hy])

            # Update cast rays.
            for idx, angle in enumerate(yellow_angles, start=1):
                ray_dist = frame.get(f"yr{idx}", 0)
                local_x = ray_dist * math.cos(angle)
                local_y = ray_dist * math.sin(angle)
                abs_x = car_x + local_x * math.cos(heading_rad) - local_y * math.sin(heading_rad)
                abs_y = car_z + local_x * math.sin(heading_rad) + local_y * math.cos(heading_rad)
                yellow_ray_lines[idx-1].set_data([car_x, abs_x], [car_z, abs_y])
            for idx, angle in enumerate(blue_angles, start=1):
                ray_dist = frame.get(f"br{idx}", 0)
                local_x = ray_dist * math.cos(angle)
                local_y = ray_dist * math.sin(angle)
                abs_x = car_x + local_x * math.cos(heading_rad) - local_y * math.sin(heading_rad)
                abs_y = car_z + local_x * math.sin(heading_rad) + local_y * math.cos(heading_rad)
                blue_ray_lines[idx-1].set_data([car_x, abs_x], [car_z, abs_y])

            # Update local centerline points (transformed to absolute).
            front_local, behind_local, global_front, global_behind = get_local_centerline_points_by_distance(
                car_x, car_z, heading_deg, centerline_pts_fw,
                front_distance=20.0, behind_distance=5.0
            )
            if global_front:
                front_scatter.set_offsets(np.array(global_front))
            else:
                front_scatter.set_offsets(np.empty((0, 2)))
            if global_behind:
                behind_scatter.set_offsets(np.array(global_behind))
            else:
                behind_scatter.set_offsets(np.empty((0, 2)))

            # --- Bottom Panel Update: Local Metrics ---
            f_tw = [frame.get(f"tw{j}", 0.0) for j in range(21)]
            f_curv = [frame.get(f"c{j}", 0.0) for j in range(21)]
            f_track_line.set_data(np.arange(21), f_tw)
            f_curv_line.set_data(np.arange(21), f_curv)
            b_tw = [frame.get(f"b_tw{j}", 0.0) for j in range(1, 6)]
            b_curv = [frame.get(f"b_c{j}", 0.0) for j in range(1, 6)]
            b_track_line.set_data(np.array(list(range(-5, 0))), b_tw)
            b_curv_line.set_data(np.array(list(range(-5, 0))), b_curv)

            plt.draw()
            plt.pause(0.001)
        print("[Visualizer] Finished absolute visualization.")
        plt.show()

augmenter = Augmenter()
processor = Processor()

track = "sim/tracks/track10.json"
track_data = augmenter.build_track_data(track)
mirrored_track = augmenter.mirror_track(track_data)
augmenter.save_track_data(mirrored_track, "sim/tracks/track10_mirrored.json")

recording = "runs/training/final/track10.csv"
augmented_recording = augmenter.mirror_recording(recording)

df_processed = processor.process_csv(augmented_recording, mirrored_track)
df_processed.to_csv("runs/processed/final/track10_mirrored.csv", index=False)

# df_processed = pd.read_csv("runs/processed/final/track10_mirrored.csv")

# Visualize the augmented recording
augmenter.visualize_absolute(df_processed, mirrored_track, heading_length=3.0)
