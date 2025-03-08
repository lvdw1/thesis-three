import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils import *
###############################################
# 3. Visualizer Class
###############################################

class Visualizer:
    """
    Visualizes a run frame-by-frame.
    
    Provides two visualization modes:
      - Relative: the local (vehicle-centered) coordinate system.
      - Absolute: the global coordinate system with scene context.
    """
    def __init__(self, processor, transformer, nn_model):
        """
        processor: instance of a Processor class that provides process_frame().
        transformer: feature transformer (e.g., scaler/PCA) used in the pipeline.
        nn_model: trained NN model (used here only if you wish to overlay predictions).
        """
        self.processor = processor
        self.transformer = transformer
        self.nn_model = nn_model

    def visualize_relative(self, csv_path, json_path, heading_length=3.0, use_postprocessed=False):
        """
        Visualize frame-by-frame in the local (relative) coordinate system.
        The car is fixed at (0,0) and local features (cast rays, centerline points, etc.)
        are plotted relative to the vehicle.
        """
        print("[Visualizer] Starting relative visualization...")
        
        # Reset realtime state so each run starts fresh.
        self.processor.reset_realtime_state()
        track_data = self.processor.build_track_data(json_path)
        
        if use_postprocessed:
            # For postprocessed CSV, just read the data directly
            print("[Visualizer] Using postprocessed CSV data...")
            df_features = pd.read_csv(csv_path)
            frames = df_features.to_dict('records')
        else:
            # Process the data on-the-fly
            print("[Visualizer] Processing CSV data on-the-fly...")
            data_dict = read_csv_data(csv_path)
            if data_dict is None:
                print("Could not load CSV data.")
                return
            
        # --- Set up Matplotlib figure and axes ---
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 10), 
                                                  gridspec_kw={'height_ratios': [3, 1]})
        ax_top.set_xlim(-10, 30)
        ax_top.set_ylim(-10, 10)
        ax_top.set_aspect('equal', adjustable='box')
        ax_top.set_title("Local Frame - Relative Visualization")
        ax_top.set_xlabel("Local X (m)")
        ax_top.set_ylabel("Local Z (m)")

        # A small subplot for displaying heading difference
        ax_heading = fig.add_axes([0.75, 0.1, 0.1, 0.19])
        ax_heading.set_title("Heading Diff (rad)")
        ax_heading.set_ylim(-1, 1)
        ax_heading.set_xticks([])
        heading_bar = ax_heading.bar(0, 0, width=0.5, color='purple')

        # Plot elements for the car and its heading.
        car_point, = ax_top.plot([], [], 'ko', ms=8, label='Car')
        heading_line, = ax_top.plot([], [], 'r-', lw=2, label='Heading')

        # Scatter plots for local centerline points.
        front_scatter = ax_top.scatter([], [], c='magenta', s=25, label='Front Centerline')
        behind_scatter = ax_top.scatter([], [], c='green', s=25, label='Behind Centerline')
        centerline_line, = ax_top.plot([], [], 'k-', lw=1, label='Centerline')
        centerline_bline, = ax_top.plot([], [], 'k-', lw=1)

        # Set up cast rays (yellow and blue)
        blue_angles_deg = np.arange(-20, 111, 10)
        yellow_angles_deg = np.arange(20, -111, -10)
        yellow_angles = np.deg2rad(yellow_angles_deg)
        blue_angles = np.deg2rad(blue_angles_deg)
        yellow_ray_lines = [ax_top.plot([], [], color='yellow', linestyle='--', lw=1)[0]
                            for _ in yellow_angles]
        blue_ray_lines = [ax_top.plot([], [], color='cyan', linestyle='--', lw=1)[0]
                          for _ in blue_angles]

        # Bottom panel for track width and curvature metrics.
        ax_bottom.set_title("Track Width and Centerline Curvature")
        ax_bottom.set_xlabel("Local X (m)")
        ax_bottom.set_ylabel("Track Width (m)")
        ax_bottom.set_xlim(-5, 20)
        ax_bottom.set_ylim(0, 10)
        track_width_line, = ax_bottom.plot([], [], 'bo-', label='Fwd Track Width')
        track_width_line_back, = ax_bottom.plot([], [], 'bo-')
        ax_bottom.legend(loc='upper left')

        ax_curv = ax_bottom.twinx()
        ax_curv.set_ylim(-0.5, 0.5)
        curvature_line, = ax_curv.plot([], [], 'r.-', label='Fwd Curvature')
        curvature_line_back, = ax_curv.plot([], [], 'r.-')
        ax_curv.legend(loc='upper right')

        if use_postprocessed:
            for frame in frames:
                # In the relative frame the car is always at (0,0).
                car_point.set_data([0], [0])
                # The heading line is drawn along the x-axis.
                heading_line.set_data([0, heading_length], [0, 0])

                # Update forward local centerline points.
                front_pts = []
                for j in range(1, 21):
                    x_val = float(j)
                    z_val = frame.get(f"rel_z{j}", 0.0)
                    front_pts.append([x_val, z_val])
                front_scatter.set_offsets(np.array(front_pts))

                # Update behind local centerline points.
                behind_pts = []
                for j, x_val in enumerate(range(-5, 0), start=1):
                    z_val = frame.get(f"b_rel_z{j}", 0.0)
                    behind_pts.append([float(x_val), z_val])
                behind_scatter.set_offsets(np.array(behind_pts))

                # Update drawn centerline segments.
                cl_x_fwd = np.arange(1, 21)
                cl_z_fwd = [frame.get(f"rel_z{k}", 0.0) for k in range(1, 21)]
                centerline_line.set_data(cl_x_fwd, cl_z_fwd)

                cl_x_b = np.arange(-5, 0)
                cl_z_b = [frame.get(f"b_rel_z{k}", 0.0) for k in range(1, 6)]
                centerline_bline.set_data(cl_x_b, cl_z_b)

                # Update cast rays.
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

                # Update bottom panel metrics.
                tws = [frame.get(f"tw{j}", 0.0) for j in range(21)]
                curvs = [frame.get(f"c{j}", 0.0) for j in range(21)]
                track_width_line.set_data(np.arange(21), tws)
                curvature_line.set_data(np.arange(21), curvs)

                btws = [frame.get(f"b_tw{j}", 0.0) for j in range(1, 6)]
                bcurvs = [frame.get(f"b_c{j}", 0.0) for j in range(1, 6)]
                track_width_line_back.set_data(np.array(list(range(-5, 0))), btws)
                curvature_line_back.set_data(np.array(list(range(-5, 0))), bcurvs)

                # Update the heading difference bar.
                dh = frame.get("head_diff", 0)
                if dh >= 0:
                    heading_bar[0].set_y(0)
                    heading_bar[0].set_height(dh)
                else:
                    heading_bar[0].set_y(dh)
                    heading_bar[0].set_height(-dh)

                plt.draw()
                plt.pause(0.001)
        else:
            # Truly on-the-fly processing and visualization
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
                # Process this individual frame
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
                frame = self.processor.process_frame(sensor_data, track_data)
                
                # Update visualization with this frame
                # (Visualization code for a single frame)
                # In the relative frame the car is always at (0,0).
                car_point.set_data([0], [0])
                # The heading line is drawn along the x-axis.
                heading_line.set_data([0, heading_length], [0, 0])

                # Update forward local centerline points.
                front_pts = []
                for j in range(1, 21):
                    x_val = float(j)
                    z_val = frame.get(f"rel_z{j}", 0.0)
                    front_pts.append([x_val, z_val])
                front_scatter.set_offsets(np.array(front_pts))

                # Update behind local centerline points.
                behind_pts = []
                for j, x_val in enumerate(range(-5, 0), start=1):
                    z_val = frame.get(f"b_rel_z{j}", 0.0)
                    behind_pts.append([float(x_val), z_val])
                behind_scatter.set_offsets(np.array(behind_pts))

                # Update drawn centerline segments.
                cl_x_fwd = np.arange(1, 21)
                cl_z_fwd = [frame.get(f"rel_z{k}", 0.0) for k in range(1, 21)]
                centerline_line.set_data(cl_x_fwd, cl_z_fwd)

                cl_x_b = np.arange(-5, 0)
                cl_z_b = [frame.get(f"b_rel_z{k}", 0.0) for k in range(1, 6)]
                centerline_bline.set_data(cl_x_b, cl_z_b)

                # Update cast rays.
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

                # Update bottom panel metrics.
                tws = [frame.get(f"tw{j}", 0.0) for j in range(21)]
                curvs = [frame.get(f"c{j}", 0.0) for j in range(21)]
                track_width_line.set_data(np.arange(21), tws)
                curvature_line.set_data(np.arange(21), curvs)

                btws = [frame.get(f"b_tw{j}", 0.0) for j in range(1, 6)]
                bcurvs = [frame.get(f"b_c{j}", 0.0) for j in range(1, 6)]
                track_width_line_back.set_data(np.array(list(range(-5, 0))), btws)
                curvature_line_back.set_data(np.array(list(range(-5, 0))), bcurvs)

                # Update the heading difference bar.
                dh = frame.get("head_diff", 0)
                if dh >= 0:
                    heading_bar[0].set_y(0)
                    heading_bar[0].set_height(dh)
                else:
                    heading_bar[0].set_y(dh)
                    heading_bar[0].set_height(-dh)

                plt.draw()
                plt.pause(0.001)

            print("[Visualizer] Finished relative visualization.")

    def visualize_absolute(self, csv_path, json_path, heading_length=3.0, use_postprocessed=False):
        """
        Visualize frame-by-frame in absolute (global) coordinates.
        The scene (cones, track edges, centerline) is drawn from the JSON file,
        and the car (and its associated cast rays and local centerline points) is animated.
        """
        print("[Visualizer] Starting absolute visualization...")

        # Build track data from the JSON.
        track_data = self.processor.build_track_data(json_path)
        # Also, get the raw cone and centerline information.
        blue_cones, yellow_cones, clx, clz = parse_cone_data(json_path)
        clx_rev = clx
        clz_rev = clz
        r_clx, r_clz = resample_centerline(clx_rev, clz_rev, resolution=1.0)
        # For the absolute view, also resample the centerline in the original order.
        clx_abs, clz_abs = resample_centerline(clx, clz, resolution=1.0)
        centerline_pts_fw = list(zip(clx_abs, clz_abs))

        if use_postprocessed:
            # For postprocessed CSV, just read the data directly
            print("[Visualizer] Using postprocessed CSV data...")
            df_features = pd.read_csv(csv_path)
            frames = df_features.to_dict('records')
        else:
            # Process the data on-the-fly
            print("[Visualizer] Processing CSV data on-the-fly...")
            data = read_csv_data(csv_path)
            if data is None:
                print("Could not load CSV data.")
                return
    
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
        self.processor.reset_realtime_state()

        if use_postprocessed:
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
        else:
                        # Truly on-the-fly processing and visualization
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
                # Process this individual frame
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
                frame = self.processor.process_frame(sensor_data, track_data)
                
                # Update visualization with this frame
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

    def visualizer_inferred(self, actual_csv_path, inferred_csv_path):
        # Read the actual and inferred CSV files
        df_actual = pd.read_csv(actual_csv_path)
        df_inferred = pd.read_csv(inferred_csv_path)

        # Create subplots: one each for steering, throttle, and brake.
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Plot steering values
        axs[0].plot(df_actual['time'], df_actual['steering'], label="Actual Steering", color='blue')
        axs[0].plot(df_inferred['time'], df_inferred['steering'], label="Inferred Steering", color='red', linestyle='--')
        axs[0].set_ylabel("Steering")
        axs[0].legend()
        axs[0].grid(True)

        # Plot throttle values
        axs[1].plot(df_actual['time'], df_actual['throttle'], label="Actual Throttle", color='blue')
        axs[1].plot(df_inferred['time'], df_inferred['throttle'], label="Inferred Throttle", color='red', linestyle='--')
        axs[1].set_ylabel("Throttle")
        axs[1].legend()
        axs[1].grid(True)

        # Plot brake values
        axs[2].plot(df_actual['time'], df_actual['brake'], label="Actual Brake", color='blue')
        axs[2].plot(df_inferred['time'], df_inferred['brake'], label="Inferred Brake", color='red', linestyle='--')
        axs[2].set_ylabel("Brake")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
