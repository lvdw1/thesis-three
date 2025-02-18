import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to your CSV file
CSV_FILE_PATH = "car_data_run_corrected.csv"


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

        # Identify and sort columns based on naming patterns
        centerline_x_cols = sorted([c for c in fieldnames if c.startswith("centerline_x")],
                                   key=lambda x: int(x[len("centerline_x"):]))
        centerline_z_cols = sorted([c for c in fieldnames if c.startswith("centerline_z")],
                                   key=lambda x: int(x[len("centerline_z"):]))
        curvature_cols = sorted([c for c in fieldnames if c.startswith("curvature_")],
                                key=lambda x: int(x[len("curvature_"):]))
        left_ray_cols = sorted([c for c in fieldnames if c.startswith("left_ray_")],
                               key=lambda x: int(x[len("left_ray_"):-3]))
        right_ray_cols = sorted([c for c in fieldnames if c.startswith("right_ray_")],
                                key=lambda x: int(x[len("right_ray_"):-3]))
        left_edge_x_cols = sorted([c for c in fieldnames if c.startswith("left_edge_x")],
                                  key=lambda x: int(x[len("left_edge_x"):]))
        left_edge_z_cols = sorted([c for c in fieldnames if c.startswith("left_edge_z")],
                                  key=lambda x: int(x[len("left_edge_z"):]))
        right_edge_x_cols = sorted([c for c in fieldnames if c.startswith("right_edge_x")],
                                   key=lambda x: int(x[len("right_edge_x"):]))
        right_edge_z_cols = sorted([c for c in fieldnames if c.startswith("right_edge_z")],
                                   key=lambda x: int(x[len("right_edge_z"):]))

        # Initialize lists to store data
        times = []
        long_vel = []
        lat_vel = []
        yaw_rate = []
        long_acc = []
        lat_acc = []
        dist_centerline = []
        rel_heading_angle = []
        left_track_width = []
        right_track_width = []
        centerline_x_data = []
        centerline_z_data = []
        curvatures_data = []
        left_rays_data = []
        right_rays_data = []
        left_edge_x_data = []
        left_edge_z_data = []
        right_edge_x_data = []
        right_edge_z_data = []

        for row in reader:
            try:
                # Parse basic scalar values
                times.append(float(row["time"]))
                long_vel.append(float(row["long_vel"]))
                lat_vel.append(float(row["lat_vel"]))
                yaw_rate.append(float(row["yaw_rate"]))
                long_acc.append(float(row["long_acc"]))
                lat_acc.append(float(row["lat_acc"]))
                dist_centerline.append(float(row["dist_centerline"]))
                rel_heading_angle.append(float(row["rel_heading_angle"]))
                left_track_width.append(float(row["left_track_width"]))
                right_track_width.append(float(row["right_track_width"]))

                # Parse centerline points
                cx = [float(row[c]) for c in centerline_x_cols]
                cz = [float(row[c]) for c in centerline_z_cols]
                centerline_x_data.append(cx)
                centerline_z_data.append(cz)

                # Parse curvatures
                curvs = [float(row[c]) for c in curvature_cols]
                curvatures_data.append(curvs)

                # Parse ray distances
                lr = [float(row[c]) for c in left_ray_cols]
                rr = [float(row[c]) for c in right_ray_cols]
                left_rays_data.append(lr)
                right_rays_data.append(rr)

                # Parse track edges
                lex = [float(row[c]) for c in left_edge_x_cols]
                lez = [float(row[c]) for c in left_edge_z_cols]
                rex = [float(row[c]) for c in right_edge_x_cols]
                rez = [float(row[c]) for c in right_edge_z_cols]
                left_edge_x_data.append(lex)
                left_edge_z_data.append(lez)
                right_edge_x_data.append(rex)
                right_edge_z_data.append(rez)

            except Exception as e:
                logging.warning(f"Error parsing row: {row} - {e}")

        # Check if data was loaded
        if not times:
            logging.error("No data loaded from CSV.")
            return None

        # Convert lists to numpy arrays for efficiency
        return {
            "time": np.array(times),
            "long_vel": np.array(long_vel),
            "lat_vel": np.array(lat_vel),
            "yaw_rate": np.array(yaw_rate),
            "long_acc": np.array(long_acc),
            "lat_acc": np.array(lat_acc),
            "dist_centerline": np.array(dist_centerline),
            "rel_heading_angle": np.array(rel_heading_angle),
            "left_track_width": np.array(left_track_width),
            "right_track_width": np.array(right_track_width),
            "centerline_x": np.array(centerline_x_data),  # shape: (frames, 10)
            "centerline_z": np.array(centerline_z_data),
            "curvatures": np.array(curvatures_data),
            "left_rays": np.array(left_rays_data),
            "right_rays": np.array(right_rays_data),
            "left_edge_x": np.array(left_edge_x_data),
            "left_edge_z": np.array(left_edge_z_data),
            "right_edge_x": np.array(right_edge_x_data),
            "right_edge_z": np.array(right_edge_z_data),
            "centerline_count": len(centerline_x_cols),
            "curvature_count": len(curvature_cols),
            "left_ray_count": len(left_ray_cols),
            "right_ray_count": len(right_ray_cols),
            "left_edge_count": len(left_edge_x_cols),
            "right_edge_count": len(right_edge_x_cols)
        }

def visualize_data(data):
    """
    Visualizes the logged data using Matplotlib animations.
    """
    if data is None:
        logging.error("No data available for visualization.")
        return

    # Unpack data
    time_array = data["time"]
    long_vel = data["long_vel"]
    lat_vel = data["lat_vel"]
    yaw_rate = data["yaw_rate"]
    long_acc = data["long_acc"]
    lat_acc = data["lat_acc"]
    dist_centerline = data["dist_centerline"]
    rel_heading_angle = data["rel_heading_angle"]
    left_track_width = data["left_track_width"]
    right_track_width = data["right_track_width"]
    centerline_x = data["centerline_x"]
    centerline_z = data["centerline_z"]
    curvatures = data["curvatures"]
    left_rays = data["left_rays"]
    right_rays = data["right_rays"]
    left_edge_x = data["left_edge_x"]
    left_edge_z = data["left_edge_z"]
    right_edge_x = data["right_edge_x"]
    right_edge_z = data["right_edge_z"]

    n_frames = len(time_array)
    centerline_count = centerline_x.shape[1]
    curvature_count = curvatures.shape[1]
    left_ray_count = left_rays.shape[1]
    right_ray_count = right_rays.shape[1]
    left_edge_count = left_edge_x.shape[1]
    right_edge_count = right_edge_x.shape[1]

    # Ray angles:
    # Assuming left rays are from -20° to 110° in 10° increments
    # and right rays are from -110° to 20° in 10° increments
    left_ray_angles = np.arange(-20, 111, 10)  # -20 to 110 degrees
    right_ray_angles = np.arange(-110, 21, 10)  # -110 to 20 degrees

    # Convert degrees to radians for polar plot
    left_ray_angles_rad = np.radians(left_ray_angles)
    right_ray_angles_rad = np.radians(right_ray_angles)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 25))
    gs = fig.add_gridspec(5, 3, height_ratios=[3, 2, 2, 2, 2])

    # Environment Plot (Top-Down View)
    ax_env = fig.add_subplot(gs[0, :])
    ax_env.set_title("Environment View (Relative to Car)")
    ax_env.set_xlabel("X (m)")
    ax_env.set_ylabel("Z (m)")
    ax_env.set_xlim(-30, 30)
    ax_env.set_ylim(-30, 30)
    ax_env.grid(True)

    # Car is always at (0,0)
    car_point, = ax_env.plot(0, 0, 'ro', label="Car")

    # Centerline Line
    cl_line, = ax_env.plot([], [], 'k-', label="Centerline")

    # Track Edges
    left_edge_line, = ax_env.plot([], [], 'b--', label="Left Track Edge")
    right_edge_line, = ax_env.plot([], [], 'g--', label="Right Track Edge")

    # Rays Lines
    ray_lines = []
    for _ in range(left_ray_count + right_ray_count):
        line, = ax_env.plot([], [], 'c-', alpha=0.6)
        ray_lines.append(line)

    # Cones
    blue_cones_plot, = ax_env.plot([], [], 'bo', label="Blue Cones (Left Edge)")
    yellow_cones_plot, = ax_env.plot([], [], 'yo', label="Yellow Cones (Right Edge)")

    ax_env.legend(loc='upper right')

    # Velocities and Yaw Rate Plot
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_vel.set_title("Velocities & Yaw Rate Over Time")
    ax_vel.set_xlabel("Time (s)")
    ax_vel.set_ylabel("Value")
    ax_vel.grid(True)
    line_long_vel, = ax_vel.plot([], [], label="Longitudinal Velocity", color="blue")
    line_lat_vel, = ax_vel.plot([], [], label="Lateral Velocity", color="green")
    line_yaw_rate, = ax_vel.plot([], [], label="Yaw Rate", color="purple")
    ax_vel.legend(loc='upper right')
    vert_line_vel = ax_vel.axvline(x=0, color='r', linestyle='--')

    # Accelerations Plot
    ax_acc = fig.add_subplot(gs[1, 1])
    ax_acc.set_title("Accelerations Over Time")
    ax_acc.set_xlabel("Time (s)")
    ax_acc.set_ylabel("Acceleration (m/s²)")
    ax_acc.grid(True)
    line_long_acc, = ax_acc.plot([], [], label="Longitudinal Acceleration", color="blue")
    line_lat_acc, = ax_acc.plot([], [], label="Lateral Acceleration", color="green")
    ax_acc.legend(loc='upper right')
    vert_line_acc = ax_acc.axvline(x=0, color='r', linestyle='--')

    # Distance to Centerline & Relative Heading Plot
    ax_dist_heading = fig.add_subplot(gs[1, 2])
    ax_dist_heading.set_title("Distance to Centerline & Relative Heading")
    ax_dist_heading.set_xlabel("Time (s)")
    ax_dist_heading.set_ylabel("Distance (m) / Angle (deg)")
    ax_dist_heading.grid(True)
    line_dist_centerline, = ax_dist_heading.plot([], [], label="Distance to Centerline", color="blue")
    line_rel_heading, = ax_dist_heading.plot([], [], label="Relative Heading Angle", color="orange")
    ax_dist_heading.legend(loc='upper right')
    vert_line_dist = ax_dist_heading.axvline(x=0, color='r', linestyle='--')

    # Track Widths Plot
    ax_widths = fig.add_subplot(gs[2, 0])
    ax_widths.set_title("Track Widths Over Time")
    ax_widths.set_xlabel("Time (s)")
    ax_widths.set_ylabel("Width (m)")
    ax_widths.grid(True)
    line_left_width, = ax_widths.plot([], [], label="Left Track Width", color="blue")
    line_right_width, = ax_widths.plot([], [], label="Right Track Width", color="green")
    ax_widths.legend(loc='upper right')
    vert_line_width = ax_widths.axvline(x=0, color='r', linestyle='--')

    # Curvatures Plot
    ax_curv = fig.add_subplot(gs[2, 1])
    ax_curv.set_title("Centerline Curvatures at Current Frame")
    ax_curv.set_xlabel("Centerline Segment #")
    ax_curv.set_ylabel("Curvature (1/m)")
    ax_curv.set_xticks(np.arange(1, curvature_count + 1))
    ax_curv.grid(True)
    line_curvatures, = ax_curv.plot([], [], 'o-', color='purple', label="Curvatures")
    ax_curv.legend(loc='upper right')

    # Rays Polar Plot
    ax_rays_polar = fig.add_subplot(gs[2, 2], polar=True)
    ax_rays_polar.set_title("Ray Distances (Polar)")
    ray_polar_line_left, = ax_rays_polar.plot([], [], 'b-o', label="Left Rays")
    ray_polar_line_right, = ax_rays_polar.plot([], [], 'g-o', label="Right Rays")
    ax_rays_polar.legend(loc='upper right')

    # Curvatures Bar Plot
    ax_curv_bar = fig.add_subplot(gs[3, 0])
    ax_curv_bar.set_title("Curvature Distribution at Current Frame")
    ax_curv_bar.set_xlabel("Centerline Segment #")
    ax_curv_bar.set_ylabel("Curvature (1/m)")
    ax_curv_bar.set_xticks(np.arange(1, curvature_count + 1))
    ax_curv_bar.grid(True)
    bars = ax_curv_bar.bar([], [], color='purple')

    # Time-Series Data for Animation
    # (These lists are no longer needed as we're using numpy arrays)

    def init():
        # Initialize all plots
        cl_line.set_data([], [])
        left_edge_line.set_data([], [])
        right_edge_line.set_data([], [])
        for line in ray_lines:
            line.set_data([], [])

        # Initialize Cones
        blue_cones_plot.set_data([], [])
        yellow_cones_plot.set_data([], [])

        # Initialize Time-Series Lines
        line_long_vel.set_data([], [])
        line_lat_vel.set_data([], [])
        line_yaw_rate.set_data([], [])
        line_long_acc.set_data([], [])
        line_lat_acc.set_data([], [])
        line_dist_centerline.set_data([], [])
        line_rel_heading.set_data([], [])
        line_left_width.set_data([], [])
        line_right_width.set_data([], [])

        # Initialize Curvatures
        line_curvatures.set_data([], [])

        # Initialize Rays Polar Plot
        ray_polar_line_left.set_data([], [])
        ray_polar_line_right.set_data([], [])

        # Initialize Curvature Bar Plot
        ax_curv_bar.set_xlim(0.5, curvature_count + 0.5)
        ax_curv_bar.set_ylim(np.min(curvatures) - 0.1, np.max(curvatures) + 0.1)
        for bar in bars:
            bar.set_height(0)

        return (cl_line, left_edge_line, right_edge_line,
                blue_cones_plot, yellow_cones_plot,
                line_long_vel, line_lat_vel, line_yaw_rate,
                line_long_acc, line_lat_acc,
                line_dist_centerline, line_rel_heading,
                line_left_width, line_right_width,
                line_curvatures,
                ray_polar_line_left, ray_polar_line_right, *bars)

    def update(frame):
        if frame >= n_frames:
            logging.info("Reached end of data frames.")
            return

        # Update Environment Plot
        # Centerline Points
        cx = centerline_x[frame]
        cz = centerline_z[frame]
        cl_line.set_data(cx, cz)

        # Track Edges
        lex = left_edge_x[frame]
        lez = left_edge_z[frame]
        left_edge_line.set_data(lex, lez)

        rex = right_edge_x[frame]
        rez = right_edge_z[frame]
        right_edge_line.set_data(rex, rez)

        # Update Cones
        # Blue Cones
        blue_cones_rel = centerline_x[frame]  # Replace with actual blue cones data if available
        # Assuming 'left_edge_x' and 'left_edge_z' represent transformed blue cones
        blue_x = data["left_edge_x"][frame]
        blue_z = data["left_edge_z"][frame]
        blue_cones_plot.set_data(blue_x, blue_z)

        # Yellow Cones
        yellow_cones_rel = centerline_z[frame]  # Replace with actual yellow cones data if available
        # Assuming 'right_edge_x' and 'right_edge_z' represent transformed yellow cones
        yellow_x = data["right_edge_x"][frame]
        yellow_z = data["right_edge_z"][frame]
        yellow_cones_plot.set_data(yellow_x, yellow_z)

        # Update Rays in Environment
        for i in range(left_ray_count):
            angle_deg = left_ray_angles[i]
            angle_rad = math.radians(angle_deg)
            dist = left_rays[frame][i]
            x_end = dist * math.cos(angle_rad)
            z_end = dist * math.sin(angle_rad)
            ray_lines[i].set_data([0, x_end], [0, z_end])

        for j in range(right_ray_count):
            angle_deg = right_ray_angles[j]
            angle_rad = math.radians(angle_deg)
            dist = right_rays[frame][j]
            x_end = dist * math.cos(angle_rad)
            z_end = dist * math.sin(angle_rad)
            ray_lines[left_ray_count + j].set_data([0, x_end], [0, z_end])

        # Update Velocities and Yaw Rate Plot
        line_long_vel.set_data(time_array[:frame + 1], long_vel[:frame + 1])
        line_lat_vel.set_data(time_array[:frame + 1], lat_vel[:frame + 1])
        line_yaw_rate.set_data(time_array[:frame + 1], yaw_rate[:frame + 1])

        # Update Accelerations Plot
        line_long_acc.set_data(time_array[:frame + 1], long_acc[:frame + 1])
        line_lat_acc.set_data(time_array[:frame + 1], lat_acc[:frame + 1])

        # Update Distance to Centerline & Relative Heading Plot
        line_dist_centerline.set_data(time_array[:frame + 1], dist_centerline[:frame + 1])
        line_rel_heading.set_data(time_array[:frame + 1], rel_heading_angle[:frame + 1])

        # Update Track Widths Plot
        line_left_width.set_data(time_array[:frame + 1], left_track_width[:frame + 1])
        line_right_width.set_data(time_array[:frame + 1], right_track_width[:frame + 1])

        # Update Vertical Lines to Current Time
        current_time = time_array[frame]
        vert_line_vel.set_xdata([current_time, current_time])
        vert_line_acc.set_xdata([current_time, current_time])
        vert_line_dist.set_xdata([current_time, current_time])
        vert_line_width.set_xdata([current_time, current_time])

        # Update Curvatures Plot
        curr_curvs = curvatures[frame]
        x_curv = np.arange(1, curvature_count + 1)
        line_curvatures.set_data(x_curv, curr_curvs)
        ax_curv.set_ylim(np.min(curr_curvs) - 0.1, np.max(curr_curvs) + 0.1)

        # Update Rays Polar Plot
        ray_polar_line_left.set_data(left_ray_angles_rad, left_rays[frame])
        ray_polar_line_right.set_data(right_ray_angles_rad, right_rays[frame])
        max_ray = max(np.max(left_rays[frame]), np.max(right_rays[frame]), 1)
        ax_rays_polar.set_ylim(0, max_ray * 1.1)

        # Update Curvature Bar Plot
        for idx, bar in enumerate(bars):
            if idx < curvature_count:
                bar.set_height(curvatures[frame][idx])

        return (cl_line, left_edge_line, right_edge_line,
                blue_cones_plot, yellow_cones_plot,
                line_long_vel, line_lat_vel, line_yaw_rate,
                line_long_acc, line_lat_acc,
                line_dist_centerline, line_rel_heading,
                line_left_width, line_right_width,
                line_curvatures,
                ray_polar_line_left, ray_polar_line_right, *bars)

    # Initialize animation
    def animate():
        # Read data from CSV
        data = read_csv_data(CSV_FILE_PATH)

        if data is None:
            logging.error("No data to visualize. Exiting visualization.")
            return

        # Unpack data
        global time_array, long_vel, lat_vel, yaw_rate, long_acc, lat_acc
        global dist_centerline, rel_heading_angle, left_track_width, right_track_width
        global centerline_x, centerline_z, curvatures, left_rays, right_rays
        global left_edge_x, left_edge_z, right_edge_x, right_edge_z

        time_array = data["time"]
        long_vel = data["long_vel"]
        lat_vel = data["lat_vel"]
        yaw_rate = data["yaw_rate"]
        long_acc = data["long_acc"]
        lat_acc = data["lat_acc"]
        dist_centerline = data["dist_centerline"]
        rel_heading_angle = data["rel_heading_angle"]
        left_track_width = data["left_track_width"]
        right_track_width = data["right_track_width"]
        centerline_x = data["centerline_x"]
        centerline_z = data["centerline_z"]
        curvatures = data["curvatures"]
        left_rays = data["left_rays"]
        right_rays = data["right_rays"]
        left_edge_x = data["left_edge_x"]
        left_edge_z = data["left_edge_z"]
        right_edge_x = data["right_edge_x"]
        right_edge_z = data["right_edge_z"]

        n_frames = len(time_array)
        logging.info(f"Loaded {n_frames} frames from CSV.")

        # Initialize FuncAnimation
        ani = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                            interval=50, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()

    animate()

if __name__ == "__main__":
    visualize_data(read_csv_data(CSV_FILE_PATH))
