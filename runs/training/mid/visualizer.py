#test
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import math 
import matplotlib.animation as animation

def animate_run_from_csv_local(output_csv_filename, heading_length=3.0, max_ray_distance=20.0):
# Load CSV data.
# Load CSV data.
    with open(output_csv_filename, "r") as f:
        reader = csv.DictReader(f)
        frames = [row for row in reader]

    # Convert all numeric fields to floats.
    for frame in frames:
        for key in frame:
            try:
                frame[key] = float(frame[key])
            except ValueError:
                pass

    # Set up the figure with two subplots.
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 10),
                                             gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(right=0.7)

    # Set the main plot axes limits as specified.
    ax_top.set_xlim(-10, 30)
    ax_top.set_ylim(-10, 10)
    ax_top.set_aspect('equal', adjustable='box')
    ax_top.set_title("Local Frame (Car Fixed at (0,0), Heading = 0°)")
    ax_top.set_xlabel("Local X (m)")
    ax_top.set_ylabel("Local Z (m)")

    # Extra axes for heading difference and distance-to-centerline bars.
    ax_heading = fig.add_axes([0.75, 0.1, 0.1, 0.19])
    ax_heading.set_title("Heading Diff (rad)")
    ax_heading.set_ylim(-1, 1)
    ax_heading.set_xticks([])
    heading_bar = ax_heading.bar(0, 0, width=0.5, color='purple')

    # Create animated objects for the local view.
    car_point, = ax_top.plot([], [], 'ko', ms=8, label='Car')
    heading_line, = ax_top.plot([], [], 'r-', lw=2, label='Heading')
    # Scatter plots for centerline points.
    front_scatter = ax_top.scatter([], [], c='magenta', s=25, label='Front Centerline')
    behind_scatter = ax_top.scatter([], [], c='green', s=25, label='Behind Centerline')
    # Line connecting all centerline points (x = 1,2,...; z from CSV).
    centerline_line, = ax_top.plot([], [], 'k-', lw=1, label='Centerline')
    centerline_bline, = ax_top.plot([], [], 'k-', lw=1, label='Centerline')

    # Define fixed ray angles.
    yellow_angles_deg = np.arange(-20, 111, 10)
    blue_angles_deg = np.arange(20, -111, -10)
    yellow_angles = np.deg2rad(yellow_angles_deg)
    blue_angles = np.deg2rad(blue_angles_deg)
    yellow_ray_lines = [ax_top.plot([], [], color='yellow', linestyle='--', lw=1)[0]
                        for _ in range(len(yellow_angles))]
    blue_ray_lines = [ax_top.plot([], [], color='cyan', linestyle='--', lw=1)[0]
                      for _ in range(len(blue_angles))]

    # Bottom subplot: track width and curvature.
    ax_bottom.set_title("Track Width and Centerline Curvature")
    ax_bottom.set_xlabel("Local X (m)")
    ax_bottom.set_ylabel("Track Width (m)")
    ax_bottom.set_xlim(-5, 20)
    ax_bottom.set_ylim(0, 10)

    track_width_line, = ax_bottom.plot([], [], 'bo-', label='Forward Track Width')
    track_width_line_back, = ax_bottom.plot([], [], 'go-', label='Backward Track Width')
    ax_bottom.legend(loc='upper left')

    ax_curv = ax_bottom.twinx()
    curvature_line, = ax_curv.plot([], [], 'r.-', label='Forward Curvature (1/m)')
    curvature_line_back, = ax_curv.plot([], [], 'm.-', label='Backward Curvature (1/m)')
    ax_curv.set_ylim(-0.5, 0.5)
    ax_curv.legend(loc='upper right')

    def update(frame_idx):
        frame = frames[frame_idx]

        # Car remains fixed at (0,0) with heading 0.
        car_point.set_data([0], [0])
        heading_line.set_data([0, heading_length], [0, 0])

        # Extract and plot front centerline points.
        dc = frame.get("dc", 0)
        front_points = [[0,dc]]
        for i in range(1, 21):
            # The x-coordinate is simply i.
            x_val = float(i)
            # Get the z-coordinate from the CSV (use NaN if missing).
            key_z = f"rel_z{i}"
            z_val = frame.get(key_z, float("nan"))
            front_points.append([x_val, z_val])
        front_scatter.set_offsets(np.array(front_points))
        

# Extract and plot behind centerline points.
        behind_points = []
# Here we use x-coordinates from -5 to -1.
        for i, x_val in enumerate(range(-5, 0), start=1):
            key_z = f"b_rel_z{i}"
            z_val = frame.get(key_z, float("nan"))
            behind_points.append([float(x_val), z_val])
        behind_scatter.set_offsets(np.array(behind_points))

        # Update the centerline line connecting all front centerline points.
        # Use sequential x-coordinates 1,2,...,20 and the corresponding z from CSV.
        centerline_x = np.arange(1, 21)
        centerline_z = [frame.get(f"rel_z{i}", float('nan')) for i in range(1, 21)]
        centerline_line.set_data(centerline_x, centerline_z)

        centerline_bx = np.arange(-5, 0)
        centerline_bz = [frame.get(f"b_rel_z{i}", float('nan')) for i in range(1, 6)]
        centerline_bline.set_data(centerline_bx, centerline_bz)

        # Update yellow ray lines.
        for i, angle in enumerate(yellow_angles):
            key = f"yr{i+1}"
            distance = frame.get(key)
            end_x = distance * math.cos(angle)
            end_z = distance * math.sin(angle)
            yellow_ray_lines[i].set_data([0, end_x], [0, end_z])
        # Update blue ray lines.
        for i, angle in enumerate(blue_angles):
            key = f"br{i+1}"
            distance = frame.get(key)
            end_x = distance * math.cos(angle)
            end_z = distance * math.sin(angle)
            blue_ray_lines[i].set_data([0, end_x], [0, end_z])

        # Bottom subplot: plot track width and curvature using front centerline data.
        # Forward track width and curvature (for points 1 to 20).
        tws, curvs = [], []
        for i in range(0, 21):
            key_tw = f"tw{i}"
            key_c = f"c{i}"
            tws.append(frame.get(key_tw, float("nan")))
            curvs.append(frame.get(key_c, float("nan")))
        local_xs = list(range(0, 21))
        track_width_line.set_data(local_xs, tws)
        curvature_line.set_data(local_xs, curvs)

        # Backward track width and curvature (for points -5 to -1).
        btws, bcurvs = [], []
        for i in range(1, 6):
            key_tw = f"b_tw{i}"
            key_c = f"b_c{i}"
            btws.append(frame.get(key_tw, float("nan")))
            bcurvs.append(frame.get(key_c, float("nan")))
        local_xs_b = list(range(-5, 0))
        track_width_line_back.set_data(local_xs_b, btws)
        curvature_line_back.set_data(local_xs_b, bcurvs)

        # Update the heading difference bar.
        dh = frame.get("dh", 0)
        if dh >= 0:
            heading_bar[0].set_y(0)
            heading_bar[0].set_height(dh)
        else:
            heading_bar[0].set_y(dh)
            heading_bar[0].set_height(-dh)

        return (car_point, heading_line, front_scatter, behind_scatter, centerline_line, centerline_bline,
                *yellow_ray_lines, *blue_ray_lines, track_width_line, track_width_line_back, curvature_line, curvature_line_back,
                heading_bar[0])

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=20, blit=True)
    plt.show()


if __name__ == "__main__":
    animate_run_from_csv_local("session3/run1.csv")
