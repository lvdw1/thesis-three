import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import processor

class Evaluator:
    def __init__(self, cone_penalty, track_leaving_penalty, steering_jerk_penalty, car_length, car_width):
        self.cone_penalty = cone_penalty
        self.track_leaving_penalty = track_leaving_penalty
        self.steering_jerk_penalty = steering_jerk_penalty
        self.car_length = car_length
        self.car_width = car_width

    def read_csv(self, csv_path):
        df_features = pd.read_csv(csv_path)
        return df_features

    def compute_car_polygon(self, x, z, yaw):
        half_length = self.car_length / 2
        half_width = self.car_width / 2
        yaw = np.deg2rad(yaw)
        corners = np.array([
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width]
        ])
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ])
        rotated_corners = (R @ corners.T).T
        polygon = rotated_corners + np.array([x, z])
        return polygon

    def point_in_polygon(self, point, poly):
        """Determine if a point (a tuple of x and z) is inside a polygon.
           Uses the ray casting algorithm."""
        x, y = point
        inside = False
        n = len(poly)
        p1x, p1y = poly[0]
        for i in range(1, n+1):
            p2x, p2y = poly[i % n]
            if ((p1y > y) != (p2y > y)) and (x < (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x):
                inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def lap_timer(self, csv_path, json_path):
        df = self.read_csv(csv_path)
        times = df["time"]
        x_pos = df["x_pos"]
        z_pos = df["z_pos"]
        track = processor.build_track_data(json_path)
        cones_blue = track["ordered_blue"]
        start_blue = cones_blue[0] 
        cones_yellow = track["ordered_yellow"]
        start_yellow = cones_yellow[0]
        cone_locations = np.concatenate([cones_blue, cones_yellow])
        
        def orientation(p, q, r):
            """
            Returns:
              0 : p, q and r are collinear
              1 : Clockwise
              2 : Counterclockwise
            """
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < 1e-6:
                return 0
            return 1 if val > 0 else 2

        def on_segment(p, q, r):
            """Check if point r lies on segment pq."""
            if (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and 
                min(p[1], q[1]) <= r[1] <= max(p[1], q[1])):
                return True
            return False

        def segments_intersect(p1, p2, q1, q2):
            """
            Returns True if the segment p1-p2 intersects with segment q1-q2.
            """
            o1 = orientation(p1, p2, q1)
            o2 = orientation(p1, p2, q2)
            o3 = orientation(q1, q2, p1)
            o4 = orientation(q1, q2, p2)

            # General case
            if o1 != o2 and o3 != o4:
                return True

            # Special cases
            if o1 == 0 and on_segment(p1, p2, q1):
                return True
            if o2 == 0 and on_segment(p1, p2, q2):
                return True
            if o3 == 0 and on_segment(q1, q2, p1):
                return True
            if o4 == 0 and on_segment(q1, q2, p2):
                return True

            return False

        # Loop through consecutive points and check for a crossing
        lap_crossings = []

        for i in range(1, len(x_pos)):
            car_prev = (x_pos[i-1], z_pos[i-1])
            car_curr = (x_pos[i], z_pos[i])
            # Check if the car segment crosses the start-finish line
            if segments_intersect(car_prev, car_curr, start_blue, start_yellow):
                crossing_time = times[i]
                lap_crossings.append(crossing_time)

        lap_times = np.diff(lap_crossings)

        adjusted_lap_times = []
        if len(lap_times) > 0:
            for lap in lap_times:
                if lap > 80 and lap < 160:
                    adjusted_lap_times.append(lap/2)
                    adjusted_lap_times.append(lap/2)
                elif lap > 160 and lap < 240:
                    adjusted_lap_times.append(lap/3)
                    adjusted_lap_times.append(lap/3)
                    adjusted_lap_times.append(lap/3)
                else:
                    adjusted_lap_times.append(lap)

        lap_times = np.array(adjusted_lap_times)

        penalties = []
        for lap_idx in range(len(lap_crossings) - 1):
            start_time = lap_crossings[lap_idx]
            end_time = lap_crossings[lap_idx+1]
            # Get all indices corresponding to this lap segment
            lap_indices = df.index[(df["time"] >= start_time) & (df["time"] < end_time)]
            touched_cones = set()
            for idx in lap_indices:
                x = df.loc[idx, "x_pos"]
                z = df.loc[idx, "z_pos"]
                yaw_val = df.loc[idx, "yaw_angle"]
                # Compute the car's polygon at this timestep
                car_poly = self.compute_car_polygon(x, z, yaw_val)
                # Check each cone for a collision if not already detected this lap
                for cone_idx, cone in enumerate(cone_locations):
                    if cone_idx in touched_cones:
                        continue
                    if self.point_in_polygon(cone, car_poly):
                        touched_cones.add(cone_idx)
            # Each detected cone adds a penalty of 2 seconds
            penalty_time = len(touched_cones) * self.cone_penalty
            penalties.append(penalty_time)
        corrected_lap_times = np.array(lap_times + penalties)
        return lap_times, corrected_lap_times

    def plot_animation(self, csv_path, json_path):
        """
        Plots an animation of the car (as a polygon) driving in between the cones.
        Every frame corresponds to a timestamp from the x, z coordinates.
        The cones (blue and yellow) remain stationary.
        """
        # Load CSV data
        df = self.read_csv(csv_path)
        times = df["time"].values
        x_pos = df["x_pos"].values
        z_pos = df["z_pos"].values
        yaw_angles = df["yaw_angle"].values

        # Load track data (cones)
        track = processor.build_track_data(json_path)
        cones_blue = np.array(track["ordered_blue"])
        cones_yellow = np.array(track["ordered_yellow"])

        # Create a figure and axis for the plot
        fig, ax = plt.subplots()

        # Plot stationary cones
        ax.scatter(cones_blue[:, 0], cones_blue[:, 1], c="blue", label="Blue Cones")
        ax.scatter(cones_yellow[:, 0], cones_yellow[:, 1], c="yellow", label="Yellow Cones")

        # Set equal aspect ratio and adjust limits with some padding
        ax.set_aspect('equal')
        padding = 5
        ax.set_xlim(np.min(x_pos) - padding, np.max(x_pos) + padding)
        ax.set_ylim(np.min(z_pos) - padding, np.max(z_pos) + padding)
        ax.set_title("Car Driving Animation")
        ax.legend()

        # Create the initial car polygon patch using the first position
        initial_poly = self.compute_car_polygon(x_pos[0], z_pos[0], yaw_angles[0])
        car_patch = plt.Polygon(initial_poly, closed=True, color="red", alpha=0.5)
        ax.add_patch(car_patch)

        # Update function for the animation: update the car polygon for each frame
        def update(frame):
            new_poly = self.compute_car_polygon(x_pos[frame], z_pos[frame], yaw_angles[frame])
            car_patch.set_xy(new_poly)
            return [car_patch]

        # Create the animation (each frame corresponds to one timestamp)
        anim = FuncAnimation(fig, update, frames=len(x_pos), interval=10, blit=True, repeat=True)
        plt.show()

# Example usage:
fs_eval = Evaluator(2, 10, 1, 2, 1)
processor = processor.Processor()

def plot(tracks):
    avg_lap_times = []
    std_lap_times = []
    avg_corr_times = []
    std_corr_times = []

    for track in tracks:
        # Adjust paths as needed to match your directory structure
        csv_path = f"runs/processed/final/{track}.csv"
        json_path = f"sim/tracks/{track}.json"
        lap_times, corrected_lap_times = fs_eval.lap_timer(csv_path, json_path)
        # Compute average and standard deviation for lap times and corrected lap times
        avg_lap_times.append(np.mean(lap_times))
        std_lap_times.append(np.std(lap_times))
        avg_corr_times.append(np.mean(corrected_lap_times))
        std_corr_times.append(np.std(corrected_lap_times))

# Set up the grouped bar plot
    x = np.arange(len(tracks))  # x locations for the tracks
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, avg_lap_times, width, yerr=std_lap_times, capsize=5, label='Average Lap Time')
    bars2 = ax.bar(x + width/2, avg_corr_times, width, yerr=std_corr_times, capsize=5, label='Corrected Lap Time')

    ax.set_xlabel('Tracks')
    ax.set_ylabel('Time (s)')
    ax.set_title('Average Lap Times and Corrected Lap Times per Track')
    ax.set_xticks(x)
    ax.set_xticklabels(tracks)
    ax.legend()

    plt.tight_layout()
    plt.show()

# lap_time, corrected_lap_time = fs_eval.lap_timer("runs/processed/final/track11.csv", "sim/tracks/track11.json")
# print(lap_time, corrected_lap_time)

fs_eval.plot_animation("runs/processed/final/track11.csv", "sim/tracks/track11.json")
# tracks = ['track1', 'track2', 'track3', 'track4', 'track5', 'track6', 'track7', 'track8', 'track9', 'track10', 'track11', 'track13', 'track14', 'track15']
# plot(tracks)
