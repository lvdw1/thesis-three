import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as m
import time
import socket
import subprocess
import matplotlib.animation as animation

from rldriver import *

class PPDriver:
    def __init__(self, lookahead_distance, wheelbase):
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase

    def find_target_point(self, state, centerline):
        x_pos = state["x_pos"]
        z_pos = state["z_pos"]
        # Convert yaw angle (in degrees) to radians for computing heading.
        yaw = m.radians(state["yaw_angle"])
        heading = np.array([m.cos(yaw), m.sin(yaw)])
        car_pos = np.array([x_pos, z_pos])
        
        candidate_points = []
        candidate_distances = []
        
        # Evaluate each point in the centerline.
        for pt in centerline:
            pt_arr = np.array([pt[0], pt[1]])
            vec = pt_arr - car_pos
            # Check if the point is ahead using the dot product.
            if np.dot(vec, heading) > 0:
                distance = np.linalg.norm(vec)
                candidate_points.append(pt)
                candidate_distances.append(distance)
        
        if candidate_points:
            # Choose the candidate whose distance is closest to the lookahead distance.
            candidate_diffs = [abs(dist - self.lookahead_distance) for dist in candidate_distances]
            idx = np.argmin(candidate_diffs)
            return candidate_points[idx]
        else:
            # Fallback: return the overall closest point if no candidate is ahead.
            distances = [np.hypot(pt[0] - x_pos, pt[1] - z_pos) for pt in centerline]
            idx = np.argmin(distances)
            return centerline[idx]

    def pure_pursuit_control(self, state, centerline):
        """
        Computes the steering angle using the pure pursuit algorithm.
        state: dict containing current vehicle state.
        Returns a tuple: (normalized steering command in [-1, 1], target point).
        """
        x_pos = state["x_pos"]
        z_pos = state["z_pos"]
        
        yaw = m.radians(state["yaw_angle"])
        x_pos, z_pos = shift_position_single(x_pos, z_pos, state["yaw_angle"], shift_distance = 1.7)
        
        # Find the target point from the centerline.
        target_point = self.find_target_point(state, centerline)
        target_x, target_z = target_point
        
        # Compute the vector from the vehicle to the target point.
        dx = target_x - x_pos
        dz = target_z - z_pos
        
        # Transform the target point into the vehicle's coordinate frame.
        x_vehicle = dx * m.cos(yaw) + dz * m.sin(yaw)
        y_vehicle = -dx * m.sin(yaw) + dz * m.cos(yaw)
        
        L_d = self.lookahead_distance

        # Pure pursuit control law: delta = arctan(2 * wheelbase * y_vehicle / L_d^2)
        steering_angle = m.atan2(2 * self.wheelbase * y_vehicle, L_d**2)
        steering_deg = m.degrees(steering_angle)
        # Clip to [-45, 45] degrees and normalize to [-1, 1]
        steering = np.clip(steering_deg, -45, 45)
        return -steering/45.0, target_point

if __name__ == "__main__":
    # Initialize Unity environment.
    unity = UnityEnv(host='127.0.0.1', port=65432)

    processor = Processor()
    track_data = processor.build_track_data("sim/tracks/track17.json")
    centerline = track_data["centerline_pts"]
    
    # Initialize Pure Pursuit Driver.
    agent = PPDriver(lookahead_distance=5, wheelbase=1.5)
    
    # Create an empty list to record run data.
    run_data = []
    
    print("Starting simulation run. Press Ctrl+C to stop and save data.")
    try:
        while True:
            # Get the current state from Unity.
            state = unity.receive_state()
            
            # Compute steering using pure pursuit and also get the target point.
            steering, target_point = agent.pure_pursuit_control(state, centerline)
            
            # Define throttle and brake (using a fixed throttle for this example).
            throttle = 0.5
            brake = 0.0
            
            # Send control command to Unity.
            unity.send_command(steering, throttle, brake)
            
            # Record data.
            run_data.append({
                "time": state["time"],
                "x_pos": state["x_pos"],
                "z_pos": state["z_pos"],
                "yaw_angle": state["yaw_angle"],
                "target_x": target_point[0],
                "target_z": target_point[1],
                "steering": steering,
                "throttle": throttle,
                "brake": brake
            })
            
    except KeyboardInterrupt:
        print("Terminating agent and recording data...")
    finally:
        unity.close()
    
    # # Convert recorded data to a DataFrame and write to CSV.
    # df = pd.DataFrame(run_data)
    # csv_filename = "run_data.csv"
    # df.to_csv(csv_filename, index=False)
    # print(f"Run data written to {csv_filename}")
    #
    # # Create an animated plot of the recorded data.
    # fig, ax = plt.subplots(figsize=(10, 8))
    # arrow_length = 2.0  # Adjust arrow length as needed.
    # centerline_x = [pt[0] for pt in centerline]
    # centerline_z = [pt[1] for pt in centerline]
    #
    # def animate(i):
    #     ax.clear()
    #     # Plot centerline.
    #     ax.plot(centerline_x, centerline_z, 'ko', markersize=3, label="Centerline")
    #     
    #     # Plot car trajectory up to current frame.
    #     ax.plot(df["x_pos"].iloc[:i+1], df["z_pos"].iloc[:i+1], 'b-', label="Car Trajectory")
    #     
    #     # Plot current car position.
    #     car_x = df.iloc[i]["x_pos"]
    #     car_z = df.iloc[i]["z_pos"]
    #     ax.plot(car_x, car_z, 'bo', markersize=8, label="Car")
    #     
    #     # Plot current target point.
    #     target_x = df.iloc[i]["target_x"]
    #     target_z = df.iloc[i]["target_z"]
    #     ax.plot(target_x, target_z, 'rx', markersize=10, label="Target Point")
    #     
    #     # Draw heading arrow.
    #     yaw = m.radians(df.iloc[i]["yaw_angle"])
    #     dx = m.cos(yaw) * arrow_length
    #     dz = m.sin(yaw) * arrow_length
    #     ax.arrow(car_x, car_z, dx, dz, head_width=0.5, head_length=0.5, fc='blue', ec='blue')
    #     
    #     ax.set_xlabel("X Position")
    #     ax.set_ylabel("Z Position")
    #     ax.set_title(f"Recorded Run Animation - Frame {i}")
    #     ax.legend()
    #     ax.grid(True)
    #
    # ani = animation.FuncAnimation(fig, animate, frames=len(df), interval=200, repeat=False)
    # plt.show()
