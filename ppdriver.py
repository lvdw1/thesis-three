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
    def __init__(self, lookahead_distance, wheelbase,
                 speed_start=10, speed_stop=50,
                 distance_start=1.2, distance_stop=2.4):
        # The initial lookahead_distance can serve as a fallback.
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        # Parameters for dynamic adjustment of lookahead distance.
        self.speed_start = speed_start
        self.speed_stop = speed_stop
        self.distance_start = distance_start
        self.distance_stop = distance_stop

    def calculate_dynamic_lookahead(self, actual_speed):
        if actual_speed < self.speed_start:
            return self.distance_start
        elif actual_speed < self.speed_stop:
            return self.distance_start + (self.distance_stop - self.distance_start) * \
                   (actual_speed - self.speed_start) / (self.speed_stop - self.speed_start)
        else:
            return self.distance_stop

    def find_target_point(self, state, centerline, lookahead_distance):
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
            # Choose the candidate whose distance is closest to the dynamic lookahead distance.
            candidate_diffs = [abs(dist - lookahead_distance) for dist in candidate_distances]
            idx = np.argmin(candidate_diffs)
            return candidate_points[idx]
        else:
            # Fallback: return the overall closest point if no candidate is ahead.
            distances = [np.hypot(pt[0] - x_pos, pt[1] - z_pos) for pt in centerline]
            idx = np.argmin(distances)
            return centerline[idx]

    def pure_pursuit_control(self, state, centerline):
        x_pos = state["x_pos"]
        z_pos = state["z_pos"]
        
        # Convert yaw to radians.
        yaw = m.radians(state["yaw_angle"])
        # Shift position if needed (using an external function, assumed available).
        x_pos, z_pos = shift_position_single(x_pos, z_pos, state["yaw_angle"], shift_distance=1.7)
        
        # Dynamically adjust the lookahead distance based on the current speed.
        actual_speed = state.get("long_vel", 0)  # Default to 0 if not provided.
        dynamic_lookahead = self.calculate_dynamic_lookahead(actual_speed)
        
        # Find the target point from the centerline using the dynamic lookahead distance.
        target_point = self.find_target_point(state, centerline, dynamic_lookahead)
        target_x, target_z = target_point
        
        # Compute the vector from the vehicle to the target point.
        dx = target_x - x_pos
        dz = target_z - z_pos
        
        # Transform the target point into the vehicle's coordinate frame.
        x_vehicle = dx * m.cos(yaw) + dz * m.sin(yaw)
        y_vehicle = -dx * m.sin(yaw) + dz * m.cos(yaw)
        
        L_d = dynamic_lookahead

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
    track_data = processor.build_track_data("sim/tracks/validation/normal.json")
    min_curv_points = "sim/tracks/validation/normal_mincurv.json"
    with open(min_curv_points) as f:
        min_curv_points = json.load(f)
    centerline = list(zip(min_curv_points["x"], min_curv_points["y"]))
    # centerline = track_data["centerline_pts"]
    
    # Initialize Pure Pursuit Driver with dynamic lookahead parameters.
    # The base lookahead_distance is provided but will be adjusted dynamically.
    agent = PPDriver(lookahead_distance=4, wheelbase=1.5,
                     speed_start=10/3.6, speed_stop=50/3.6,
                     distance_start=1.5, distance_stop=3)

    # Create an empty list to record run data.
    run_data = []
    try:
        while True:
            # Get the current state from Unity.
            state = unity.receive_state()
            
            # Compute steering using pure pursuit and also get the target point.
            steering, target_point = agent.pure_pursuit_control(state, centerline)
            
            safety = 1.5
            throttle = (1-abs(steering))/safety
            # throttle = 0.2
            brake = 0.0
            
            # Send control command to Unity.
            unity.send_command(steering, throttle, brake)
            
            # Record data.
            run_data.append({
                "time": state["time"],
                "x_pos": state["x_pos"],
                "z_pos": state["z_pos"],
                "yaw_angle": state["yaw_angle"],
                "speed": state.get("speed", 0),
                "target_x": target_point[0],
                "target_z": target_point[1],
                "steering": steering,
                "throttle": throttle,
                "brake": brake
            })
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Terminating agent and recording data...")
    finally:
        unity.close()
