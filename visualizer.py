import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

def read_csv_data(file_path):
    """
    Reads CSV data and extracts car location, heading, recognized cone locations, and control inputs.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list of dict: List of rows containing car, heading, cone data, and control inputs.
    """
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                car_x = float(row["X Position"])
                car_z = float(row["Z Position"])
                yaw = float(row["Yaw"])  # Car's heading (yaw)
                next_blue_cones = eval(row["Next Blue Cones"])  # Parse stringified list
                next_yellow_cones = eval(row["Next Yellow Cones"])  # Parse stringified list
                steering_angle = float(row["Steering Angle"])
                throttle = float(row["Throttle"])
                brake = float(row["Brake"])

                data.append({
                    "car_position": (car_x, car_z),
                    "yaw": yaw,
                    "blue_cones": next_blue_cones,
                    "yellow_cones": next_yellow_cones,
                    "steering_angle": steering_angle,
                    "throttle": throttle,
                    "brake": brake,
                })
            except Exception as e:
                print(f"Error parsing row: {row}. Error: {e}")
    return data

def rotate_point(x, z, angle):
    """
    Rotates a point around the origin by a given angle.

    Args:
        x (float): X-coordinate of the point.
        z (float): Z-coordinate of the point.
        angle (float): Rotation angle in degrees.

    Returns:
        tuple: Rotated (x, z) coordinates.
    """
    rad = math.radians(angle)
    cos_angle = math.cos(rad)
    sin_angle = math.sin(rad)
    x_rot = cos_angle * x + sin_angle * z
    z_rot = -sin_angle * x + cos_angle * z
    return x_rot, z_rot

def transform_cones(cones, car_position, yaw):
    """
    Transforms cone coordinates to be relative to the car's position and heading.

    Args:
        cones (list of tuple): Original cone positions.
        car_position (tuple): Car's current position (x, z).
        yaw (float): Car's heading in degrees.

    Returns:
        list of tuple: Transformed cone positions relative to the car.
    """
    transformed = []
    car_x, car_z = car_position
    for cone_x, cone_z in cones:
        # Translate cones relative to car's position
        dx = cone_x - car_x
        dz = cone_z - car_z

        # Rotate cones relative to car's heading
        x_rel, z_rel = rotate_point(dx, dz, -yaw)
        transformed.append((x_rel, z_rel))
    return transformed

def visualize_data(data):
    """
    Visualizes car location, heading, recognized cones, and control inputs using animated plots.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Plot 1: Car and cone visualization
    car_point, = ax1.plot(0, 0, 'ro', label="Car (stationary)", markersize=8)  # Car stays at (0, 0)
    blue_cones_plot, = ax1.plot([], [], 'bo', label="Blue Cones")
    yellow_cones_plot, = ax1.plot([], [], 'yo', label="Yellow Cones")
    car_heading, = ax1.plot([], [], 'g-', label="Car Heading")

    ax1.set_xlim(-20, 20)
    ax1.set_ylim(-20, 20)
    ax1.set_xlabel("Relative X Position")
    ax1.set_ylabel("Relative Z Position")
    ax1.set_title("Cone Positions Relative to Car")
    ax1.legend()

    # Plot 2: Control inputs (steering, throttle, brake)
    ax2.set_xlim(0, len(data))  # Frame count on the x-axis
    ax2.set_ylim(-1.1, 1.1)  # Normalized range for control inputs
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Control Values")
    ax2.set_title("Control Inputs Over Time")
    ax2.grid(True)
    line_steering, = ax2.plot([], [], label="Steering Angle", color="blue")
    line_throttle, = ax2.plot([], [], label="Throttle", color="green")
    line_brake, = ax2.plot([], [], label="Brake", color="red")
    ax2.legend()

    # Initialize data storage for Plot 2
    frames = []
    steering_values = []
    throttle_values = []
    brake_values = []

    def update(frame):
        # Update Plot 1: Cones relative to stationary car
        if frame >= len(data):
            return blue_cones_plot, yellow_cones_plot, car_heading, line_steering, line_throttle, line_brake

        car_position = data[frame]["car_position"]
        yaw = data[frame]["yaw"]
        blue_cones = data[frame]["blue_cones"]
        yellow_cones = data[frame]["yellow_cones"]
        steering_angle = data[frame]["steering_angle"]
        throttle = data[frame]["throttle"]
        brake = data[frame]["brake"]

        # Transform cones to be relative to the car
        rel_blue_cones = transform_cones(blue_cones, car_position, yaw) if blue_cones else []
        rel_yellow_cones = transform_cones(yellow_cones, car_position, yaw) if yellow_cones else []

        # Update blue cones
        if rel_blue_cones:
            blue_x, blue_y = zip(*rel_blue_cones)
            blue_cones_plot.set_data(blue_x, blue_y)
        else:
            blue_cones_plot.set_data([], [])

        # Update yellow cones
        if rel_yellow_cones:
            yellow_x, yellow_y = zip(*rel_yellow_cones)
            yellow_cones_plot.set_data(yellow_x, yellow_y)
        else:
            yellow_cones_plot.set_data([], [])

        # Heading arrow (stationary car at origin)
        heading_length = 5  # Length of the arrow

        # Use -yaw for the arrow to match cone rotation logic
        heading_x = heading_length * math.cos(math.radians(yaw*2))
        heading_y = heading_length * math.sin(math.radians(yaw*2))

        # Update the car's heading line to point in the correct direction
        car_heading.set_data([0, heading_x], [0, heading_y])

        # Update Plot 2: Control inputs
        frames.append(frame)
        steering_values.append(steering_angle)
        throttle_values.append(throttle)
        brake_values.append(brake)

        line_steering.set_data(frames, steering_values)
        line_throttle.set_data(frames, throttle_values)
        line_brake.set_data(frames, brake_values)

        return blue_cones_plot, yellow_cones_plot, car_heading, line_steering, line_throttle, line_brake

    ani = FuncAnimation(fig, update, frames=len(data), interval=20, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Replace with your CSV file path
    csv_file_path = "car_data_run_nn_v001.csv"

    # Read data
    car_and_cone_data = read_csv_data(csv_file_path)

    # Visualize data
    visualize_data(car_and_cone_data)
