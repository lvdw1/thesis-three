import matplotlib.pyplot as plt
import json

def parse_cone_data(json_file_path):
    """
    Parses a .json file containing cone data and organizes them by color.

    Args:
        json_file_path (str): Path to the .json file containing cone data.

    Returns:
        tuple: Two lists, blue_cones and yellow_cones, each containing (x, y) tuples of cone positions.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    x_values = data.get("x", [])
    y_values = data.get("y", [])
    colors = data.get("color", [])

    if not (len(x_values) == len(y_values) == len(colors)):
        raise ValueError("JSON file data lengths for 'x', 'y', and 'color' must be equal.")

    blue_cones = [(x, y) for x, y, color in zip(x_values, y_values, colors) if color.lower() == "blue"]
    yellow_cones = [(x, y) for x, y, color in zip(x_values, y_values, colors) if color.lower() == "yellow"]

    return blue_cones, yellow_cones


def plot_cones(blue_cones, yellow_cones, title="Cone Positions"):
    """
    Plots blue and yellow cones on a 2D plane.

    Args:
        blue_cones (list): List of (x, y) tuples for blue cones.
        yellow_cones (list): List of (x, y) tuples for yellow cones.
        title (str): Title of the plot.
    """
    # Extract x and y coordinates for blue cones
    blue_x, blue_y = zip(*blue_cones) if blue_cones else ([], [])
    # Extract x and y coordinates for yellow cones
    yellow_x, yellow_y = zip(*yellow_cones) if yellow_cones else ([], [])

    # Plot blue cones
    plt.scatter(blue_x, blue_y, c='blue', label='Blue Cones', s=30)
    # Plot yellow cones
    plt.scatter(yellow_x, yellow_y, c='yellow', label='Yellow Cones', s=30)

    # Add labels and legend
    plt.title(title)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Keep the scale equal for proper visualization

    # Show the plot
    plt.show()

# Assuming you have already parsed the cones from your JSON file
blue_cones, yellow_cones = parse_cone_data("squiggle.json")
plot_cones(blue_cones, yellow_cones)


