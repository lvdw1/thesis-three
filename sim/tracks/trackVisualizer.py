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

def plot_cones_on_axis(ax, blue_cones, yellow_cones, title="Cone Positions"):
    """
    Plots blue and yellow cones on a given matplotlib axis.

    Args:
        ax (matplotlib.axes.Axes): The axis on which to plot the cones.
        blue_cones (list): List of (x, y) tuples for blue cones.
        yellow_cones (list): List of (x, y) tuples for yellow cones.
        title (str): Title of the subplot.
    """
    # Extract x and y coordinates for blue cones
    if blue_cones:
        blue_x, blue_y = zip(*blue_cones)
    else:
        blue_x, blue_y = [], []
    
    # Extract x and y coordinates for yellow cones
    if yellow_cones:
        yellow_x, yellow_y = zip(*yellow_cones)
    else:
        yellow_x, yellow_y = [], []
    
    # Plot blue and yellow cones on the provided axis
    ax.scatter(blue_x, blue_y, c='blue', label='Blue Cones', s=30)
    ax.scatter(yellow_x, yellow_y, c='yellow', label='Yellow Cones', s=30)
    
    # Customize the axis
    ax.set_title(title)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

def plot_four_tracks(json_files):
    """
    Plots cone data from four different JSON files in a 2x2 grid of subplots.

    Args:
        json_files (list): List of four JSON file paths.
    """
    if len(json_files) != 4:
        raise ValueError("Exactly four JSON files must be provided.")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # flatten to easily iterate through axes
    
    for i, json_file in enumerate(json_files):
        blue_cones, yellow_cones = parse_cone_data(json_file)
        plot_cones_on_axis(axes[i], blue_cones, yellow_cones, title=f"Track {i+1}")
    
    plt.tight_layout()
    plt.show()

def plot_single_track(json_file):
    """
    Plots cone data from a single JSON file in a single subplot.

    Args:
        json_file (str): JSON file path.
    """
    blue_cones, yellow_cones = parse_cone_data(json_file)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_cones_on_axis(ax, blue_cones, yellow_cones, title="Hard Layout")
    plt.tight_layout()
    plt.show()

# Main Code
# If you want to plot a single track, use the plot_single_track function:
json_file = "validation/hard.json"
plot_single_track(json_file)

# If you want to plot four tracks, prepare a list of four JSON files and call plot_four_tracks:
# json_files = ["track5.json", "track6.json", "track7.json", "track8.json"]
# plot_four_tracks(json_files)
