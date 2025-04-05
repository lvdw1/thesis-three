#!/usr/bin/env python3

import yaml
import json

# Mapping from observation_class to color.
class_to_color = {
    0: "blue",
    1: "yellow",
    2: "orange"
}

# Load the YAML file.
with open("fsg24.yaml", "r") as f:
    data = yaml.safe_load(f)

x_coords = []
y_coords = []
colors = []

# Iterate over each observation entry.
# The script assumes that 'observations' is a list of dictionaries,
# each containing a key 'observation' with the relevant data.
for entry in data["observations"]:
    obs = entry["observation"]
    x_coords.append(obs["location"]["x"])
    y_coords.append(obs["location"]["y"])
    colors.append(class_to_color[obs["observation_class"]])

# Create the result dictionary.
result = {"x": x_coords, "y": y_coords, "color": colors}

# Write the result to a JSON file.
with open("fsg_trackdrive_2024.json", "w") as f:
    json.dump(result, f, indent=2)
