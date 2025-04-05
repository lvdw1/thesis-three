import yaml
import json

# Load the YAML file
with open('fssim_fsi2.yaml', 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)

# Extract x and y coordinates from each pose
centerline_x = []
centerline_y = []
for pose_entry in data.get('poses', []):
    pos = pose_entry.get('pose', {}).get('position', {})
    centerline_x.append(pos.get('x', 0))
    centerline_y.append(pos.get('y', 0))

with open('fssim_fsi.yaml', 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)

cone_x = []
cone_y = []
cone_color = []
# Define mapping from observation class to color
color_map = {0: "blue", 1: "yellow", 2: "orange"}

for obs in data.get('observations', []):
    observation = obs.get('observation', {})
    loc = observation.get('location', {})
    cone_x.append(loc.get('x', 0))
    cone_y.append(loc.get('y', 0))
    obs_class = observation.get('observation_class', None)
    cone_color.append(color_map.get(obs_class, "unknown"))

# Create the output dictionary
output = {
    "x": cone_x,
    "y": cone_y,
    "color": cone_color,
    "centerline_x": centerline_x,
    "centerline_y": centerline_y
}
# Write the result to a JSON file
with open('fssim_fsi.json', 'w') as json_file:
    json.dump(output, json_file, indent=2)

print("Conversion complete. Data written to output.json")
