import json


data = "../sim/tracks/validation/normal.json"
with open(data, "r") as f:
    track = json.load(f)

left_boundary_x = []
left_boundary_y = []
right_boundary_x = []
right_boundary_y = []

for i, col in enumerate(track["color"]):
    if col == "blue":
        left_boundary_x.append(track["x"][i])
        left_boundary_y.append(track["y"][i])
    elif col == "yellow":
        right_boundary_x.append(track["x"][i])
        right_boundary_y.append(track["y"][i])

result = {
        "left_boundary_x": left_boundary_x,
        "left_boundary_y": left_boundary_y,
        "right_boundary_x": right_boundary_x,
        "right_boundary_y": right_boundary_y,
        "centerline_x": track["centerline_x"],
        "centerline_y": track["centerline_y"],
        }

result_folder = "adjusted/normal.json"
# Save the result to a new JSON file
try:
    with open(result_folder, "w") as f:
        json.dump(result, f, indent=4)
    print(f"File saved successfully to {result_folder}")
except Exception as e:
    print(f"An error occurred while saving the file: {e}")

