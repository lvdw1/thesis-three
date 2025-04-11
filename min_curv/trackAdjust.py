import json

data = "../sim/tracks/track10.json"
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

with open("track10_adjusted.json", "w") as f:
    json.dump(result, f, indent=4)
