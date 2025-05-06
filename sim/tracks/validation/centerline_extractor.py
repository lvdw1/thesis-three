import json

with open('normal.json', 'r') as f:
    data = json.load(f)

centerline_x = data["centerline_x"]
centerline_y = data["centerline_y"]

points = [{"x": x, "y": y} for x, y in zip(centerline_x, centerline_y)]

out = {"centerline": points}

with open('normal_centerline.csv', 'w') as f:
    f.write("x,y\n")
    for x, y in zip(data["centerline_x"], data["centerline_y"]):
        f.write(f"{x},{y}\n")
