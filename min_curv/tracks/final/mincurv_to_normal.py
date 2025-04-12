import json

with open('tracks/final/normal_mincurv.json', 'r') as f:
    path_mincurv = json.load(f)

x = []
y = []
for i in range(len(path_mincurv)):
    x.append(path_mincurv[i][0])
    y.append(path_mincurv[i][1])

result = {
        "x": x, 
        "y": y
        }

with open('normal_mincurv_final.json', 'w') as f:
    json.dump(result, f)
