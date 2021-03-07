import csv
import os
import numpy as np
import sys

csvFiles = []
for root, dirs, files in os.walk("./data"):
    for file in files:
        if file.endswith(".csv"):
             csvFiles.append(os.path.join(root, file))

print(csvFiles)

data = []
for f in csvFiles:
    with open(f, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='#')
        for row in spamreader:
            temp = row[0].split(',')
            if temp[0] == "Data" and temp[2] == "record" and "position_lat" in temp:
                temp = [value for value in temp if value != '']
                data.append(temp)

indices = [[i for i, x in enumerate(data[d]) if '"' in x] for d in range(len(data))]
labels = [[data[d][j] for j in [i - 1 for i in indices[d]]] for d in range(len(data))]
data = [[data[d][j] for j in indices[d]] for d in range(len(data))]
data = [[float("".join(x.replace('"', '').split(","))) for x in d] for d in data]


uniqueLabels = list(set(np.array(labels).flatten()))

uniqueLabelsDict = {}

dataT = []
for r in range(len(data)):
    for uL in range(len(uniqueLabels)):
        uniqueLabelsDict[uniqueLabels[uL]] = uL
        dataT.append([])
        if uniqueLabels[uL] in labels[r]:
            dataT[uL].append(data[r][labels[r].index(uniqueLabels[uL])])
        else:
            dataT[uL].append(0)

###########################################

def normalize(arr):
    mi = min(arr)
    arr = [i - mi for i in arr]
    ma = max(arr)
    return [i / ma for i in arr]

from vispy import app, visuals, scene
from vispy.color import ColorArray
from scipy.interpolate import PchipInterpolator as pchip
from sklearn.preprocessing import QuantileTransformer

# build visuals
Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)

# build canvas
canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'fly'
view.camera.fov = 45
view.camera.distance = 6

# prepare data
x = dataT[uniqueLabelsDict["position_long"]]
y = dataT[uniqueLabelsDict["position_lat"]]
z = dataT[uniqueLabelsDict["enhanced_altitude"]]
x -= np.mean(x)
y -= np.mean(y)
z -= np.mean(z)
z *= 1000

# plot
pos = np.c_[x, y, z]
r1 = Plot3D(pos, parent=view.scene)

speed = dataT[uniqueLabelsDict["enhanced_speed"]]
#s = normalize(np.array([(speeds[i] + speeds[i+1]) / 2 for i in range(len(speeds)-1)]))

hr = dataT[uniqueLabelsDict["heart_rate"]]
hrdelta = [(hr[i+1] - hr[i]) for i in range(len(hr)-1)]
travel = [(((x[i+1] - x[i]) ** 2) + ((y[i+1] - y[i]) ** 2)) ** 0.5 for i in range(len(hr)-1)]
hgrad = np.array(hrdelta) / np.array(travel)
for i in range(len(travel)):
    if travel[i] == 0:
        hgrad[i] = 0
hgrad = np.nan_to_num(hgrad)
hgrad = normalize(hgrad)
quantile = QuantileTransformer(output_distribution='uniform')
hgrad = quantile.fit_transform(np.reshape(hgrad, (-1, 1)))
interpol = pchip([i+0.5 for i in range(len(hr)-1)], hgrad)
ip = interpol([i for i in range(1, len(hr)-1)])**4
ip = np.nan_to_num(ip)
h = [0]
h.extend(list(ip))
h.extend([0])

colors = ColorArray(color=[[1,1-h[i],1-h[i]] for i in range(len(h))], color_space='rgb')
r1.set_data(pos, width=1, color=colors, marker_size=0)

### APP ###

if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()