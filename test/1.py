import numpy as np
from scipy.spatial import distance

THRESHOLD_DISTANCE = 0.1
object_centroids = [[0.4, 0.6],[0.25, 0.25], [0.75,0.35]]
input_centroids = [[0.3, 0.3],[0.75,0.7],[0.5, 0.5],[0.75, 0.5]]

dist = distance.cdist(object_centroids, input_centroids)
(dist.min(axis = 1) < 0.1).argmax()
# follow PyImageSearch's algorithm
# if dist.min(axis=1)
rows = dist.min(axis=1).argsort()
cols = dist.argmin(axis=1)[rows]
print(dist)
print(rows)
print(cols)





