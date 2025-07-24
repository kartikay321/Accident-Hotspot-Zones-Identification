import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
iris = pd.read_csv("Iris.csv")
x = iris.iloc[:, [0, 1, 2, 3]].values
# print(x)
iris.info()
iris[0:10]
from sklearn.cluster import KMeans
m = DBSCAN(eps=0.5, min_samples=10)
ydbscan=m.fit_predict(x)
# clusters = m.labels_

print(ydbscan)
plt.scatter(x[ydbscan == 0, 0], x[ydbscan == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(x[ydbscan == -1, 0], x[ydbscan == -1, 1], s = 100, c = 'red')
plt.scatter(x[ydbscan == 1, 0], x[ydbscan == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(x[ydbscan == 2, 0], x[ydbscan == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
# plt.scatter(m.cluster_centers_[:, 0], m.cluster_centers_[:,1], s = 100, c = 'red', label = 'Centroids')

plt.legend()
plt.show()