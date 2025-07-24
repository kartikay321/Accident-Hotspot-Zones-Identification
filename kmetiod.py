import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.datasets import load_iris
Iris=load_iris()
target=Iris.target
x = Iris.data
print(target)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
kMedoids = KMedoids(n_clusters = 3, random_state = 0)
kMedoids.fit(x_scaled)
y_kmed = kMedoids.fit_predict(x_scaled)
print(y_kmed)
fig, axes = plt.subplots(1, 2, figsize=(14,6))
axes[1].scatter(x[y_kmed == 0, 0], x[y_kmed== 0, 1], s = 100, c = 'purple')
axes[1].scatter(x[y_kmed == 1, 0], x[y_kmed == 1, 1], s = 100, c = 'orange')
axes[1].scatter(x[y_kmed == 2, 0], x[y_kmed == 2, 1], s = 100, c = 'green')
axes[1].set_title('After clusturing')
axes[0].scatter(x[target == 0, 0], x[target == 0, 1], s = 100, c = 'purple')
axes[0].scatter(x[target == 1, 0], x[target == 1, 1], s = 100, c = 'orange')
axes[0].scatter(x[target == 2, 0], x[target == 2, 1], s = 100, c = 'green')
axes[0].set_title('before clusturing')
#Plotting the centroids of the clusters
plt.legend()
plt.show()
print(pd.crosstab(target,y_kmed))