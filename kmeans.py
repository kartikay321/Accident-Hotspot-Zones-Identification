import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
Iris=load_iris()
target=Iris.target
iris = pd.read_csv("Iris.csv")
x = iris.iloc[:, [0, 1, 2, 3]].values
iris[0:10]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, max_iter = 300, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
print(target)
print(y_kmeans)

fig, axes = plt.subplots(1, 2, figsize=(14,6))
axes[1].scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'purple')
axes[1].scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'orange')
axes[1].scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green')
axes[1].set_title('After clusturing')
#Plotting the centroids of the clusters
axes[1].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red' )
axes[0].scatter(x[target == 0, 0], x[target == 0, 1], s = 100, c = 'purple')
axes[0].scatter(x[target == 1, 0], x[target == 1, 1], s = 100, c = 'orange')
axes[0].scatter(x[target == 2, 0], x[target == 2, 1], s = 100, c = 'green')
axes[0].set_title('before clusturing')
#Plotting the centroids of the clusters


plt.legend()
plt.show()
print(pd.crosstab(target,y_kmeans))
