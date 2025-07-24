import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.preprocessing import StandardScaler
df=pd.read_excel('final_accident_data.xlsx')
print(df)
df.drop(0,inplace=True)
print(df.isna().sum())
print(df.shape)
df.drop(['Number'], inplace=True, axis=1)
print("after dropping Number collumn")
print(df.shape)
print(df.isna().sum())
df.dropna(axis=0,inplace=True)
df.isna().sum()
x = df.iloc[:, [5, 6]].values
arr_x=[]
arr_y=[]
X=np.empty((137955, 2),float)
for i in range(137955):
    arr_x.append(x[i,0])
    arr_y.append(x[i,1])
    X[i][0]=x[i,0]
    X[i][1]=x[i,1]
plt.scatter(arr_x,arr_y, s = 10, c = 'orange')
plt.title('Initial data')
plt.show()
outlier_percent = [] 

for eps in np.linspace(0.001,3,50): # check 50 values of epsilon between 0.001 and 3
    try:
    # Create Model
        dbscan = DBSCAN(eps=eps,min_samples=12)
        dbscan.fit(X)
        # Percentage of points that are outliers
        perc_outliers = 100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
        outlier_percent.append(perc_outliers)
        print(perc_outliers,eps,i)
    except:
        pass
sns.lineplot(x=np.linspace(0.001,3,50),y=outlier_percent, color='green')
plt.ylabel("Percentage of Points Classified as Outliers")
plt.xlabel("Epsilon Value")
plt.show()