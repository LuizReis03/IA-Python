# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:24:55 2022

@author: DISRCT
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset = pd.read_csv("./Mall_Customers.csv")
dataset.head()

x = dataset.values[..., -2:]
print(x.shape)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, init="k-means++", random_state=7)
y_pred = kmeans.fit_predict(x)
print(kmeans.inertia_)
print(y_pred)

def plot_clusters(x, y_pred):
    for i in range(100):
        indexes = y_pred == i
        color = np.random.rand(3,)
        plt.scatter(x[indexes, 0], x[indexes, 1], s=100, color=color, label=f"Cluster {i}")
    plt.show()
plot_clusters(x, y_pred)

kmeans_list = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=7)
    kmeans.fit(x)
    kmeans_list.append(kmeans)
wcss = [kmeans.inertia_ for kmeans in kmeans_list]
plt.plot(range(1, 11), wcss)

kmeans = kmeans_list[5 - 1]
y_pred = kmeans.predict(x)

plot_clusters(x, y_pred)

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=2, min_samples=3)
dbscan.fit(x)
y_pred = dbscan.fit_predict(x)
print(y_pred)
y_pred += 1
plot_clusters(x, y_pred)