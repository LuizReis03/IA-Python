# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:28:01 2022

@author: DISRCT
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import time

n_samples = 1500
tempo = time.time()

#DATASETS COM OS VALORES
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

#PLOTA OS GRÁFICOS
def plot_clusters(x, y_pred):
    for i in range(100):
        indexes = y_pred == i
        color = np.random.rand(3,)
        plt.scatter(x[indexes, 0], x[indexes, 1], s=100, color=color, label=f"Cluster {i}")
    plt.show()


#PEGA TODOS OS VALORES DAS TUPLAS DE DATASETS
toy_datasets = [noisy_circles, noisy_moons, blobs, no_structure]

print("\n\nGRAFICOS KMEANS")
for X, y in toy_datasets:
    kmeans = KMeans(n_clusters=3, init="k-means++", random_state=7)
    y_pred = kmeans.fit_predict(X)
    plot_clusters(X, y_pred)
    
    print("Tempo de execução:", round(tempo,2))
    plt.show()
    
print("\n\nGRAFICOS DBSCAN")
for X, y in toy_datasets:
    dbscan = DBSCAN(eps=0.1, min_samples=5)
    dbscan.fit(X)
    y_pred = dbscan.fit_predict(X)
    y_pred += 1
    plot_clusters(X, y_pred)
    
    print("Tempo de execução:", round(tempo,2))
    plt.show()
    
#PLOTA A MELHOR QUANTIDADE DE CLUSTERS
kmeans_list = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=7)
    kmeans.fit(X)
    kmeans_list.append(kmeans)
wcss = [kmeans.inertia_ for kmeans in kmeans_list]
plt.plot(range(1, 11), wcss)
