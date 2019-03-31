from copy import deepcopy
from scipy.io import loadmat
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
import time
import random

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

# load data
X =loadmat('dataset.mat')['Points']

# Number of clusters
num_cluster = [2, 10, 20, 30]

number_of_colors =30

colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]

for k in num_cluster:
    start_time = time.time()
    C =X[np.random.choice(len(X), size=k, replace=False)] 
    print(C)

    # To store the value of centroids when it updates
    C_old = np.zeros(C.shape)
    # Cluster Lables(0, 1, 2)
    clusters = np.zeros(len(X))
    # Error func. - Distance between new centroids and old centroids
    error = dist(C, C_old, None)
    # Loop will run till the error becomes zero
    while error != 0:
        # Assigning each value to its closest cluster
        for i in range(len(X)):
            distances = dist(X[i], C)
            cluster = np.argmin(distances)
            clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
        # Finding the new centroids by taking the average value
        for i in range(k):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)

    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
    file = 'kmean_' + str(k) +'.txt'
    output =pd.DataFrame(columns=('point id', 'x-coordinate', 'y-coordinate', 'cluster id'))
    output['point id'] = np.arange(0,len(X))
    output['x-coordinate'] = X[:, 0]
    output['y-coordinate'] = X[:, 1]
    output['cluster id'] = clusters
    output.to_csv(file ,index= False)
    print("--- %s seconds ---" % (time.time() - start_time))