import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from sklearn.cluster import DBSCAN
import time


#load data
X =loadmat('dataset.mat')['Points']

start_time = time.time()

y_pred = DBSCAN(eps = 0.12, min_samples = 3).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()

output =pd.DataFrame(columns=('point id', 'x-coordinate', 'y-coordinate', 'cluster id'))
output['point id'] = np.arange(0,len(X))
output['x-coordinate'] = X[:, 0]
output['y-coordinate'] = X[:, 1]
output['cluster id'] = y_pred

output.to_csv('dbscan.txt',index= False)

print("--- %s seconds ---" % (time.time() - start_time))