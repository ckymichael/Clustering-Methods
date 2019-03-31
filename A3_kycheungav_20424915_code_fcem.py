import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
import pandas as pd
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
import time



# load data
X =loadmat('dataset.mat')['Points']

start_time = time.time()

# Initialize Clustering Centers
C1 = np.matrix(X[0])
C2 = np.matrix(X[1])

# Iteration
for i in range(300):
    # Calaulate square distance to C1 & C2
    dist_1 = np.sum(np.square(X - C1),axis=1)
    dist_2 = np.sum(np.square(X - C2),axis=1)
    # Calculate W_C1 & W_C2
    W_C1 = dist_2/(dist_1 + dist_2)
    W_C2 = 1 - W_C1
    # Calculate SSE(sum of squared error)
    SSE = np.sum(np.multiply(dist_1, W_C1)) + np.sum(np.multiply(dist_2, W_C2))
    # Save as old value
    C1_old = C1
    C2_old = C2
    # Calculate new Clustering Centers
    C1 = np.matmul(np.transpose(np.square(W_C1)), X)/np.sum(np.square(W_C1))
    C2 = np.matmul(np.transpose(np.square(W_C2)), X)/np.sum(np.square(W_C2))
    # Print
    print("After iteration "+ str(i+1) + ":")
    print("C1 is:" + str(C1) + "C2 is:" + str(C2))
    # Calculate the sum of L1 distance of two clustering centers
    L1_sum = np.sum(np.absolute(C1_old - C1)) + np.sum(np.absolute(C2_old - C2))
    # Terminate
    if L1_sum <= 0.001:
        print("After "+ str(i+1) + " iterations, with L1_sum="+str(L1_sum)+",the clusters converge.")
        break
# Print result
print("Converged C1:" + str(C1))
print("Converged C2:" + str(C2))
print("the final SSE(sum of squared error) is:")
print(SSE)


colors = ['r', 'g']
fig, ax = plt.subplots()

W_C1 = np.squeeze(np.asarray(W_C1))
W_C2 = np.squeeze(np.asarray(W_C2))

C1 = np.squeeze(np.asarray(C1))
C2 = np.squeeze(np.asarray(C2))
clusters = np.zeros(len(X))

for i in range (len(X)):
    if W_C1[i] > W_C2[i]:
        clusters[i] = 0
    else:
        clusters[i] = 1
        
for i in range(2):
    points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    
ax.scatter(C1[0], C1[1], marker='*', s=200, c='#050505')
ax.scatter(C2[0], C2[1], marker='*', s=200, c='#050505')
           
output =pd.DataFrame(columns=('point id', 'x-coordinate', 'y-coordinate', 'cluster id'))
output['point id'] = np.arange(0,len(X))
output['x-coordinate'] = X[:, 0]
output['y-coordinate'] = X[:, 1]
output['cluster id'] = clusters

output.to_csv('fcem.txt',index= False)


print("--- %s seconds ---" % (time.time() - start_time))