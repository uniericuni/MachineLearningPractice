import numpy as np
import scipy.io as sio
#
# Demo code for LR for bi-classification using Newton optimization
# x: feature vectors
# y: labels
#

import matplotlib.pyplot as plt
import LR2 as lr

# training env setting
trainingNum = 2000                                  # size of training data
e = 0.01                                            # acceptable error rate
la = 10                                             # regularizing constant
it = 0                                              # iterator
mnist_49_3000 = sio.loadmat('mnist_49_3000.mat')    # import data
x = mnist_49_3000['x']                              # feature vectors
y = mnist_49_3000['y']                              # labels
dim, n = x.shape

# seperating training and testing data
xT = x[:,0:trainingNum]
yT = y[:,0:trainingNum]
xTest = x[:,trainingNum:n]
yTest = y[:,trainingNum:n]

# Newton optimization
theta = np.zeros(dim+1)                             # norm offset
error = float('Inf')                                # error rate offset

for i in range(0,1000):
    it += 1
    G = lr.gradFunc(xT, yT, theta, la)
    H = lr.hessFunc(xT, yT, theta, la)
    delta = np.dot(np.linalg.inv(H),G)[:,0]
    theta -= delta
    error = lr.oFunc(x, y, theta, la)
    print('error: ', str(error), '| iteration: ', str(it), '| G: ', np.linalg.norm(G))

sio.savemat('theta.mat', {'theta':theta})

dim, n = xTest.shape
X = np.ones((dim+1,n))
X[1:,:] = xTest
t = -np.dot(theta,X)

count1 = 0
count2 = 0
count3 = 0
count4 = 0
for i in range(0,n):
    if t[i]>=0 and yTest[0,i]==-1:
        count1 += 1
    if t[i]>=0 and yTest[0,i]==1:
        count2 += 1
    if t[i]<=0 and yTest[0,i]==-1:
        count3 += 1
    if t[i]<=0 and yTest[0,i]==1:
        count4 += 4

print(count1, '|', count2, '|', count3, '|', count4)
