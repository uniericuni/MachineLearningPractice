# Demo code for LR for bi-classification using Newton optimization
# x: feature vectors
# y: labels
#

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import LR as lr

# training env setting
trainingNum = 2000                                  # size of training data
e = 0.01                                            # acceptable error rate
la = 10                                             # regularizing constant
mnist_49_3000 = sio.loadmat('mnist_49_3000.mat')    # import data
x = mnist_49_3000['x']                              # feature vectors
y = mnist_49_3000['y']                              # labels
dim, n = x.shape

# seperating training and testing data
xT = x[:,0:trainingNum]
yT = y[:,0:trainingNum]
xTest = x[:,trainingNum:n]
yTest = y[:,trainingNum:n]

# training
theta = np.zeros(dim+1)                             # starting point
oTheta = lr.newton(xT, yT, theta, la)               # Newton optimization

# testing
dim, n = xTest.shape                                # measuring dimension and data set size
X = np.ones((dim+1,n))                              # augmented X with additional row of 1s
X[1:,:] = xTest                                     
t = np.dot(theta,X)                                 # exponential term of sigmoid function
error = 0
for i in range(0,n):                                # error accumulation
    if (t[i]>=0 and yTest[0,i]==-1) or (t[i]<0 and yTest[0,i]==1):
        error += 1

# export results
print('error rate: ', error/n)                      
sio.savemat('result.mat', {'theta':theta, 'theta_x':t, 'error_rate':error/n})
