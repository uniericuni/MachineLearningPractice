import numpy as np
import scipy.io as sio
#
# Demo code for LR for bi-classification using Newton optimization
# x: feature vectors
# y: labels
#
 

import matplotlib.pyplot as plt
import LR as lr

# training env setting
trainingNum = 2000
e = 0.01
la = 10
mnist_49_3000 = sio.loadmat('mnist_49_3000.mat')    # import data
x = mnist_49_3000['x']
y = mnist_49_3000['y']

# seperating training and testing data
dim, n = x.shape
xT = x[:,0:trainingNum-1]
yT = y[:,0:trainingNum-1]
xTest = x[:,trainingNum:n-1]
yTest = y[:,trainingNum:n-1]
dim, n = xT.shape

# Newton optimization
it = 0
error = float('Inf')
theta = np.ones(dim+1)  # training offset
while error > e
    it += 1
    G = lr.gradFunc(x, y, theta, la)
    H = lr.hessFunc(x, y, theta, la)
    theta -= inv(H) * G
    error = lr.oFunc(x, y, theta, la)
    print('error: %s | iteration: %s', str(error), str(it))

print('error: %s | iteration: %s', str(error), str(it))
print('theta: ' + theta)
