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
trainingNum = 2000
e = 0.01
la = 10
mnist_49_3000 = sio.loadmat('mnist_49_3000.mat')    # import data
x = mnist_49_3000['x']
y = mnist_49_3000['y']
dim, n = x.shape

# modify label
# for i in range(0,n):
#     if y[:,i] == [-1]:    
#         y[:,i] = [0] 

# seperating training and testing data
xT = x[:,0:trainingNum]
dim, n = xT.shape
yT = y[:,0:trainingNum]
xTest = x[:,trainingNum:n-1]
yTest = y[:,trainingNum:n-1]

print(str(trainingNum))
theta = np.zeros(dim+1)  # training offset
H=lr.hessFunc(xT,yT,theta,la)
print(H)
'''
# Newton optimization
it = 0
error = float('Inf')
theta = np.zeros(dim+1)  # training offset
while error > e:
    it += 1
    G = lr.gradFunc(xT, yT, theta, la)
    print G
    H = lr.hessFunc(xT, yT, theta, la)
    print H
    theta -= np.linalg.inv(H) * G
    error = lr.oFunc(x, y, theta, la)
    print('error: %s | iteration: %s', str(error), str(it))

print('error: %s | iteration: %s', str(error), str(it))
print('theta: ' + theta)
'''
