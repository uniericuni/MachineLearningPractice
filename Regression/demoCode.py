# Demo code for LR for bi-classification using Newton optimization
# x: feature vectors
# y: labels
#

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from Regress import Reg

# training env setting
trainingNum = 150                                   # size of training data
alpha = 1e-4
la = 10                                             # regularizing constant
e = 1e-14
data = sio.loadmat('bodyfat_data.mat')              # import data
x = data['X']                                       # feature vectors
y = data['y']                                       # labels
dim, n = x.shape

# seperating training and testing data
xTr = x[:,0:trainingNum]
yTr = y[:,0:trainingNum]
xTst = x[:,trainingNum:n]
yTst = y[:,trainingNum:n]

# regression object built up
rg = Reg()
# theta = np.zeros(dim+1)                           # starting point
rg.dataUpdate(xTr, yTr, xTst, yTst)                 # Newton optimization
out, err = rg.ridgeReg(w, la, e, alpha, 'GD')            # Smallscale ridge regression

# testing
test = np.array([1, 100,100])
y = theta*test

# info export
print('theta: ', out)
print('error: ', err)
print('given [100, 100], the corresponding y is: ', y)

