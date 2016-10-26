# Demo code for Linear Regression
# x: feature vectors
# y: labels

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib import cm
from KRR import KRR
from mpl_toolkits.mplot3d import Axes3D

# training env setting
data = sio.loadmat('bodyfat_data.mat')              # import data
x = data['X']                                       # feature vectors
y = data['y']                                       # labels
n, dim = x.shape
trainingNum = 150                                   # size of training data

# seperating training and testing data
xTr = x[0:trainingNum,:]
yTr = y[0:trainingNum,:]
xTst = x[trainingNum:n,:]
yTst = y[trainingNum:n,:]

# regression object built up
print "IP kernel"
la = 10                                             # regularizing constant
rg = KRR()
rg.dataUpdate(xTr, yTr, xTst, yTst)                 
b, err1, err2 = rg.ridgeReg(la)                     # Regression

print "Gaussian kernel"
la = 0.003                                          # regularizing constant
rg.clear()
rg = KRR()
rg.dataUpdate(xTr, yTr, xTst, yTst)                 
b, err1, err2 = rg.ridgeReg(la,kr="Gaussian")                     # Regression
