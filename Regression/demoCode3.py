# Demo code for Linear Regression
# x: feature vectors
# y: labels

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pylab as pl
import random
from matplotlib import cm
from Regress import Reg
from mpl_toolkits.mplot3d import Axes3D

random.seed(0)
# training env setting
data = sio.loadmat('nuclear.mat')                   # import data
x = data['x']                                       # feature vectors
y = data['y']                                       # labels
x = x.transpose()
y = y.transpose()
n, dim = x.shape
la = 0.001                                          # regularizing constant
e = 1e-5                                            # stop criteria
trainingNum = 20000

# seperating training and testing data
xTr = x[0:trainingNum,:]
yTr = y[0:trainingNum,:]
xTst = x[0:trainingNum,:]
yTst = y[0:trainingNum,:]

# --------------- gradient regression --------------- #
rg = Reg()
w = np.zeros((dim+1, 1))                            # starting point
rg.dataUpdate(xTr, yTr, xTst, yTst)                 # Newton optimization
out, err = rg.ridgeReg(w, la, e, method='GD')       # Regression

# plotting
fig = plt.figure()
pid = np.argwhere(y-1)
nid = np.argwhere(y+1)
pid = pid[:,0]
nid = nid[:,0]
xp = x[pid.transpose()]
xn = x[nid.transpose()]
plt.scatter(xn[:,0],xn[:,1], c='b', label='y=-1')
plt.scatter(xp[:,0],xp[:,1], c='r', label='y=1')
t = pl.frange(0, 8, 0.01)
w = -out[1]/out[2]
b = -out[0]/out[2]
plt.plot(t, w*t+b, 'g-', label='hyperplane')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()

# --------------- stochastic gradient regression --------------- #
rg.clear()
w = np.zeros((dim+1, 1))                            # starting point
rg.dataUpdate(xTr, yTr, xTst, yTst)                 # Newton optimization
out, err = rg.ridgeReg(w, la, e, method='SGD')      # Regression

# plotting
fig = plt.figure()
pid = np.argwhere(y-1)
nid = np.argwhere(y+1)
pid = pid[:,0]
nid = nid[:,0]
xp = x[pid.transpose()]
xn = x[nid.transpose()]
plt.scatter(xn[:,0],xn[:,1], c='b', label='y=-1')
plt.scatter(xp[:,0],xp[:,1], c='r', label='y=1')
t = pl.frange(0, 8, 0.01)
w = -out[1]/out[2]
b = -out[0]/out[2]
plt.plot(t, w*t+b, 'g-', label='hyperplane')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()
