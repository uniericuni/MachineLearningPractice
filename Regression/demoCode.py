# Demo code for Linear Regression
# x: feature vectors
# y: labels

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib import cm
from Regress import Reg
from mpl_toolkits.mplot3d import Axes3D

# training env setting
data = sio.loadmat('bodyfat_data.mat')              # import data
x = data['X']                                       # feature vectors
y = data['y']                                       # labels
n, dim = x.shape
trainingNum = 150                                   # size of training data
la = 10                                             # regularizing constant
e = 1e-14

# seperating training and testing data
xTr = x[0:trainingNum,:]
yTr = y[0:trainingNum,:]
xTst = x[trainingNum:n,:]
yTst = y[trainingNum:n,:]

# regression object built up
rg = Reg()
w = np.zeros((dim, 1))                              # starting point
rg.dataUpdate(xTr, yTr, xTst, yTst)                 # Newton optimization
out, err = rg.ridgeReg(w, la, e)                    # Regression

# single value testing
test = np.array([1, 100,100])
yout = np.dot(out, test)
print('given [100, 100], the corresponding y is: ', yout, '\n')

# plotting
fig = plt.figure()
x1,x2 = np.meshgrid(pl.frange(60,120,0.1), pl.frange(50,120,0.1))
z = (x1*out[1]+x2*out[2]+out[0])
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, z, cmap=cm.autumn, linewidth=0, antialiased=False)
ax.scatter(x[:,0], x[:,1], y)
plt.show()
