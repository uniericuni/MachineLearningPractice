# Demo code for Linear Regression
# x: feature vectors
# y: labels

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from Kernels import CauchKernel

# training env setting
data = sio.loadmat('diabetes_scaled.mat')           # import data
x = data['X']                                       # feature vectors
y = data['y']                                       # labels
n, dim = x.shape
trainingNum = 500
k = 5

# seperating training and testing data
xTr = x[0:trainingNum,:]
yTr = y[0:trainingNum,:]
xTst = x[trainingNum:n,:]
yTst = y[trainingNum:n,:]

# SVM training
count = 0
delta = trainingNum/k
min_error = float('inf')
min_index = (1,1)
for i,C in enumerate([2**x for x in range(6,12)]):  # k-fold cross validation
    for j,sigma in enumerate([2**x for x in range(0,6)]):
        CauchKernel.sigma = sigma
        c = float(C)/float((k-1)*delta)
        classifier = SVC(C=c, tol=1e-2, verbose=False, kernel=CauchKernel)
        error = 0
        for i in range(0,k):                       
            xTrk = xTr[delta*i:delta*(i+1),:]
            yTrk = yTr[delta*i:delta*(i+1),:]
            t = np.linspace(delta*i, delta*(i+1)-1, num=delta)
            xTstk = np.delete(xTr, t, 0)
            yTstk = np.delete(yTr, t, 0)
            err= 1-classifier.fit(xTrk, np.ravel(yTrk)).score(xTstk, np.ravel(yTstk))
            error += err/k
        print "C:", '{:>4}'.format(str(c)), "| sigma: ", '{:>2}'.format(str(sigma)), "| error: ", '%.4f'%err
        if min_error > err:
            min_error = err
            min_index = (c, sigma)

print ""
print "parameters: ", min_index
CauchKernel.sigma = min_index[1]
classifier = SVC(C=min_index[0], tol=1e-2, verbose=False, kernel=CauchKernel)
print "number of support vector: ", classifier.fit(xTr, np.ravel(yTr)).n_support_
print "training error: ", 1-classifier.fit(xTr, np.ravel(yTr)).score(xTr, np.ravel(yTr))
print "tresting error: ", 1-classifier.fit(xTr, np.ravel(yTr)).score(xTst, np.ravel(yTst))
