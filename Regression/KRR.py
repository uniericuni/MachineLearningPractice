from dist2 import dist2
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import random

# Regression Object
class KRR:

    # constructor
    def __init__(self):
        self.clear()
        random.seed(0)

    # clear regression object
    def clear(self):
        self.trX = None
        self.trY = None
        self.tstX = None
        self.tstY = None
        self.dim = None
        self.trSize = None
        self.tstSize = None
    
    # data parsing    
    def dataUpdate(self, trX, trY, tstX, tstY):
        self.trSize, self.dim = trX.shape               # training size and dimension
        self.trX = trX                                  # training X
        self.trY = trY                                  # training Y
        self.tstSize, self.dim = tstX.shape             # testing size and dimension
        self.tstX = tstX                                # testing X
        self.tstY = tstY                                # testing Y
        print('\ndata parsed ...')
        print('------------------------------------')
        print('training X ... dim: ', self.dim, ' | size: ', self.trSize)
        n,d = trY.shape
        print('training Y ... dim: ', d, ' | size: ', n)
        print('testing X  ... dim: ', self.dim, ' | size: ', self.tstSize)
        n,d = tstY.shape
        print('testing Y  ... dim: ', d, ' | size: ', n)
        print('------------------------------------\n')

    # KRR
    def ridgeReg(self, la=0, kr='IP', offset=True):
        
        self.kr = kr
        x = self.trX
        y = self.trY
        n, d = x.shape
        rtn = np.zeros(self.dim+1)
        
        # KRR
        print('kernel ridge regression ...')
        y_tel = y - np.average(y)                   # y_telda
        y_bar = np.average(y)                       # y_bar
        I = np.eye(n)
        if offset:
            A = self.kernel(x, x) + n*la*I
            b = y_bar + np.dot(np.dot(y_tel.transpose(), np.linalg.inv(A)), self.kernel(x,np.zeros((1,d))))
            y_hatTr  = y_bar + np.dot(np.dot(y_tel.transpose(), np.linalg.inv(A)), self.kernel(x,x))
            y_hatTst = y_bar + np.dot(np.dot(y_tel.transpose(), np.linalg.inv(A)), self.kernel(x,self.tstX))
        else:
            A = self.kernel(x, x, False) + n*la*I
            b = np.array([[0]])
            y_hatTr  = np.dot(np.dot(y.transpose(), np.linalg.inv(A)), self.kernel(x,x,False))
            y_hatTst = np.dot(np.dot(y.transpose(), np.linalg.inv(A)), self.kernel(x,self.tstX,False))
        err1 = np.average((y_hatTr.transpose()-self.trY)**2)
        err2 = np.average((y_hatTst.transpose()-self.tstY)**2)

        print('------------------------------------')
        print('b', b[0,0])
        print('training error', err1)
        print('testing error', err2)
        print('------------------------------------\n')
        return b, err1, err2

    def kernel(self, x1, x2, offset=True):
        rho = 15
        n, d = x1.shape
        m, d = x2.shape

        if self.kr=='IP':
            K1 = np.dot(x1,x1.transpose());    
            K2 = np.dot(x1,x2.transpose());    
        elif self.kr=='Gaussian':
            K1 = self.gKernel(x1,x1,rho)
            K2 = self.gKernel(x1,x2,rho)
        else:
            print "wrong kernel method"
            quit()

        O1 = np.ones((n,n))/n
        O2 = np.ones((n,m))/n
        
        if offset: 
            return K2 - np.dot(K1, O2) - np.dot(O1, K2) + np.dot( np.dot(O1, K1), O2)
        else:
            return K2

    def gKernel(self, x1, x2, rho):
        return np.exp((-1/(2*(rho**2)))*dist2(x1,x2))
