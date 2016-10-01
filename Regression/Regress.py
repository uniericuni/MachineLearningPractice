# Logistic Regression
#
# x: feature vector | dimxn matrix, array object
# y: lables         | nx1 matrix, array object
# theta: norm       | (dim+1)x1 matrix, array object
# la: regularizer   | constant
#
# dim: dimension of feature vector
# n: size of testing data

import scipy.io as sio
import numpy as np

# Regression Object
class Reg:

    # constructor
    def __init__(self):
        self.trX = np.array([[]])
        self.trY = np.array([[]])
        self.tstX = np.array([[]])
        self.tstY = np.array([[]])
        self.dim = 0
        self.trSize = 0
        self.tstSize = 0
    
    # data parsing    
    def dataUpdate(self, trX, trY, tstX, tstY):
        self.dim, self.trSize = trX.shape               # matrix scale
        self.trX = trX
        self.trY = trY
        self.dim, self.tstSize = tstX.shape             # matrix scale
        self.tstX = tstX
        self.tstY = tstY
        print('data parsed')

    # ridge regressoin
    def ridgeReg(self, w, la, e, alpha, method='GD'):
        wout = w                                        # regression starting point
        it = 0                                          # iterator
        err = 0                                         # error rate
        g = float('Inf')                                # gradient norm offset
        pre = -float('Inf')                             # previous gradient norm offset
         
        if method is 'GD':
            while abs(g-pre) > e:                       # iterate until the gradient norm barely changes
                pre = g
                it += 1                                 # iterator
                G = gradFunc(w, la)                     # gradient
                g = np.linalg.norm(G)                   # gradient norm
                wout -= alpha*G                         # gradient descent
                err = oFunc(w, la)                      # objective function
                print('iteration: ', str(it), ' | error: ', str(err),  ' | G: ', g)
        
        X = np.average(self.trX)
        y = np.average(self.trY)
        out = np.array(self.dim+1)
        out[0] = y-np.dot(w,X)
        out[1:] = w
        return out, err 

    # gradient of the objective function
    def gradFunc(self, w, la):
        X = self.trX - np.average(self.trX)
        y = self.trY - np.average(self.trY)
        A = np.dot(np.transpose(X),X)+self.trSize*la*np.eye(self.dim)
        B = np.dot(np.transpose(X), y)
        return 2n*p.dot(A,w)-2*B

    # objective function for input x, y, theta and la(lambda)
    def oFunc(w, la):
        X = self.trX - np.average(self.trX)
        y = self.trY - np.average(self.trY)
        return np.linalg.(norm(y-np.dot(X,w))**2) + self.trSize*(np.linalg.norm(w)**2)
