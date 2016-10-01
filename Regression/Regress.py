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
        self.trSize, self.dim = trX.shape               # matrix scale
        self.trX = trX
        self.trY = trY
        self.tstSize, self.dim = tstX.shape             # matrix scale
        self.tstX = tstX
        self.tstY = tstY
        print('\ndata parsed ...')
        print('------------------------------------')
        print('training X ... dim: ', self.dim, ' | size: ', self.trSize)
        n,d = trY.shape
        print('training Y ... dim: ', d, ' | size: ', n)
        print('testing X  ... dim: ', self.dim, ' | size: ', self.tstSize)
        n,d = tstY.shape
        print('testing Y  ... dim: ', d, ' | size: ', n)
        print('------------------------------------\n')

    # ridge regressoin
    def ridgeReg(self, w, la, e, alpha, method='GD'):
        wout = w                                        # regression starting point
        it = 0                                          # iterator
        err = 0                                         # error rate
        g = float('Inf')                                # gradient norm offset
        pre = -float('Inf')                             # previous gradient norm offset
         
        # Gradient Descent
        if method is 'GD':
            while it == 0:                              # iterate until the gradient norm barely changes
            #while abs(g-pre) > e:                      # iterate until the gradient norm barely changes
                pre = g
                it += 1                                 # iterator
                G = self.gradFunc(w, la)                # gradient
                g = np.linalg.norm(G)                   # gradient norm
                wout -= alpha*G                         # gradient descent
                TrErr = self.oFunc(w, la)               # objective function
                print('iteration: ', str(it), ' | error: ', str(TrErr),  ' | G: ', g)

        # Small Scale
        elif method is 'SS':
            X = self.trX - np.average(self.trX, axis=0)
            y = self.trY - np.average(self.trY)
            A = np.dot(np.transpose(X),X)+self.trSize*la*np.eye(self.dim)
            B = np.dot(np.transpose(X), y)
            wout = -np.dot(np.linalg.inv(A), B)
       
        # Wrong Method
        else:
            print('error: regression method must exist ...\n')
            quit()

        X = np.average(self.trX, axis=0)
        y = np.average(self.trY, axis=0)
        out = np.zeros(self.dim+1)
        out[0] = y[0]-np.dot(X,wout)[0]
        out[1:] = wout[:,0]
        err = self.sqrErr(out)
        return out, err 

    # gradient of the objective function
    def gradFunc(self, w, la):
        X = self.trX - np.average(self.trX, axis=0)
        y = self.trY - np.average(self.trY, axis=0)
        A = np.dot(np.transpose(X),X)+self.trSize*la*np.eye(self.dim)
        B = np.dot(np.transpose(X), y)
        return 2*np.dot(A,w)-2*B

    # objective function for input x, y, theta and la(lambda)
    def oFunc(self, w, la):
        X = self.trX - np.average(self.trX, axis=0)
        y = self.trY - np.average(self.trY, axis=0)
        return (np.linalg.norm(y-np.dot(X,w))**2) + self.trSize*(np.linalg.norm(w)**2)

    # square Error
    def sqrErr(self, theta, x=None, y=None, n=None):
        if x is None:
            x=self.tstX
            n=self.tstSize
        if y is None:
            y=self.tstY
            n=self.tstSize
        if n is None:
            n,d = x.shape
        X = np.ones((self.tstSize,self.dim+1))
        X[:,1:] = x
        return np.linalg.norm(y[:,0]-np.dot(X,theta))**2/n
