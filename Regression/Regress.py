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
        self.clear()

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

    # Ridge Regression
    def ridgeReg(self, theta=None, la=0, e=1e-14, alpha=1e-3, method='SLSR'):
        it = 0                                          # iterator
        err = 0                                         # error rate
        pre = -float('Inf')                             # previous value offset
        if theta is None:                               # regression starting point
            out = np.zeros(self.dim+1)
        else:
            out = theta 

        # Robust Regression
        if method is 'RR':
            print('robut regresion ...')
            f = open('trainingDetail.txt', 'w')         # record training detail
            while abs(np.linalg.norm(out)-pre) > e:     # not terminate until the minimizer barely changes
                pre = np.linalg.norm(out)
                it += 1
                out = self.sol(True, rtn=out)           # minimizer for this iteration
                msg = 'iteration: ' + str(it) + ' | norm of theta: ' + str(np.linalg.norm(out)) + ' | error: ' + str(self.oFunc(out))
                f.write(msg)
            f.close()

        # Small Scale Least Squre Regression
        elif method is 'SLSR':
            print('ordinary least square regression ...')
            out = self.sol(False, la=la)                # minimizer
        
        # Wrong Method
        else:
            print('error: regression method must exist ...\n')
            quit()
        
        err = self.oFunc(out)                           # measuring the square error
        print('------------------------------------')
        print('theta', out)
        print('error', err)
        print('------------------------------------\n')
        return out, err 
    
    # solution to regression
    def sol(self, weight, x=None, y=None, la=0, n=None, rtn=None):
        if x is None:
            x = self.trX
        if y is None:
            y = self.trY
        if n is None:
            n, d = x.shape
        if rtn is None:
            rtn = np.zeros(self.dim+1)

        X = x - np.average(x, axis=0)                                   # X bar
        Y = y - np.average(y)                                           # Y bar
        if weight:                                                      # whether it is weighted or not
            C = np.diag(self.psi(Y[:,0]-np.dot(X,rtn[1:])-rtn[0])) 
        else:
            C = np.eye(n)
        A = np.dot(np.transpose(X),np.dot(C,X))+n*la*np.eye(self.dim)   # A
        B = np.dot(np.transpose(X), np.dot(C,Y))                        # B
        wout = np.dot(np.linalg.inv(A), B)                              # solution to the minimizer
        b = np.average(np.dot(C,y)) - np.dot(np.average(np.dot(C,x), axis=0), wout)[0]
        rtn[0] = b
        rtn[1:] = wout[:,0]
        return rtn

    # objective function for input x, y, theta and la(lambda)
    def oFunc(self, theta, x=None, y=None, la=0, n=None):
        if x is None:
            x=self.tstX
        if y is None:
            y=self.tstY
        if n is None:
            n,d = x.shape
        X = np.ones((self.tstSize,self.dim+1))
        X[:,1:] = x
        return (np.linalg.norm(y[:,0]-np.dot(X,theta))**2)/n + la * np.linalg.norm(theta[1:])**2  # return the square error with regularizing penalty
    
    # psi/r
    def psi(self, r):
        return 1/np.sqrt(1+r**2)                                    # psi/r weight of a given rho 
        
