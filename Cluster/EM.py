# theta[0]: epsilon
# theta[1]: w
# theta[2]: b
# theta[3]: sigma

import numpy as np
from Cluster import cluster

class EM(cluster):

    # data parsing    
    def dataUpdate(self, x, y):
        self.x = x.reshape((-1,1))
        self.y = y.reshape((-1,1))
        self.size, self.dim = self.x.shape          # data size and dimension
        s,d = self.y.shape                          # data size and dimension
        print('\ndata parsed ...')
        print('------------------------------------')
        print('training X ... dim: ', self.dim, ' | size: ', self.size)
        print('training Y ... dim: ', d, ' | size: ', s)
        print('------------------------------------\n')

    def EM(self, k, theta=None, halt=10e-4):
        errs = []
        it = 0
        self.k = k
        self.initialize(theta)
        err = self.oFunc()
        prev_err = float('inf')

        print('clustering ...')
        print('------------------------------------')
        while abs(err-prev_err)>halt:
            prev_err = err
            self.Estep()
            self.Mstep()
            err = self.oFunc()
            errs.append(err)
            it += 1
            print('iteration:', '{:>3}'.format(str(it)), '| error: %.3f'%err, '| error difference: %.5f'%abs(err-prev_err))
        print('------------------------------------\n')
        return errs

    def initialize(self, theta=None):
        if theta == None: theta = [np.zeros(self.k)]*4
        self.theta = theta
    
    def Estep(self):
        g = self.gauss()
        e = self.theta[0]
        self.r = g*e/np.sum(g*e,axis=1).reshape((-1,1))
        
    def Mstep(self):
        x = self.x
        y = self.y
        x_bar = np.sum(x*self.r,axis=0)/np.sum(self.r,axis=0)
        y_bar = np.sum(y*self.r,axis=0)/np.sum(self.r,axis=0)
        x_til = (x-x_bar)
        y_til = (y-y_bar)
        self.theta[0] = np.average(self.r,axis=0)
        self.theta[1] = (((x_til*x_til*self.r).sum(axis=0))**(-1)) * (x_til*y_til*self.r).sum(axis=0)
        self.theta[2] = (y_bar - self.theta[1]*x_bar)
        r = self.y-self.x*self.theta[1]-self.theta[2]
        self.theta[3] = (self.r*r*r).sum(axis=0)/self.r.sum(axis=0)

    def oFunc(self):
        g = self.gauss()
        e = self.theta[0]
        return np.abs(np.sum(np.log(np.sum(g*e,axis=0))))

    def gauss(self):
        theta = self.theta
        r = self.y-self.x*theta[1]-theta[2]
        sigma = theta[3]
        return (2*np.pi)**(-self.dim/2) * np.abs(sigma)**(-1/2) * np.exp(-r*r*(sigma**(-1))/2)

