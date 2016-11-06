from dist2 import dist2
import numpy as np

# Cluster Object
class cluster:

    # constructor
    def __init__(self):
        self.clear()
        np.random.seed(0)

    # clear regression object
    def clear(self):
        self.x = None
        self.dim = None
        self.size = None
    
    # data parsing    
    def dataUpdate(self, x):
        self.size, self.dim = x.shape               # data size and dimension
        self.x = x                                  # data
        print('\ndata parsed ...')
        print('------------------------------------')
        print('training X ... dim: ', self.dim, ' | size: ', self.size)
        print('------------------------------------\n')

    # KMeans
    def KMeans(self,k):
        errs = []
        it = 0
        self.k = k
        prev_c = np.zeros(self.size)
        self.c = np.ones(self.size)
        self.m = self.initialize()
        print('clustering ...')
        print('------------------------------------')
        while np.any(prev_c != self.c):
            err = self.oFunc()
            prev_c = self.c                                                             # preserve previsou result
            self.c = dist2(self.x,self.m).argmin(axis=1)                                # assign cluster by distance
            m = [self.x[np.where(self.c==i)].mean(axis=0).tolist() for i in range(0,self.k)]
            self.m = np.array(m)                                                        # assign cluster centers
            err = self.oFunc()                                                          # objective error
            errs.append(err)
            it += 1
            print('iteration:', '{:>3}'.format(str(it)), '| error: %.3f'%err, ' | current cluster: ', self.c)
        print('------------------------------------\n')
        return errs

    # objective function
    def oFunc(self):
        return  dist2(self.x, self.m).min(axis=1).sum()

    # cluster initialization
    def initialize(self):
        perm = np.random.permutation(self.size)
        return self.x[perm[0:self.k],:]

    # cluster quantization
    def quant(self, c=None, m=None):
        if c==None: c=self.c
        if m==None: m=self.m
        return [m[c[i]] for i in range(0,self.size)]

