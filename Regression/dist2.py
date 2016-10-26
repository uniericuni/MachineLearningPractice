import numpy as np

def dist2(x,c):
    ndata,dimx = x.shape
    ncenters, dimc = c.shape

    xsum = np.sum(x**2,axis = 1)
    xsum = xsum[:,np.newaxis]
    csum = np.sum(c**2,axis = 1)
    csum = csum[:,np.newaxis]

    n2 =  xsum.dot(np.ones([1,ncenters]))+ np.ones([ndata,1]).dot(csum.T)- 2*x.dot(c.T)
    return n2
