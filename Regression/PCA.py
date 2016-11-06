import numpy as np
import matplotlib.pyplot as plt
import random

# Regression Object
class PCA:

    # constructor
    def __init__(self):
        self.clear()
        random.seed(0)

    # clear regression object
    def clear(self):
        self.trX = None
        self.dim = None
        self.size = None
    
    # data parsing    
    def dataUpdate(self, trX):
        self.dim, self.size = trX.shape                 # training size and dimension
        self.trX = trX                                  # training X
        print('\ndata parsed ...')
        print('------------------------------------')
        print('ipnut X ... dim: ', self.dim, ' | size: ', self.size)
        print('------------------------------------\n')

    # PCA
    def PCA(self):
        U, S, V = np.linalg.svd(self.trX, full_matrices=False)  # SVD solution for PCA
        n = S.shape[0]
        t = np.arange(0, n, 1)
        self.eigenVectors = U
        self.singularValuess = S
        
        # singular value distribution
        fig1 = plt.figure()
        plt.semilogy(t, S**2/n)
        plt.title('histogram of eigenvalues')
        plt.show()
 
        # variation percentage 
        totalVar = np.var(np.dot(np.dot(U,np.diag(S)),V))
        var = np.zeros(n)
        vP = 0
        f = open('PCA.txt', 'w') 
        print('variation growth ...')
        print('------------------------------------')
        for k in range(0,n):
            Sk = np.zeros((self.dim, self.dim))         # proojection result
            Sk[0:k+1, 0:k+1] = np.diag(S[:k+1])
            Zk = np.dot(np.dot(U, Sk), V)              
            var[k] = np.var(Zk)                         # variation
            vP = var[k]/totalVar                        # variation percentage
            msg = 'PCs dimension:' + '{:>3}'.format(str(k+1)) + ', %.3f'%((k+1)*100/self.dim) + '% | variation percentage: ' + '|'*int(vP*100) + ' %.4f'%vP + ' %'
            f.write(msg+'\n')
            print(msg, end='\r')                        # variation growth as message
            if vP > 0.99: break
        f.close()
        t = np.arange(0, k, 1)
        fig2 = plt.figure()
        plt.semilogy(t, var[:k])
        plt.title('histogram of variance')
        print(msg)
        print('done')
        print('------------------------------------\n')
        plt.show()
