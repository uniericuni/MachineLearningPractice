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
        U, S, V = np.linalg.svd(self.trX, full_matrices=False)
        n = S.shape[0]
        t = np.arange(0, n, 1)
        """
        fig1 = plt.figure()
        plt.semilogy(t, S)
        plt.title('histogram of singular values')
        plt.show()
        """
        
        print('\nvariation growth ...')
        print('------------------------------------')
        totalVar = np.var(np.dot(np.dot(U,np.diag(S)),V))
        var = np.zeros(n)
        vP = 0
        for k in range(0,n):
            f = open('PCA.txt', 'w')                    # reduction
            Sk = np.zeros((self.dim, self.dim))
            Sk[0:k+1, 0:k+1] = np.diag(S[:k+1])
            Zk = np.dot(np.dot(U, Sk), V)
            var[k] = np.var(Zk)
            vP = var[k]/totalVar
            msg = 'principle components: ' + '{:>3}'.format(str(k+1)) + '| variation percentage: ' + '|'*int(vP*100) + ' %.4f'%vP + ' %'
            print(msg, end='\r')
            if vP > 0.99: break
        f.close()
        t = np.arange(0, k, 1)
        fig2 = plt.figure()
        plt.semilogy(t, var[:k])
        plt.title('histogram of variance')
        print('\n------------------------------------\n')
        plt.show()
