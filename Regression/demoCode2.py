import numpy as np
import matplotlib.pyplot as plt
import random
import pylab as pl
from Regress import Reg

# trainging setups
n = 200
random.seed()
x = np.random.rand(n, 1)
z = np.zeros([n,1])
k = int(n*0.4)
rp = np.random.permutation(n)
outlier_subset = rp[1:k]
z[outlier_subset] = 1                                                                   # outliers
y = (1-z) * (10*x + 5 + np.random.randn(n,1)) + z*(20-20*x+10*np.random.randn(n,1))
t = pl.frange(0,1,0.01)
plt.scatter(x, y, label='data')                                                         # data plotting
plt.plot(t, 10*t+5, 'k-', label='true line')                                            # ideal regression plotting

# Ordinary Least Square
rg = Reg()
rg.dataUpdate(x, y, x, y)
out, err = rg.ridgeReg()
plt.plot(t, out[1:]*t+out[0], 'g--', label='least square')

# Robust Regression MM Algrg.dataUpdate(x, y, x, y)
outR, err = rg.ridgeReg(method='RR', e=1e-14)
plt.plot(t, outR[1:]*t+outR[0], 'r:', label='robust')

#plotting
legned = plt.legend(loc='upper right', shadow=True)
plt.show()
