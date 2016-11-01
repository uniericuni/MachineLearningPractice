# u: first input feature
# v: second input feature
import numpy as np
from dist2 import dist2

# Gaussian Kernel
def GaussKernel(u, v):
    if not hasattr(GaussKernel, 'sigma'):
        GaussKernel.sigma = 1.0  
    return np.exp((-1/(2*(GaussKernel.sigma**2)))*dist2(u,v))

# Cauchy Kernel
def CauchKernel(u, v):
    if not hasattr(CauchKernel, 'sigma'):
        CauchKernel.sigma = 1.0
    return (1+dist2(u,v)/CauchKernel.sigma**2)**(-1)
