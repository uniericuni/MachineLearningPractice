import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from PCA import PCA

# training env setting
yale = sio.loadmat('yalefaces.mat')
yalefaces = yale['yalefaces']
"""
for i in range(0,yalefaces.shape[2]):
    x = yalefaces[:,:,i]
    fig, ax = plt.subplots()
    ax.imshow(x, extent=[0,1,0,1])
    plt.imshow(x, cmap=plt.get_cmap('gray'))
    time.sleep(0.1)
    plt.show()
"""

h,w,n = yalefaces.shape
xTr = np.zeros((h*w,n))
for i in range(0, n):
    x = yalefaces[:,:,i]
    xTr[:,i] = x.reshape((h*w))

# PCA
p = PCA()
p.dataUpdate(xTr)
p.PCA()
