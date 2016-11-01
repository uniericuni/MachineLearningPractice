import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time
from PCA import PCA

# training env setting
yale = sio.loadmat('yalefaces.mat')
yalefaces = yale['yalefaces']

"""
# eignefaces plotting
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

# eigen vector plotting
n = 20
u = p.eigenVectors[:,:n]
u = u.reshape((h,w,n)) 
u[:,:,1:] = u[:,:,:-1]                                  # substituting the first image with average
u[:,:,0] = np.average(yalefaces, axis=2)
f, axs = plt.subplots(4, 5, sharex='col', sharey='row')
for i in range(0,4):
    for j in range(0,5):
        index = i*5+j
        im = u[:,:,index]
        ax = axs[i,j]
        ax.imshow(im, cmap=plt.get_cmap('gray'))
        ax.set_title(str(index+1)+'th eigenface')
plt.show()
