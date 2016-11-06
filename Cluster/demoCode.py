from PIL import Image
from Cluster import cluster
import matplotlib.pyplot as plt
import numpy as np

def main():

    # paramteres
    M = 2
    k = 100

    # data parsing
    im = Image.open('mandrill.tiff')
    y = np.array(im)
    n = np.prod(y.shape)//(3*M*M)
    d = y.shape[0]
    c = 0
    x = np.zeros([n,3*M*M])
    for i in range(0,d,M):
        for j in range(0,d,M):
            x[c,:] = np.reshape(y[i:i+M,j:j+M,:], [1,M*M*3])
            c = c+1

    # clustering
    cl = cluster()
    cl.dataUpdate(x)
    errs = cl.KMeans(k)

    # reconstruction
    qX = cl.quant()
    c = 0
    yR = y
    for i in range(0,d,M):
        for j in range(0,d,M):
            yR[i:i+M,j:j+M,:] = np.reshape(qX[c], [M,M,3])
            c += 1
    imR = Image.fromarray(yR)
    imD = Image.fromarray(yR-y)
    imR.save('hw7_1.tiff')
    imD.save('hw7_2.tiff')
    
    # plotting
    t = np.arange(0, len(errs))
    plt.plot(t, errs)
    plt.show()

if __name__ == "__main__":
    main()
