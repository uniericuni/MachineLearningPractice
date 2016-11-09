import matplotlib.pyplot as plt
import numpy as np
from EM import EM

def main():

    # data generating
    n = 200                     # sample size
    K = 2                       # number of lines
    e = np.array([0.7,0.3])     # mixing weights
    w = np.array([-2,1])        # slopes of lines
    b = np.array([0.5,-0.5])    # offset of lines
    v = np.array([0.2,0.1])     # variances
    x = np.zeros([n])
    y = np.zeros([n])
    for i in range(0,n):
        x[i] = np.random.rand(1)
        if np.random.rand(1) < e[0]:
            y[i] = w[0]*x[i] + b[0] + np.random.randn(1)*np.sqrt(v[0])
        else: 
            y[i] = w[1]*x[i] + b[1] + np.random.randn(1)*np.sqrt(v[1])
    
    # clustering
    theta = []
    theta.append(np.array([[.5, .5]]))
    theta.append(np.array([[-1, 1]]))
    theta.append(np.array([[0, 0]]))
    theta.append(np.array([[np.var(y), np.var(y)]]))
    em = EM()
    em.dataUpdate(x,y)
    errs = em.EM(k=2,theta=theta)
    theta = em.theta
    print('regressin results ...')
    print('------------------------------------')
    print('ans: {:>12}'.format(str(e)), '| weights:', theta[0],)
    print('ans: {:>12}'.format(str(w)), '| slope:', theta[1])
    print('ans: {:>12}'.format(str(b)), '| offset:', theta[2])
    print('ans: {:>12}'.format(str(v)), '| variance:', theta[3])
    print('------------------------------------\n')
    
    #  plotting
    plt.plot(x,y,'bo')
    t = np.linspace(0,1,num=100)
    plt.plot(t,w[0]*t+b[0],'k')
    plt.plot(t,w[1]*t+b[1],'k')
    plt.plot(t,theta[1][0]*t+theta[2][0],'r--')
    plt.plot(t,theta[1][1]*t+theta[2][1],'r--')
    plt.show()

    plt.figure()
    t = np.arange(0, len(errs))
    plt.plot(t, errs)
    plt.show()

if __name__ == "__main__":
    main()
