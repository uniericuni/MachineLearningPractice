import numpy as np

# objective function for input x, y, theta and la(lambda)
def oFunc(x, y, theta, la):
    
    l_theta = 0
    dim, n = x.shape
    x_bar = np.ones(dim+1)
    for i in range(0,n):
        x_bar[1:] = x[:,i]
        k = 1. + np.exp(-y[:,i] * np.dot(x_bar,theta))
        l_theta += np.log(k)
         
    penalty = la * np.linalg.norm(theta)

    return (l_theta + penalty)[0]

# gradient of the objective function
def gradFunc(x, y, theta, la):

    dim, n = x.shape
    ret = np.zeros(dim+1)
    x_bar = np.ones(dim+1)
    for i in range(0,n):
        yt = -1 * y[:,i]
        xt = x[:,i]
        x_bar[1:] = x[:,i]
        theta_t = yt*theta
        sig = sigmoid(xt, theta_t)
        ret = ret + sig * yt * np.exp(np.dot(theta_t,x_bar)) * x_bar
    
    return ret

# hessian of the objective function
def hessFunc(x, y, theta, la):

    dim, n = x.shape
    ret = np.zeros(dim+1, dim+1)
    x_bar = np.ones(dim+1)
    for i in range(0,n):
        yt = -1 * y[:,i]
        xt = x[:,i]
        x_bar[1:] = x[:,i]
        theta_t = yt*theta
        sig = sigmoid(xt, theta_t)
        ret = ret + sig * (1-sig) * np.outer(x_bar, x_bar)

    return ret

# sigmoid function for input x and theta
def sigmoid(x, theta):
  
    dim, n = x.shape 
    x_bar = np.ones((dim+1,1))
    x_bar[1:] = x
    dim1, n1 = x_bar.shape
    dim2, n2 = theta.shape 
    if dim1 != dim2:
        print('dimesion mismatch')
        return 0
    
    return 1./(1. + np.exp( np.dot(x_bar,theta)))
