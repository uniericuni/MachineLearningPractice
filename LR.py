import numpy as np

# sigmoid function for input x and theta
def sigmoid(x):

    return 1./(1. + np.exp(x))

# objective function for input x, y, theta and la(lambda)
def oFunc(x, y, theta, la):
    
    l_theta = 0
    dim, n = x.shape
    x_bar = np.ones(dim+1)
    for i in range(0,n):
        x_bar[1:] = x[:,i]
        t = -np.dot(x_bar, theta)
        l_theta += y[:,i]*np.log(sigmoid(t)) + (1-y[:,i])*np.log(1-sigmoid(t))
         
    penalty = la * (np.linalg.norm(theta))**2

    return (l_theta + penalty)[0]

# gradient of the objective function
def gradFunc(x, y, theta, la):

    dim, n = x.shape
    ret = np.zeros(dim+1)
    X = np.ones((dim+1,n))
    X[1:,:] = x
    S = sigmoid( -np.dot(np.transpose(X), theta) )
    return np.dot(X, np.transpose(y-S)) + 2*la*theta 
'''    
for i in range(0,n):
        x_bar[:] = x[:,i]
        t = -np.dot(x_bar, theta)
        ret += x_bar * (y[:,i]-sigmoid(t))
    ret += 2*la*theta
 
    return ret
'''

# hessian of the objective function
def hessFunc(x, y, theta, la):

    dim, n = x.shape
    ret = np.zeros((dim+1, dim+1))
    X = np.ones((dim+1,n))
    X[1:,:] = x
    St = sigmoid( -np.dot(np.transpose(X), theta) )
    S = np.diagflat(St)
    s1 = np.dot(X, S)
    s2 = np.dot((1-S), np.transpose(X))
    return np.dot(s1,s2)-np.eye(dim+1)*2*la
'''
x_bar = np.ones(dim+1)
    for i in range(0,n):
        x_bar[1:] = x[:,i]
        t = -np.dot(x_bar, theta)
        ret += sigmoid(t)*(1-sigmoid(t))*np.outer(x_bar,x_bar)        
    ret += 2*la
    print ret

    return ret
'''
