# Logistic Regression
#
# x: feature vector | dimxn matrix, array object
# y: lables         | nx1 matrix, array object
# theta: norm       | (dim+1)x1 matrix, array object
# la: regularizer   | constant
#
# dim: dimension of feature vector
# n: size of testing data

import scipy.io as sio
import numpy as np

#  sigmoid function for input x and theta
def sigmoid(x):

    return 1./(1. + np.exp(-x))

# objective function for input x, y, theta and la(lambda)
def oFunc(x, y, theta, la):
    
    l_theta = 0                                     # return value initialization
    dim, n = x.shape                                # matrix scale
    x_bar = np.ones(dim+1)                          # x_bar initialization
    for i in range(0,n):
        x_bar[1:] = x[:,i]                          # augmented x by first row
        e = np.exp(-y[:,i]*np.dot(x_bar,theta))     # sigmoid function
        l_theta += np.log(1+e)                      # MLE accumulation
                                                    # return - MLE + penalty
    return (l_theta+la*(np.linalg.norm(theta)**2))[0]

# gradient of the objective function
def gradFunc(x, y, theta, la):

    dim, n = x.shape                                # matrix scale
    X = np.ones((dim+1,n))                          # augmented X with additional row of 1s
    X[1:,:] = x
    S = sigmoid(np.dot(theta,X*y))-1                # all sigmoid as a vector
                                                    # return the matrix solution of Gradient
    return  np.dot(S*y,np.transpose(X))+2*la*theta 

# hessian of the objective function
def hessFunc(x, y, theta, la):

    dim, n = x.shape                                # matrix scale
    X = np.ones((dim+1,n))                          # augmented X with additional row of 1s
    X[1:,:] = x
    S = sigmoid(-np.dot(theta,X*y))                 # all sigmoid as a vector
    T = np.diag(S*(1-S))                            # eigen value diagonal matrix
                                                    # return the matrix solution of Hessian
    return np.dot(np.dot(X,T),np.transpose(X)) + 2*la*np.eye(dim+1)

# Newton optimizatino
def newton(x, y, default, la):
    bound = 1e-14                                               # low-bound
    it = 0                                                      # iterator
    theta = default                                             # starting point
    error = 0                                                   # error rate
    g = float('Inf')                                            # gradient norm offset
    pre = -float('Inf')                                         # previous gradient norm offset
    gList = []                                                  
    eList = []
    tList = []

    while abs(g-pre) > bound:                                   # iterate until the gradient norm barely changes
        pre = g
        it += 1                                                 # iterator
        G = gradFunc(x, y, theta, la)                           # gradient
        g = np.linalg.norm(G)                                   # gradient norm
        H = hessFunc(x, y, theta, la)                           # hessian
        theta -= np.dot(np.linalg.inv(H),np.transpose(G))[:,0]  # Newton method
        error = oFunc(x, y, theta, la)                          # objective function
        print('error: ', str(error), '| iteration: ', str(it), '| G: ', g)
        gList.append(G)
        eList.append(error)
        tList.append(theta)

    # export training detail
    sio.savemat('trainingDetail.mat', {'theta_List':tList, 'gradient_List':gList, 'error_List':eList, 'bound':bound})
    return theta
