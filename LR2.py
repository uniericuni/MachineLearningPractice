# Logistic Regression
#
# x: feature vector | dimxn matrix, array object
# y: lables         | nx1 matrix, array object
# theta: norm       | (dim+1)x1 matrix, array object
# la: regularizer   | constant
#
# dim: dimension of feature vector
# n: size of testing data

import numpy as np

#  sigmoid function for input x and theta
def sigmoid(x):

    return 1./(1. + np.exp(x))

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
    return (l_theta+la*np.dot(theta,theta))[0]

# gradient of the objective function
def gradFunc(x, y, theta, la):

    dim, n = x.shape                                # matrix scale
    X = np.ones((dim+1,n))                          # augmented X by first row
    X[1:,:] = x
    S = sigmoid(np.dot(theta,X*y))                  # all sigmoid as a vector
                                                    # return the matrix solution of Gradient
    return np.dot(X,np.transpose(-S*y)) 

# hessian of the objective function
def hessFunc(x, y, theta, la):

    dim, n = x.shape                                # matrix scale
    X = np.ones((dim+1,n))                          # augmented X by first row
    X[1:,:] = x
    S = sigmoid(np.dot(theta,X*y))                  # all sigmoid as a vector
    T = np.diag(S*(1-S))                            # eigen value diagonal matrix
                                                    # return the matrix solution of Hessian
    return np.dot(np.dot(X,T),np.transpose(X)) + 2*la*np.eye(dim+1)
