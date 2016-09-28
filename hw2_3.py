# This program uses Python3 to compile so that the user doesn't need to worried about floating point declaration
import random
import numpy as np

# quantization function to quantize x by median
# x: input feature vector
# med: median value of the vector
# m: 1 or 2, for > or >=
def quant(x, med, m):
    dim = x.shape                                               # measuring dimension
    assert (m==1 or m==2), 'method must be 1 or 2'              # system assert
    if m is 1:                                                  # evaluate element-wise truth value
        com = np.greater(x, med)                                
    else:
        com = np.greater_equal(x, med)                        
    return np.where(com, np.ones(dim), np.zeros(dim))           # translate truth value to 1 and 0

# training env setting
trainingNum = 2000                                              # size of training data
z = np.genfromtxt('spambase.data', dtype=float, delimiter=',')  # load data
random.seed()                                                   # rSeed the random number generator
rp = np.random.permutation(z.shape[0])                          # random permutation of indices
z = z[rp,:]                                                     # shuffle the rows of z
x = z[:,:-1]                                                    # feature vectors
y = z[:,-1]                                                     # labels
# training and testing data seperation
xT = x[0:trainingNum,:]
yT = y[0:trainingNum]
xTest = x[trainingNum:,:]
yTest = y[trainingNum:]
qy = yT
 
# using quantization method1
# training
n, dim = xT.shape                                               # measuring dimension and size
m = 1                                                           # method
xMedian = np.median(xT, axis=0)                                 # calculating median of x
yMedian = np.median(yT, axis=0)                                 # calculating median of y
q1 = np.zeros(dim)                                              # initialization of counter1
for i in range(0,trainingNum):                                  # accumulating quantized x being 1 under y being 1
    q1 += quant(xT[i,:],xMedian,m)*qy[i]
p1 = q1/sum(qy)                                                 # this is the prob of Y=1|X=1
# given y is 0, x is 1
q2 = np.zeros(dim)                                              # initialization of counter2
for i in range(0,trainingNum):                                  # accumulating quantized x being 1 under y being 2
    q2 += quant(xT[i,:],xMedian,m)*(1-qy[i])
p2 = q2/sum(1-qy)                                               # this is the prob of Y=0|X=1

# testing
n, dim = xTest.shape
error1 = 0
for i in range(0,n):
    qx = quant(xTest[i,:], xMedian, m)                          # quantized testing vector
    com = np.equal(qx, np.ones(dim))                            # determine whether the quantized x is one
    prob1 = sum(qy)*np.prod( np.where(com, p1, 1-p1) )          # calculate probability if labeled 1
    prob2 = sum(1-qy)*np.prod( np.where(com, p2, 1-p2) )        # calculate probability if labeled 0
    if prob1 > prob2:                                           # choose the larger probability
        label = 1
    else:
        label = 0
    if label != yTest[i]:                                       # check if we have the correct prediction
        error1 += 1

# using quantization method2
# training
n, dim = xT.shape                                               # measuring dimension and size
m = 2                                                           # method
xMedian = np.median(xT, axis=0)                                 # calculating median of x
yMedian = np.median(yT, axis=0)                                 # calculating median of y
q1 = np.zeros(dim)                                              # initialization of counter1
for i in range(0,trainingNum):                                  # accumulating quantized x being 1 under y being 1
    q1 += quant(xT[i,:],xMedian,m)*qy[i]
p1 = q1/sum(qy)                                                 # this is the prob of Y=1|X=1
# given y is 0, x is 1
q2 = np.zeros(dim)                                              # initialization of counter2
for i in range(0,trainingNum):                                  # accumulating quantized x being 1 under y being 2
    q2 += quant(xT[i,:],xMedian,m)*(1-qy[i])
p2 = q2/sum(1-qy)                                               # this is the prob of Y=0|X=1

# testing
n, dim = xTest.shape
error2 = 0
for i in range(0,n):
    qx = quant(xTest[i,:], xMedian, m)                          # quantized testing vector
    com = np.equal(qx, np.ones(dim))                            # determine whether the quantized x is one

    prob1 = sum(qy)*np.prod( np.where(com, p1, 1-p1) )          # calculate probability if labeled 1
    prob2 = sum(1-qy)*np.prod( np.where(com, p2, 1-p2) )        # calculate probability if labeled 0
    if prob1 > prob2:                                           # choose the larger probability
        label = 1
    else:
        label = 0
    if label != yTest[i]:                                       # check if we have the correct prediction
        error2 += 1


print('  method1 predicting error rate: ', error1/n)
print('  method2 predicting error rate: ', error2/n)
print('majority classifying error rate: ', min(sum(qy)/trainingNum, sum(1-qy)/trainingNum))
