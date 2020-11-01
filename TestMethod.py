import numpy as np
import cvxpy as cp


def VanillaSVM(xtrain, ytrain, xtest, ytest):
    # vanilla SVM
    print('vanilla SVM')
    # formulate the problem
    n, d = xtrain.shape
    c = np.ones(n)
    e = cp.Variable(n)
    w = cp.Variable(d)
    b = cp.Variable(1)
    prob = cp.Problem(cp.Minimize(c.T@e), [e>=(1-cp.multiply(np.squeeze(ytrain),(xtrain@w+b))),e>=0])
    prob.solve('SCS')

    # get weights
    w = w.value
    b = b.value
    #w = np.array(x.value[0:d])
    #b = np.array(x.value[d + 1])

    # get the result for training set
    y_pred = (xtrain.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in training set: ' + str(np.sum(y_pred == ytrain) / ytrain.shape[0]))

    #get result for testing set
    y_pred = (xtest.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in testing set:  ' + str(np.sum(y_pred == ytest) / ytest.shape[0]))

    return w,b

def VanillaSVM_WithbReg(xtrain, ytrain, xtest, ytest):
    # SVM to shrink b
    print('\nSVM to shrink b')
    # formulate the problem
    n, d = xtrain.shape
    c = np.ones(n)

    e = cp.Variable(n)
    lambda_reg = 0.1
    w = cp.Variable(d)

    b = cp.Variable(1)
    b1 = cp.pnorm(b,p=1)
    prob = cp.Problem(cp.Minimize(c.T@e+lambda_reg*b1), [e>=(1-cp.multiply(np.squeeze(ytrain),(xtrain@w+b))),e>=0])
    prob.solve(solver='SCS')
    print('done')
    # get weights
    w = w.value
    b = b.value

    # get the result for training set
    y_pred = (xtrain.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in training set: ' + str(np.sum(y_pred == ytrain) / ytrain.shape[0]))

    #get result for testing set
    y_pred = (xtest.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in testing set:  ' + str(np.sum(y_pred == ytest) / ytest.shape[0]))

    return w,b

def VanillaSVM_WithwReg(xtrain, ytrain, xtest, ytest):
    # SVM to shrink b
    print('\nSVM to shrink w')
    # formulate the problem
    n, d = xtrain.shape
    c = np.ones(n)

    e = cp.Variable(n)
    lambda_reg = 0.01
    w = cp.Variable(d)

    b = cp.Variable(1)
    w2 = cp.pnorm(w,p=2)**2
    prob = cp.Problem(cp.Minimize(c.T@e+lambda_reg*w2), [e>=(1-cp.multiply(np.squeeze(ytrain),(xtrain@w+b))),e>=0])
    prob.solve(solver='SCS')
    print('done')
    # get weights
    w = w.value
    b = b.value

    # get the result for training set
    y_pred = (xtrain.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in training set: ' + str(np.sum(y_pred == ytrain) / ytrain.shape[0]))

    #get result for testing set
    y_pred = (xtest.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in testing set:  ' + str(np.sum(y_pred == ytest) / ytest.shape[0]))
    return w,b
    
def VanillaSVM_WithwandbReg(xtrain, ytrain, xtest, ytest):
    # SVM to shrink b
    print('\nSVM to shrink w and b')
    # formulate the problem
    n, d = xtrain.shape
    c = np.ones(n)

    e = cp.Variable(n)
    lambda_reg = 0.01
    w = cp.Variable(d)

    b = cp.Variable(1)
    b1 = cp.pnorm(b,p=1)
    w2 = cp.pnorm(w,p=2)**2
    prob = cp.Problem(cp.Minimize(c.T@e+lambda_reg*w2+0.1*b1), [e>=(1-cp.multiply(np.squeeze(ytrain),(xtrain@w+b))),e>=0])
    prob.solve(solver='SCS')  
    print('done')
    # get weights
    w = w.value
    b = b.value

    # get the result for training set
    y_pred = (xtrain.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in training set: ' + str(np.sum(y_pred == ytrain) / ytrain.shape[0]))

    #get result for testing set
    y_pred = (xtest.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in testing set:  ' + str(np.sum(y_pred == ytest) / ytest.shape[0]))

    return w,b
def TestAvgAcc(w,b,xtest,ytest):
    # get the result for testing set+
    accuracies = []
    for _ in range(10):
        y_pred = (xtest.dot(w) + b > 0) * 2 - 1
        y_pred = y_pred.reshape(y_pred.shape[0],-1)
        accuracy = (y_pred == ytest).sum()/ytest.shape[0]
        accuracies.append(accuracy)
    return accuracies

def AdversarySVM(xtrain, ytrain, xtest, ytest):
    # adversary model
    print('\nAdversary Model')
    # keep = 0.1, keep 10% of training data
    print("shape of traning: " + str(xtrain.shape))
    n, d = xtrain.shape
    YX = xtrain * ytrain
    li = np.ones([1, n])
    ri = np.ones([d, 1])
    # K = 73, just remove 10% of features
    K = int(d * 0.1)

    # get variables
    w = cp.Variable((d, 1))
    v = cp.Variable((n, d))
    z = cp.Variable((n, 1))
    t = cp.Variable((n, 1))
    s = cp.Variable((n, 1))

    # solve
    prob = cp.Problem(cp.Minimize(li @ s),
                      [t >= K * z + v @ ri,
                       z + v >= cp.multiply(YX, w.T),
                       s >= 1 - YX @ w + t,
                       s >= 0,
                       v >= 0])
    prob.solve('SCS')
    # get weights
    w = np.array(w.value)
    b = 0
    # get the result for training set
    y_pred = (xtrain.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in training set: ' + str(np.sum(y_pred == ytrain) / ytrain.shape[0]))

    # get the result for testing set
    y_pred = (xtest.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in testing set:  ' + str(np.sum(y_pred == ytest) / ytest.shape[0]))

    return w,b
def ModifiedDataSVM(xtrain, ytrain, xtest, ytest):
    # svm of modified data

    print('\nSVM of modified data')
    # formulate the problem
    n, d = xtrain.shape
    c = np.ones(n)

    e = cp.Variable(n)
    lambda_reg = 0.1
    w = cp.Variable(d)

    b = cp.Variable(1)
    w1 = cp.pnorm(b,p=2)**2
    prob = cp.Problem(cp.Minimize(c.T@e+lambda_reg*w1), [e>=(1-multiply(np.squeeze(ytrain),(xtrain@w+b))),e>=0])
    prob.solve(solver='SCS')
    print('done')
    # get weights
    w = w.value
    b = b.value

    # get weights
    #w = np.array(x.value[0:d])
    #b = np.array(x.value[d + 1])

    # get the result for training set
    y_pred = (xtrain.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in training set: ' + str(np.sum(y_pred == ytrain) / ytrain.shape[0]))

    # get the result for testing set
    y_pred = (xtest.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in testing set:  ' + str(np.sum(y_pred == ytest) / ytest.shape[0]))
    return w,b

def Vanilla_multiSVM(xtrain, ytrain, xtest, ytest):
    print('\nMulti-class Model')
    n, d = xtrain.shape
    k = ytrain.shape[1]

    # get variables
    w = cp.Variable((d, k))
    s = cp.Variable((n,1))
    ri = np.ones((k, 1))
    li = np.ones((n, 1))
    # solve
    prob = cp.Problem(cp.Minimize(li.T@s),
                      [s >= (xtrain @ w - cp.multiply(xtrain @ w, ytrain) @ ri + 1)@ri,
                       s >= 0])
    prob.solve('SCS')
    # get weights
    w = np.array(w.value)
    b = 0
    # get the result for training set
    y_pred = xtrain.dot(w) + b
    rate = np.sum(np.argmax(y_pred, axis=1) == np.argmax(ytrain, axis=1)) / (ytrain.shape[0])
    print('the accuracy in testing set:  ' + str(rate))

    # get the result for testing set
    y_pred = xtest.dot(w) + b
    rate = np.sum(np.argmax(y_pred, axis=1) == np.argmax(ytest, axis=1)) / (ytest.shape[0])
    print('the accuracy in testing set:  ' + str(rate))
    return w,b

def ModifiedData_multiSVM(xtrain, ytrain, xtest, ytest):
    print('\nMulti-class Model with modified data')
    n, d = xtrain.shape
    k = ytrain.shape[1]
    inverted_y = 1-ytrain
    print(inverted_y)
    # get variables
    w = cp.Variable((d, k))
    s = cp.Variable((n,1))
    ri = np.ones((k, 1))
    li = np.ones((n,1))
    w1 = cp.pnorm(w,p=1)
    # solve

    prob = cp.Problem(cp.Minimize(cp.sum(s)+0.01*w1),
                      [s >= cp.multiply((1+ xtrain @ w - cp.multiply(xtrain @ w, ytrain)@ri),inverted_y)@ri,
                       #s >= 1 + ((xtrain @ w - cp.multiply(xtrain @ w, ytrain)@ri))@ri,
                       s >= 0])
    prob.solve('SCS')
    # get weights
    w = np.array(w.value)
    b = 0
    # get the result for training set
    '''
    y_pred = xtrain.dot(w) + b
    rate = np.sum(np.argmax(y_pred, axis=1) == np.argmax(ytrain, axis=1)) / (ytrain.shape[0])
    print('the accuracy in testing set:  ' + str(rate))

    # get the result for testing set
    y_pred = xtest.dot(w) + b
    rate = np.sum(np.argmax(y_pred, axis=1) == np.argmax(ytest, axis=1)) / (ytest.shape[0])
    print('the accuracy in testing set:  ' + str(rate))
    '''
    return w,b