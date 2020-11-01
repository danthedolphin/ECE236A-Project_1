import numpy as np
import cvxpy as cp


def VanillaSVM(xtrain, ytrain, xtest, ytest):
    # vanilla SVM
    print('vanilla SVM')
    # formulate the problem
    n, d = xtrain.shape
    c = np.concatenate((np.zeros([d + 1]), np.ones([n])))
    A1 = np.concatenate((-ytrain * xtrain, -ytrain, -np.diag(np.ones(n))), axis=1)
    A2 = np.concatenate((0.0 * xtrain, 0.0 * ytrain, -np.diag(np.ones(n))), axis=1)
    A = np.concatenate((A1, A2), axis=0)
    b = np.concatenate((-np.ones([n]), np.zeros([n])))
    print('shape of c: ' + str(c.shape))
    print('shape of A: ' + str(A.shape))
    print('shape of b: ' + str(b.shape))

    # solve
    x = cp.Variable(n + d + 1)
    prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
    prob.solve()

    # get weights
    w = np.array(x.value[0:d])
    b = np.array(x.value[d + 1])

    # get the result for training set
    y_pred = (xtrain.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in training set: ' + str(np.sum(y_pred == ytrain) / ytrain.shape[0]))

    # get the result for testing set
    y_pred = (xtest.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in testing set:  ' + str(np.sum(y_pred == ytest) / ytest.shape[0]))


def VanillaSVM_WithReg(xtrain, ytrain, xtest, ytest):
    # SVM to shrink b
    print('\nSVM to shrink b')
    # formulate the problem
    n, d = xtrain.shape
    c = np.concatenate((np.zeros([d + 1]), np.ones([n]), np.array([0.5])))
    A1 = np.concatenate((-ytrain * xtrain, -ytrain, -np.diag(np.ones(n)), np.zeros([n, 1])), axis=1)
    A2 = np.concatenate((0.0 * xtrain, 0.0 * ytrain, -np.diag(np.ones(n)), np.zeros([n, 1])), axis=1)
    A3 = np.concatenate((np.zeros([2, d]), np.array([[1], [-1]]), np.zeros([2, n]), np.array([[-1], [-1]])), axis=1)
    A = np.concatenate((A1, A2, A3), axis=0)
    b = np.concatenate((-np.ones([n]), np.zeros([n + 2])))
    print('shape of c: ' + str(c.shape))
    print('shape of A: ' + str(A.shape))
    print('shape of b: ' + str(b.shape))

    # solve
    x = cp.Variable(n + d + 2)
    prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
    prob.solve()

    # get weights
    w = np.array(x.value[0:d])
    b = np.array(x.value[d + 1])

    # get the result for training set
    y_pred = (xtrain.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in training set: ' + str(np.sum(y_pred == ytrain) / ytrain.shape[0]))

    # get the result for testing set
    y_pred = (xtest.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in testing set:  ' + str(np.sum(y_pred == ytest) / ytest.shape[0]))


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


def ModifiedDataSVM(xtrain, ytrain, xtest, ytest):
    # svm of modified data

    print('\nSVM of modified data')
    # formulate the problem
    n, d = xtrain.shape
    c = np.concatenate((np.zeros([d + 1]), np.ones([n]), np.array([0.1])))
    A1 = np.concatenate((-ytrain * xtrain, -ytrain, -np.diag(np.ones(n)), np.zeros([n, 1])), axis=1)
    A2 = np.concatenate((0.0 * xtrain, 0.0 * ytrain, -np.diag(np.ones(n)), np.zeros([n, 1])), axis=1)
    A3 = np.concatenate((np.zeros([2, d]), np.array([[1], [-1]]), np.zeros([2, n]), np.array([[-1], [-1]])), axis=1)
    A = np.concatenate((A1, A2, A3), axis=0)
    b = np.concatenate((-np.ones([n]), np.zeros([n + 2])))
    print('shape of c: ' + str(c.shape))
    print('shape of A: ' + str(A.shape))
    print('shape of b: ' + str(b.shape))

    # solve
    x = cp.Variable(n + d + 2)
    prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
    prob.solve('SCS')

    # get weights
    w = np.array(x.value[0:d])
    b = np.array(x.value[d + 1])

    # get the result for training set
    y_pred = (xtrain.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in training set: ' + str(np.sum(y_pred == ytrain) / ytrain.shape[0]))

    # get the result for testing set
    y_pred = (xtest.dot(w) + b > 0) * 2 - 1
    y_pred = y_pred.reshape(y_pred.shape[0], -1)
    print('the accuracy in testing set:  ' + str(np.sum(y_pred == ytest) / ytest.shape[0]))


def Vanilla_multiSVM(xtrain, ytrain, xtest, ytest):
    print('\nMulti-class Model')
    n, d = xtrain.shape
    k = ytrain.shape[1]

    # get variables
    w = cp.Variable((d, k))
    s = cp.Variable((n, k))
    ri = np.ones((k, 1))
    li = np.ones((n, 1))
    # solve
    prob = cp.Problem(cp.Minimize(li.T @ s @ ri),
                      [s >= xtrain @ w - cp.multiply(xtrain @ w, ytrain) @ ri + 1,
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


def ModifiedData_multiSVM(xtrain, ytrain, xtest, ytest):
    print('\nMulti-class Model with modified data')
    n, d = xtrain.shape
    k = ytrain.shape[1]

    # get variables
    w = cp.Variable((d, k))
    s = cp.Variable((n, k))
    ri = np.ones((k, 1))
    li = np.ones((n, 1))
    # solve
    prob = cp.Problem(cp.Minimize(li.T @ s @ ri),
                      [s >= xtrain @ w - cp.multiply(xtrain @ w, ytrain) @ ri + 1,
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