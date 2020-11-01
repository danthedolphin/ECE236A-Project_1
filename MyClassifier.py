# -*- coding: utf-8 -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""
import pandas as pd
import numpy as np
import cvxpy as cp
from DataProcessor import *


class MyClassifier:
    def __init__(self,K,M):
        self.K = K   # Number of classes
        self.M = M   # Number of features
        self.W = []  # Weight
        self.w = []  # Bias
        # increase this value will lead to a higher memory used 1.4 in K=2 will use 21.2Gb for Minist
        # but a higher memorycontrol value will lead to higher accuracy
        self.memorycontrol = 1.4

    def __preprocess(self, train_data, train_label, p):
        # change to array
        data = np.array(train_data)
        label = np.array(train_label)
        # change to 2-d matrix
        data = data.reshape(data.shape[0], self.M)
        label = label.reshape(label.shape[0], -1)
        # change type
        data = data.astype('uint8')
        label = label.astype('int8')
        # change label
        label2tag, tag2label, label = get_label(label, self.K)
        self.label2tag = label2tag
        self.tag2label = tag2label
        # get training set
        p = [max(0.01, p - 0.2), p, min(1, p + 0.2)]
        train, label = drop_data(data, label, p, self.K, self.memorycontrol)
        return train, label

    def train(self, p, train_data, train_label):
        # THIS IS WHERE YOU SHOULD WRITE YOUR TRAINING FUNCTION
        #
        # The inputs to this function are:
        #
        # self: a reference to the classifier object.
        # train_data: a matrix of dimesions N_train x M, where N_train
        # is the number of inputs used for training. Each row is an
        # input vector.
        # trainLabel: a vector of length N_train. Each element is the
        # label for the corresponding input column vector in trainData.
        #
        # Make sure that your code sets the classifier parameters after
        # training. For example, your code should include a line that
        # looks like "self.W = a" and "self.w = b" for some variables "a"
        # and "b".
        xtrain, ytrain = self.__preprocess(train_data, train_label, p)
        print('\nshape of modified training set: ' + str(xtrain.shape))
        print('shape of modified lbaels      : ' + str(ytrain.shape))

        # formulate the problem
        if self.K == 2:
            n, d = xtrain.shape
            c = np.concatenate((np.zeros([d + 1]), np.ones([n]), np.array([0.1])))
            A1 = np.concatenate((-ytrain * xtrain, -ytrain, -np.diag(np.ones(n)), np.zeros([n, 1])), axis=1)
            A2 = np.concatenate((0.0 * xtrain, 0.0 * ytrain, -np.diag(np.ones(n)), np.zeros([n, 1])), axis=1)
            A3 = np.concatenate((np.zeros([2, d]), np.array([[1], [-1]]), np.zeros([2, n]), np.array([[-1], [-1]])),
                                axis=1)
            A = np.concatenate((A1, A2, A3), axis=0)
            b = np.concatenate((-np.ones([n]), np.zeros([n + 2])))
            print('\nshape of c: ' + str(c.shape))
            print('shape of A: ' + str(A.shape))
            print('shape of b: ' + str(b.shape))

            # solve
            x = cp.Variable(n + d + 2)
            prob = cp.Problem(cp.Minimize(c.T @ x), [A @ x <= b])
            prob.solve('SCS')

            # get weights
            self.W = np.array(x.value[0:d])
            self.w = np.array(x.value[d + 1])
        else:
            n, d = xtrain.shape
            k = self.K

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
            self.W = np.array(w.value)
            self.w = 0

    def f(self, input):
        # THIS IS WHERE YOU SHOULD WRITE YOUR CLASSIFICATION FUNCTION
        #
        # The inputs of this function are:
        #
        # input: the input to the function f(*), equal to g(y) = W^T y + w
        #
        # The outputs of this function are:
        #
        # s: this should be a scalar equal to the class estimated from
        # the corresponding input data point, equal to f(W^T y + w)
        # You should also check if the classifier is trained i.e. self.W and
        # self.w are nonempty

        try:
            y_pred = input.dot(self.W + self.w)
        except:
            raise Exception('not trained')
        if self.K == 2:
            s = (y_pred > 0) * 2 - 1
        else:
            s = np.argmax(y_pred, axis=1)
        res = [self.tag2label[x] for x in s]
        return np.array(res)

    def classify(self, test_data):
        # THIS FUNCTION OUTPUTS ESTIMATED CLASSES FOR A DATA MATRIX
        # 
        # The inputs of this function are:
        # self: a reference to the classifier object.
        # test_data: a matrix of dimesions N_test x M, where N_test
        # is the number of inputs used for training. Each row is an
        # input vector.
        #
        #
        # The outputs of this function are:
        #
        # test_results: this should be a vector of length N_test,
        # containing the estimations of the classes of all the N_test
        # inputs.
        result = self.f(test_data)
        return result

    def TestCorrupted(self, p, test_data):
        # THIS FUNCTION OUTPUTS ESTIMATED CLASSES FOR A DATA MATRIX
        #
        #
        # The inputs of this function are:
        #
        # self: a reference to the classifier object.
        # test_data: a matrix of dimesions N_test x M, where N_test
        # is the number of inputs used for training. Each row is an
        # input vector.
        #
        # p:erasure probability
        #
        #
        # The outputs of this function are:
        #
        # test_results: this should be a vector of length N_test,
        # containing the estimations of the classes of all the N_test
        # inputs.drop_data(data, label, p, k, memorycontrol)
        # drop
        mask = np.random.random(test_data.shape)
        data_drop = test_data * (mask < (1 - p))
        return data_drop


from tensorflow.keras.datasets import mnist
a = MyClassifier(4, 1)
# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
xtrain = X_train.reshape(X_train.shape[0], -1)
ytrain =Y_train
xtest = X_test.reshape(X_test.shape[0], -1)
ytest = Y_test

# test 2 labels
e = [1, 7]
idx = [l in e for l in ytrain]
xtrain = xtrain[idx, ]
ytrain = ytrain[idx, ]
idx = [l in e for l in ytest]
xtest = xtest[idx, ]
ytest = ytest[idx, ]

# train
a = MyClassifier(2, 784)
a.train(0.6, xtrain, ytrain)

# accuracy on training sey
t = a.TestCorrupted(0.6, xtrain)
res = a.classify(t)
print('in training set')
print(np.sum(res == ytrain)/ytrain.shape[0])
#show_img(t, ytrain, 6, 6, type='F', p_label=res)

# on testing set
t = a.TestCorrupted(0.6, xtest)
res = a.classify(t)
print('in testing set')
print(np.sum(res == ytest)/ytest.shape[0])
show_img(t, ytest, 6, 6, type='F', p_label=res)



#test multi labels
xtrain = X_train.reshape(X_train.shape[0], -1)
ytrain =Y_train
xtest = X_test.reshape(X_test.shape[0], -1)
ytest = Y_test

# test 2 labels
e = [1, 6, 7, 9]
idx = [l in e for l in ytrain]
xtrain = xtrain[idx, ]
ytrain = ytrain[idx, ]
idx = [l in e for l in ytest]
xtest = xtest[idx, ]
ytest = ytest[idx, ]

# train
a = MyClassifier(4, 784)
a.train(0.6, xtrain, ytrain)

# accuracy on training sey
t = a.TestCorrupted(0.6, xtrain)
res = a.classify(t)
print('in training set')
print(np.sum(res == ytrain)/ytrain.shape[0])
#show_img(t, ytrain, 6, 6, type='F', p_label=res)

# on testing set
t = a.TestCorrupted(0.6, xtest)
res = a.classify(t)
print('in testing set')
print(np.sum(res == ytest)/ytest.shape[0])
show_img(t, ytest, 6, 6, type='F', p_label=res)