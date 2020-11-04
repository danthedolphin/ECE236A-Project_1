# -*- coding: utf-8 -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""
import pandas as pd
import numpy as np
import cvxpy as cp
from scipy import stats
from DataProcessor import *


class MyClassifier:
    def __init__(self,K,M):
        self.K = K   # Number of classes
        self.M = M   # Number of features
        self.W = np.zeros([int(K*(K-1)/2), M])  # Weight
        self.w = np.zeros([int(K*(K-1)/2), 1])  # Bias
        self.mapping = []  # used for K>2, to map the real labesl to training labels
        self.label2tag = []  # used for K=2, to map the real labesl to training labels
        self.tag2label = []  # used for K=2, to map the training labesl to real labels
        # increase this value will lead to a higher memory used， 1.5 will use 15Gb for Minist
        # but a higher memorycontrol value will lead to higher accuracy
        self.memorycontrol = 1.4

    def __preprocess(self, data, label, p):
        # change label
        label2tag, tag2label, label = get_label(label, self.K)
        self.label2tag = label2tag
        self.tag2label = tag2label
        # get training set
        p = [max(0.01, p - 0.2), p, min(1, p + 0.2)]
        train, label = drop_data(data, label, p, self.memorycontrol)
        return train, label

    def __trainTwoClass(self, xtrain, ytrain):
        '''
        n, d = xtrain.shape
        c = np.concatenate((np.zeros([d + 1]), np.ones([n]), np.array([0.1])))
        A1 = np.concatenate((-ytrain * xtrain, -ytrain, -np.diag(np.ones(n)), np.zeros([n, 1])), axis=1)
        A2 = np.concatenate((0.0 * xtrain, 0.0 * ytrain, -np.diag(np.ones(n)), np.zeros([n, 1])), axis=1)
        A3 = np.concatenate((np.zeros([2, d]), np.array([[1], [-1]]), np.zeros([2, n]), np.array([[-1], [-1]])),
                            axis=1)
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
        W = np.array(x.value[0:d])
        w = np.array(x.value[d + 1])
        '''

        # formulate the problem
        n, d = xtrain.shape
        c = np.ones(n)

        e = cp.Variable(n)
        lambda_reg = 0.01
        w = cp.Variable(d)

        b = cp.Variable(1)
        w2 = cp.pnorm(w, p=2) ** 2
        prob = cp.Problem(cp.Minimize(c.T @ e + lambda_reg * w2),
                          [e >= (1 - cp.multiply(np.squeeze(ytrain), (xtrain @ w + b))), e >= 0])
        prob.solve(solver='SCS')
        # get weights
        W = w.value
        w = b.value

        return W, w

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

        # change to array
        data = np.array(train_data)
        label = np.array(train_label)
        # change to 2-d matrix
        data = data.reshape(data.shape[0], self.M)
        label = label.reshape(label.shape[0], -1)
        # change type
        data = data.astype('uint8')
        label = label.astype('int8')

        # preprocessing
        if self.K == 2:
            xtrain, ytrain = self.__preprocess(data, label, p)
            print('\nshape of modified training set: ' + str(xtrain.shape))
            print('shape of modified labels      : ' + str(ytrain.shape))
            W, w = self.__trainTwoClass(xtrain, ytrain)
            self.W = W
            self.w = w
            print('shape of W: ' + str(self.W.shape))
            print('shape of w: ' + str(self.w.shape))

        else:
            labels = np.unique(train_label)
            num_calssifier = 0
            p = [max(0.01, p - 0.2), p, min(1, p + 0.2)]
            for x in range(len(labels)):
                for y in range(x, len(labels)):
                    if x != y:
                        self.mapping.append((labels[x], labels[y]))
                        xtrain, ytrain, map = data_extract(data, label, [labels[x], labels[y]])
                        xtrain, ytrain = drop_data(xtrain, ytrain, p, self.memorycontrol)
                        print("\ntraining for : " + str(map))
                        W, w = self.__trainTwoClass(xtrain, ytrain)
                        self.W[num_calssifier, ] = W
                        self.w[num_calssifier, ] = w
                        num_calssifier = num_calssifier + 1
            print('shape of W: ' + str(self.W.shape))
            print('shape of w: ' + str(self.w.shape))

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
        if self.K == 2:
            s = (input > 0) * 2 - 1
            # = [self.tag2label[x] for x in s]
            res = self.tag2label[s]
        else:
            g = input
            votes = np.zeros_like(g)
            #g = np.where(g > 0, True, False) #True for positive values, False for negative values
            for i, x in enumerate(self.mapping):
                votes[i]= x[1] if g[i]>0 else x[0] #map the True and False to the predicted class
            res = int(stats.mode(votes).mode[0]) #take the mode to get highest amount of votes along an axis
        return (res)

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
        if (self.W==np.zeros_like(self.W)).all() or (self.w==np.zeros_like(self.w)).all():
            raise Exception('Not Trained')
        if self.K == 2:
            scores = test_data.dot(self.W) + self.w
        else:
            scores = test_data.dot(self.W.T) + self.w.T
        result = [self.f(score) for score in scores]
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
        est_classes = self.classify(data_drop)
        return est_classes


