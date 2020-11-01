# -*- coding: utf-8 -*-
"""
ECE 236A Project 1, MyClassifier.py template. Note that you should change the
name of this file to MyClassifier_{groupno}.py
"""
import pandas as pd
import numpy as np
from scipy import stats
import cvxpy as cp
from DataProcessor import *


class MyClassifier:
    def __init__(self,K,M):
        self.K = K   # Number of classes
        self.M = M   # Number of features
        self.W = np.zeros([int(K*(K-1)/2),self.M])
        self.w = np.zeros([int(K*(K-1)/2)])  # Bias
        self.mapping = []  # used for K>2, to map the real labesl to training labels
        self.label2tag = []  # used for K=2, to map the real labesl to training labels
        self.tag2label = []  # used for K=2, to map the training labesl to real labels
        # increase this value will lead to a higher memory used， 1.4 will use 21.2Gb for Minist
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
        n, d = xtrain.shape
        c = np.ones(n)

        e = cp.Variable(n)
        lambda_reg = 1
        w = cp.Variable(d)

        b = cp.Variable(1)
        w2 = cp.pnorm(w,p=2)**2
        prob = cp.Problem(cp.Minimize(c.T@e+lambda_reg*w2), [e>=(1-cp.multiply(np.squeeze(ytrain),(xtrain@w+b))),e>=0])
        prob.solve(solver='SCS')

        w = w.value
        b = b.value
        return w, b

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
            print('shape of modified lbaels      : ' + str(ytrain.shape))
            W, w = self.__trainTwoClass(xtrain, ytrain)
            print('\nshape of modified training set: ' + str(self.W.shape))
            print('shape of modified lbaels      : ' + str(self.w.shape))
            self.W = W
            self.w = w
        else:
            labels = np.unique(train_label[:-1])
            num_classifier = 0
            for i,x in enumerate(labels):
                for y in labels[i+1:]:
                    self.mapping.append((x, y))
                    xtrain, ytrain, mapping = data_extract(data, label, [x, y])
                    print("training for : " + str(mapping))
                    W, w = self.__trainTwoClass(xtrain, ytrain)
                    self.W[num_classifier, ] = W
                    self.w[num_classifier, ] = w
                    num_classifier+=1
            print('done making {} classifiers'.format(self.K*(self.K-1)/2))
    def f(self, g):
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
            s = (g > 0) * 2 - 1
            res = [self.tag2label[x] for x in s]
        else:
            votes = np.zeros_like(g)
            g = np.where(g>0, True, False) #True for positive values, False for negative values
            for i,x in enumerate(self.mapping):
                votes[:,i] = [x[1] if z else x[0] for z in g[:,i]] #map the True and False to the predicted class
            votes = votes.astype(int)
            res = stats.mode(votes,axis=1).mode #take the mode to get highest amount of votes along an axis
        return res.squeeze()

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
        try:
            if self.K == 2:
                scores = test_data.dot(self.W) + self.w
            else:
                scores = test_data @ self.W.T + self.w
        except:
            raise Exception('not trained')
        result = self.f(scores)
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


from tensorflow.keras.datasets import mnist

a = MyClassifier(4, 1)
# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
xtrain = X_train.reshape(X_train.shape[0], -1)
ytrain =Y_train
xtest = X_test.reshape(X_test.shape[0], -1)
ytest = Y_test
'''
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
'''
# test 2 labels
e = [1,6,9]
idx = [l in e for l in ytrain]
xtrain = xtrain[idx, ]
ytrain = ytrain[idx, ]
idx = [l in e for l in ytest]
xtest = xtest[idx, ]
ytest = ytest[idx, ]

# train
a = MyClassifier(len(e), 784)
a.train(0.6, xtrain, ytrain)

# accuracy on training sey
t = a.TestCorrupted(0.6, xtrain)

print('in training set')
print(np.sum(t == ytrain)/ytrain.shape[0])
#show_img(t, ytrain, 6, 6, type='F', p_label=res)

# on testing set
t = a.TestCorrupted(0.6, xtest)

print('in testing set')
print(np.sum(t == ytest)/ytest.shape[0])
#show_img(t, ytest, 6, 6, type='F', p_label=t)