from tensorflow.keras.datasets import mnist
from DataProcessor import *
from TestMethod import *

# load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# dataProcessor
xtrain = X_train.reshape(X_train.shape[0], -1)
ytrain =Y_train.reshape(Y_train.shape[0], -1)
xtest = X_test.reshape(X_test.shape[0], -1)
ytest = Y_test.reshape(Y_test.shape[0], -1)
# data type
xtrain = xtrain.astype('uint8')
ytrain = ytrain.astype('int8')
xtest = xtest.astype('uint8')
ytest = ytest.astype('int8')
# get data
xtrain, ytrain, _ = data_extract(xtrain, ytrain, [1, 7])
xtest, ytest, _ = data_extract(xtest, ytest, [1, 7])
xtest, ytest = drop_data(xtest, ytest, 0.6, 1, 1.0)
xtrain_adv, ytrain_adv = drop_data(xtrain, ytrain, 0, 1.0, 0.1)
xtrain_mod, ytrain_mod = drop_data(xtrain, ytrain, [0.4, 0.6, 0.8], 1, 0.7)

# test
VanillaSVM(xtrain, ytrain, xtest, ytest)
VanillaSVM_WithReg(xtrain, ytrain, xtest, ytest)
AdversarySVM(xtrain_adv, ytrain_adv, xtest, ytest)
ModifiedDataSVM(xtrain_mod, ytrain_mod, xtest, ytest)


#multi-class
# dataProcessor
xtrain = X_train.reshape(X_train.shape[0], -1)
ytrain =Y_train.reshape(Y_train.shape[0], -1)
xtest = X_test.reshape(X_test.shape[0], -1)
ytest = Y_test.reshape(Y_test.shape[0], -1)
# data type
xtrain = xtrain.astype('uint8')
ytrain = ytrain.astype('int8')
xtest = xtest.astype('uint8')
ytest = ytest.astype('int8')
# get data
xtrain, ytrain, _ = data_extract(xtrain, ytrain, [1, 2, 3, 4])
xtest, ytest, _ = data_extract(xtest, ytest, [1, 2, 3, 4])
xtest, ytest = drop_data(xtest, ytest, 0.6, 1, 1)
xtrain_mod, ytrain_mod = drop_data(xtrain, ytrain, [0.4, 0.6, 0.8], 1, 0.3)

# test
Vanilla_multiSVM(xtrain, ytrain, xtest, ytest)
ModifiedData_multiSVM(xtrain_mod, ytrain_mod, xtest, ytest)