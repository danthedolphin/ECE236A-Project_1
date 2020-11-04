from MyClassifier import *
from tensorflow.keras.datasets import mnist

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


train_acc = []
test_acc = []
test_p = 0.6

for ite in range(10):
    # train
    print("binary for : " + str(ite + 1))
    a = MyClassifier(2, 784)
    a.train(0.6, xtrain, ytrain)

    # accuracy on training sey
    res = a.TestCorrupted(0.6, xtrain)
    #res = a.classify(t)
    train_acc.append(np.sum(res == ytrain) / ytrain.shape[0])
    # print('in training set')
    # print(np.sum(res == ytrain) / ytrain.shape[0])
    # show_img(t, ytrain, 6, 6, type='F', p_label=res)

    # on testing set
    res = a.TestCorrupted(test_p, xtest)
    #res = a.classify(t)
    test_acc.append(np.sum(res == ytest) / ytest.shape[0])
    # print('in testing set')
    # print(np.sum(res == ytest) / ytest.shape[0])
    # show_img(t, ytest, 6, 6, type='F', p_label=res)

print(train_acc)
print(test_acc)

#test multi labels
xtrain = X_train.reshape(X_train.shape[0], -1)
ytrain =Y_train
xtest = X_test.reshape(X_test.shape[0], -1)
ytest = Y_test

# test 2 labels
e = [1, 9, 6, 7]
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
res = a.TestCorrupted(0.6, xtrain)
#res = a.classify(t)
print('in training set')
print(np.sum(res == ytrain)/ytrain.shape[0])
#show_img(t, ytrain, 6, 6, type='F', p_label=res)

# on testing set
res = a.TestCorrupted(0.8, xtest)
#res = a.classify(t)
print('in testing set')
print(np.sum(res == ytest)/ytest.shape[0])
#show_img(t, ytest, 6, 6, type='F', p_label=res)
