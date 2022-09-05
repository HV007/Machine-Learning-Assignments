import numpy as np
from cvxopt import matrix, solvers
import sys
import math
import pandas as pd
from time import time
from libsvm.svmutil import *
from scipy import spatial

DIGIT1 = 6
DIGIT2 = 7

def load_data(path):
    data = pd.read_csv(path).to_numpy()
    X = data[:,:-1]
    y = data[:,-1]
    X = X / 255
    y = np.expand_dims(y, axis = 1)
    return X, y

def generate_subset(X, y):
    m, n = X. shape
    X_sub = X[np.any(np.column_stack((y == DIGIT1, y == DIGIT2)), axis = 1), :]
    y_sub = (y[np.any(np.column_stack((y == DIGIT1, y == DIGIT2)), axis = 1)] - (DIGIT1 + DIGIT2)/2) * (2/(DIGIT2 - DIGIT1))
    return X_sub, y_sub

def generate_all_subsets(X, y):
    X_subs = [[[] for j in range(10)] for i in range(10)]
    y_subs = [[[] for j in range(10)] for i in range(10)]
    for i in range(len(y)):
        dig = y[i][0]
        for k in range(10):
            X_subs[dig][k].append(X[i])
            y_subs[dig][k].append(-1)
            X_subs[k][dig].append(X[i])
            y_subs[k][dig].append(1)
    X_subs = [[np.array(X_subs[i][j]) for j in range(10)] for i in range(10)]
    y_subs = [[np.array(y_subs[i][j]) for j in range(10)] for i in range(10)]

    return X_subs, y_subs

def get_alphas(X, y, C = 1):
    m, n = X.shape
    P = matrix(np.dot(y * X , (y * X).T), tc = 'd')
    q = matrix(-np.ones((m, 1)), tc = 'd')
    G = matrix(np.vstack((-np.eye(m), np.eye(m))), tc = 'd')
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)), tc = 'd')
    A = matrix(y.T, tc = 'd')
    b = matrix(np.zeros(1), tc = 'd')

    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    S = np.nonzero(alphas > 1e-4)[0]

    return alphas, S

def get_paramteres(X, y, alphas, S):
    m, n = X.shape
    w = np.dot((y*alphas).T, X).T

    b = np.mean(y[S] - np.dot(X[S], w))

    return w, b

def get_alphas_with_kernel(X, y, C = 1, gamma = 0.05):
    m, n = X.shape

    K = np.exp(-1*gamma*spatial.distance.squareform(spatial.distance.pdist(X, 'sqeuclidean')))
    
    P = matrix(np.outer(y, y) * K, tc = 'd')
    q = matrix(-np.ones((m, 1)), tc = 'd')
    G = matrix(np.vstack((-np.eye(m), np.eye(m))), tc = 'd')
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * C)), tc = 'd')
    A = matrix(y.T, tc = 'd')
    b = matrix(np.zeros(1), tc = 'd')

    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    S = np.nonzero(alphas > 1e-4)[0]

    for i in S:
        pred = 0
        for j in S:
            pred += alphas[j] * y[j] * np.exp(-np.dot(X[i] - X[j], (X[i] - X[j]).T) * gamma)
        b = y[i] - pred
        break
    b = b[0]

    return alphas, S, b

def get_accuracy(X, y, w, b):
    m, n = X.shape
    y_pred = np.where((np.dot(X, w) + b) > 0, 1, -1)
    return np.count_nonzero(y_pred == y) / len(y)

def get_accuracy_with_kernel(X, y, X_test, y_test, alphas, b, S, gamma = 0.05):
    m, n = X.shape

    y_test_pred = np.zeros((len(y_test), )) + b
    for j in S:
        y_test_pred += alphas[j] * y[j] * np.exp(-np.sum((X_test - np.expand_dims(X[j], axis = 1).T)**2, axis = 1) * gamma)
    y_test_pred = np.expand_dims(np.where(y_test_pred > 0, 1, -1), axis = 1)

    return np.count_nonzero(y_test_pred == y_test) / len(y_test), y_test_pred

def get_confusion_matrix(y, y_pred):
    C = np.zeros((10, 10))
    m = len(y)
    for i in range(10):
        for j in range(10):
            for k in range(m):
                if y_pred[k] == j and y[k] == i:
                    C[i][j] += 1
    return C

if __name__ == '__main__':
    TRAIN_PATH = sys.argv[1]
    TEST_PATH = sys.argv[2]

    if sys.argv[3] == '0' and sys.argv[4] == 'a':
        X, y = load_data(TRAIN_PATH)
        X_sub, y_sub = generate_subset(X, y)
        alphas, S = get_alphas(X_sub, y_sub)
        w, b = get_paramteres(X_sub, y_sub, alphas, S)
        train_acc = get_accuracy(X_sub, y_sub, w, b)
        print('Training Accuracy: ', train_acc)
        X_test, y_test = load_data(TEST_PATH)
        X_sub_test, y_sub_test = generate_subset(X_test, y_test)
        test_acc = get_accuracy(X_sub_test, y_sub_test, w, b)
        print('Test Accuracy: ', test_acc)
    elif sys.argv[3] == '0' and sys.argv[4] == 'b':
        X, y = load_data(TRAIN_PATH)
        X_sub, y_sub = generate_subset(X, y)
        alphas, S, b = get_alphas_with_kernel(X_sub, y_sub)
        train_acc, y_pred = get_accuracy_with_kernel(X_sub, y_sub, X_sub, y_sub, alphas, b, S)
        X_test, y_test = load_data(TEST_PATH)
        X_sub_test, y_sub_test = generate_subset(X_test, y_test)
        test_acc, y_test_pred = get_accuracy_with_kernel(X_sub, y_sub, X_sub_test, y_sub_test, alphas, b, S)
        print('Training Accuracy: ', train_acc)
        print('Test Accuracy: ', test_acc)
    elif sys.argv[3] == '0' and sys.argv[4] == 'c':
        X, y = load_data(TRAIN_PATH)
        X_test, y_test = load_data(TEST_PATH)
        X_sub, y_sub = generate_subset(X, y)
        X_sub_test, y_sub_test = generate_subset(X_test, y_test)

        print('LIBSVM Linear Kernel')
        time1 = time()
        prob = svm_problem(y_sub.reshape(len(y_sub)), X_sub)
        train = svm_train(prob, svm_parameter('-s 0 -t 0 -c 1 -g 0.05'))
        svm_predict(y_sub_test.reshape(len(y_sub_test)), X_sub_test, train)
        time2 = time()
        print('Time Taken: ', time2 - time1)

        print('LIBSVM RBF Kernel')
        time1 = time()
        prob = svm_problem(y_sub.reshape(len(y_sub)), X_sub)
        train = svm_train(prob, svm_parameter('-s 0 -t 2 -c 1 -g 0.05'))
        svm_predict(y_sub_test.reshape(len(y_sub_test)), X_sub_test, train)
        time2 = time()
        print('Time Taken: ', time2 - time1)

        time1 = time()
        alphas, S = get_alphas(X_sub, y_sub)
        nSV = np.count_nonzero(alphas > 1e-4)
        w, b = get_paramteres(X_sub, y_sub, alphas, S)
        print('CVXOPT Linear Kernel nSV: ', nSV)
        test_acc = get_accuracy(X_sub_test, y_sub_test, w, b)
        print('CVXOPT Linear Kernel Accuracy: ', test_acc)
        time2 = time()
        print('Time Taken: ', time2 - time1)

        time1 = time()
        alphas, S, b = get_alphas_with_kernel(X_sub, y_sub)
        nSV = len(S)
        print('CVXOPT RBF Kernel nSV: ', nSV)
        test_acc, y_test_pred = get_accuracy_with_kernel(X_sub, y_sub, X_sub_test, y_sub_test, alphas, b, S)
        print('CVXOPT RBF Kernel Accuracy: ', test_acc)
        time2 = time()
        print('Time Taken: ', time2 - time1)
    elif sys.argv[3] == '1' and  sys.argv[4] == 'a':
        time1 = time()
        X, y = load_data(TRAIN_PATH)
        X_subs, y_subs = generate_all_subsets(X, y)
        X_test, y_test = load_data(TEST_PATH)
        y_pred = np.zeros((len(y_test), 10))
        for i in range(10):
            for j in range(i+1, 10):
                print(i, j)
                DIGIT1, DIGIT2 = i, j
                X_sub, y_sub = X_subs[i][j], np.expand_dims(y_subs[i][j], axis = 1)
                alphas, S, b = get_alphas_with_kernel(X_sub, y_sub)
                test_acc, y_test_pred = get_accuracy_with_kernel(X_sub, y_sub, X_test, y_test, alphas, b, S)
                for idx, val in enumerate(y_test_pred):
                    if val == 1:
                        y_pred[idx][DIGIT2] += 1
                    else :
                        y_pred[idx][DIGIT1] += 1
        y_pred = np.argmax(y_pred, axis = 1)
        y_pred = np.expand_dims(y_pred, axis = 1)
        acc = np.count_nonzero(y_test == y_pred)/len(y_test)
        print('Accuracy: ', acc)
        time2 = time()
        print('Time Taken: ', time2 - time1)
    elif sys.argv[3] == '1' and sys.argv[4] == 'b':
        time1 = time()
        X, y = load_data(TRAIN_PATH)
        X_test, y_test = load_data(TEST_PATH)
        prob = svm_problem(y.reshape(len(y)), X)
        train = svm_train(prob, svm_parameter('-s 0 -t 2 -c 1 -g 0.05'))
        svm_predict(y_test.reshape(len(y_test)), X_test, train)
        time2 = time()
        print('Time Taken: ', time2 - time1)
    elif sys.argv[3] == '1' and sys.argv[4] == 'c':
        time1 = time()
        X, y = load_data(TRAIN_PATH)
        X_subs, y_subs = generate_all_subsets(X, y)
        X_test, y_test = load_data(TEST_PATH)
        y_pred = np.zeros((len(y_test), 10))
        for i in range(10):
            for j in range(i+1, 10):
                print(i, j)
                DIGIT1, DIGIT2 = i, j
                X_sub, y_sub = X_subs[i][j], np.expand_dims(y_subs[i][j], axis = 1)
                alphas, S, b = get_alphas_with_kernel(X_sub, y_sub)
                test_acc, y_test_pred = get_accuracy_with_kernel(X_sub, y_sub, X_test, y_test, alphas, b, S)
                for idx, val in enumerate(y_test_pred):
                    if val == 1:
                        y_pred[idx][DIGIT2] += 1
                    else :
                        y_pred[idx][DIGIT1] += 1
        y_pred = np.argmax(y_pred, axis = 1)
        y_pred = np.expand_dims(y_pred, axis = 1)
        with open('pred_cvxopt.npy', 'wb') as f:
            np.save(f, y_pred)
        acc = np.count_nonzero(y_test == y_pred)/len(y_test)
        print('Accuracy: ', acc)
        C = get_confusion_matrix(y_test, y_pred)
        print('CVXOPT Confusion Matrix: ', C)
        time2 = time()
        print('Time Taken: ', time2 - time1)

        time1 = time()
        X, y = load_data(TRAIN_PATH)
        X_test, y_test = load_data(TEST_PATH)
        prob = svm_problem(y.reshape(len(y)), X)
        train = svm_train(prob, svm_parameter('-s 0 -t 2 -c 1 -g 0.05'))
        p_label, _, _ = svm_predict(y_test.reshape(len(y_test)), X_test, train)
        with open('pred_libsvm.npy', 'wb') as f:
            np.save(f, p_label)
        C = get_confusion_matrix(y_test, p_label)
        print('LIBSVM Confusion Matrix: ', C)
        time2 = time()
        print('Time Taken: ', time2 - time1)
