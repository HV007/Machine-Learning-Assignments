import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

EPSILON = 0.0000000001

def normalise(X):
    m, n = X.shape
    mu = np.sum(X, axis = 0)/m
    std = np.sqrt(np.sum((X - mu) * (X - mu), axis = 0)/m)
    return (X - mu)/std

def load_data():
    with open('logisticX.csv') as csvfile:
        x = list(csv.reader(csvfile))
    with open('logisticY.csv') as csvfile:
        y = list(csv.reader(csvfile))
    x = np.array(x).astype(np.float)
    y = np.array(y).astype(np.float)
    x = normalise(x)
    m, n = x.shape
    X = np.ones((m, n+1))
    X[:,1:] = x
    return X, y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_hessian(X, y, theta):
    m, n = X.shape
    H = np.zeros((n, n))
    for i in range(m):
        X_i = np.expand_dims(X[i,:], axis = 1).T
        H += np.dot(X_i.T, X_i)*(sigmoid(np.dot(X_i, theta))*(1 - sigmoid(np.dot(X_i, theta))))
    return H

def compute_grad(X, y, theta):
    m, n = X.shape
    return np.expand_dims(np.sum((sigmoid(np.dot(X, theta)) - y) * X, axis = 0), axis = 1)

def compute_loss(X, y, theta):
    m, n = X.shape
    return -np.sum(y*np.log(sigmoid(np.dot(X, theta))) + (1 - y)*np.log(1 - sigmoid(np.dot(X, theta))), axis = 0)[0]

def logistic_reg(X, y):
    m, n = X.shape
    theta = np.zeros((n, 1))
    old_loss = 1e9
    iteration = 0
    while True:
        iteration += 1
        loss = compute_loss(X, y, theta)
        grad = compute_grad(X, y, theta)
        hess = compute_hessian(X, y, theta)
        theta -= np.dot(np.linalg.inv(hess), grad)
        if abs(loss - old_loss) < EPSILON:
            break
        old_loss = loss
    return theta, old_loss, iteration

def plot_hypothesis(X, y, theta):
    fig = plt.figure()
    zeros = np.where(y == 0)
    ones = np.where(y == 1)
    plt.scatter(X[zeros, 1], X[zeros, 2], marker = 'x')
    plt.scatter(X[ones, 1], X[ones, 2], marker = 'o')
    plt.plot(X[:,1], -(theta[1]*X[:,1] + theta[0])/theta[2], color = 'green')
    plt.show()

if __name__ == '__main__':
    if sys.argv[1] == 'a':
        X, y = load_data()
        theta, loss, iteration = logistic_reg(X, y)
        print('Theta: ', theta)
        print('Loss: ', loss)
        print('Iterations: ', iteration)
    else :
        X, y = load_data()
        theta, loss, iteration = logistic_reg(X, y)
        plot_hypothesis(X, y, theta)
    