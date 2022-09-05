import csv
import numpy as np
import matplotlib.pyplot as plt
import sys

def normalise(X):
    m, n = X.shape
    mu = np.sum(X, axis = 0)/m
    std = np.sqrt(np.sum((X - mu) * (X - mu), axis = 0)/m)
    return (X - mu)/std

def load_data():
    with open('q4x.dat') as csvfile:
        x = list(csv.reader(csvfile, delimiter = ' '))
    with open('q4y.dat') as csvfile:
        y = list(csv.reader(csvfile, delimiter = ' '))
    x = np.array(x)[:, [0,2]].astype(np.float)
    y = np.array(y)
    X = normalise(x)
    m, n = X.shape
    y = (y == 'Canada').astype(np.float)
    return X, y

def get_parameters1(X, y):
    m, n = X.shape
    mu0 = np.sum((1 - y)*X, axis = 0) / np.sum(1 - y, axis = 0)
    mu1 = np.sum(y*X, axis = 0) / np.sum(y, axis = 0)
    sigma = np.zeros((n, n))
    for i in range(m):
        temp = np.expand_dims(X[i,:] - (mu0 if y[i] == 0 else mu1), axis = 1)
        sigma += np.dot(temp, temp.T)
    sigma /= m
    phi = np.sum(y, axis = 0)/m
    return mu0, mu1, sigma, phi

def get_parameters2(X, y):
    m, n = X.shape
    mu0 = np.sum((1 - y)*X, axis = 0) / np.sum(1 - y, axis = 0)
    mu1 = np.sum(y*X, axis = 0) / np.sum(y, axis = 0)
    sigma0 = np.zeros((n, n))
    sigma1 = np.zeros((n, n))
    count0, count1 = 0, 0
    for i in range(m):
        temp = np.expand_dims(X[i,:] - (mu0 if y[i] == 0 else mu1), axis = 1)
        if y[i] == 0:
            sigma0 += np.dot(temp, temp.T)
            count0 += 1
        else :
            sigma1 += np.dot(temp, temp.T)
            count1 += 1
    sigma0 /= count0
    sigma1 /= count1
    phi = np.sum(y, axis = 0)/m
    return mu0, mu1, sigma0, sigma1, phi

def plot_points(X, y):
    zeros = np.where(y == 0)
    ones = np.where(y == 1)
    plt.scatter(X[zeros, 0], X[zeros, 1], marker = 'x')
    plt.scatter(X[ones, 0], X[ones, 1], marker = 'o')

def plot_hypothesis(X, y, mu0, mu1, sigma0, sigma1, phi):
    xx, yy = np.meshgrid(np.arange(-2, 2, 0.1), np.arange(-2, 2, 0.1))
    A = (np.linalg.inv(sigma1) - np.linalg.inv(sigma0))/2
    B = -(np.dot(mu1.T, np.linalg.inv(sigma1)) - np.dot(mu0.T, np.linalg.inv(sigma0)))
    C = (np.dot(np.dot(mu1.T, np.linalg.inv(sigma1)), mu1) - np.dot(np.dot(mu0.T, np.linalg.inv(sigma0)), mu0))/2
    D = np.log(((1-phi)/phi)*(np.linalg.det(sigma1)/np.linalg.det(sigma0))**0.5)
    Z = []
    for x in np.c_[xx.ravel(), yy.ravel()]:
        x = np.expand_dims(x, axis = 1)
        Z.append(np.dot(np.dot(x.T, A), x) + np.dot(B, x) + C + D)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired, levels = [0])


if __name__ == '__main__':
    if sys.argv[1] == 'a':
        X, y = load_data()
        mu0, mu1, sigma, phi = get_parameters1(X, y)
        print('mu0: ', mu0)
        print('mu1: ', mu1)
        print('sigma: ', sigma)
    elif sys.argv[1] == 'b':
        X, y = load_data()
        fig = plt.figure()
        plot_points(X, y)
        plt.show()
    elif sys.argv[1] == 'c':
        X, y = load_data()
        fig = plt.figure()
        plot_points(X, y)
        mu0, mu1, sigma, phi = get_parameters1(X, y)
        plot_hypothesis(X, y, mu0, mu1, sigma, sigma, phi)
        plt.show()
    elif sys.argv[1] == 'd':
        X, y = load_data()
        mu0, mu1, sigma0, sigma1, phi = get_parameters2(X, y)
        print('mu0: ', mu0)
        print('mu1: ', mu1)
        print('sigma0: ', sigma0)
        print('sigma1: ', sigma1)
    elif sys.argv[1] == 'e':
        X, y = load_data()
        fig = plt.figure()
        plot_points(X, y)
        mu0, mu1, sigma, phi = get_parameters1(X, y)
        plot_hypothesis(X, y, mu0, mu1, sigma, sigma, phi)
        mu0, mu1, sigma0, sigma1, phi = get_parameters2(X, y)
        plot_hypothesis(X, y, mu0, mu1, sigma0, sigma1, phi)
        plt.show()