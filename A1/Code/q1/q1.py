import csv
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

EPSILON = 0.0000000001

def normalise(X):
    m, n = X.shape
    mu = np.sum(X, axis = 0)/m
    std = np.sqrt(np.sum((X - mu) * (X - mu), axis = 0)/m)
    return (X - mu)/std

def load_data():
    with open('linearX.csv') as csvfile:
        x = list(csv.reader(csvfile))
    with open('linearY.csv') as csvfile:
        y = list(csv.reader(csvfile))
    x = np.array(x).astype(np.float)
    y = np.array(y).astype(np.float)
    x = normalise(x)
    m, n = x.shape
    X = np.ones((m, n+1))
    X[:,1:] = x
    return X, y

def compute_grad(X, y, theta):
    m, n = X.shape
    return np.expand_dims(np.true_divide(np.sum(X * (np.dot(X, theta) - y), axis = 0), m), axis = 0).T

def compute_loss(X, y, theta):
    m, n = X.shape
    return np.dot((np.dot(X, theta) - y).T, np.dot(X, theta) - y)[0][0] / (2*m)

def linear_reg(X, y, eta = 0.025):
    m, n = X.shape
    theta = np.zeros((n, 1))
    old_loss = 1e9
    iter = 0

    while True:
        iter += 1
        loss = compute_loss(X, y, theta)
        grad = compute_grad(X, y, theta)
        theta -= eta * grad
        if abs(loss - old_loss) < EPSILON:
            break
        old_loss = loss
    return theta, old_loss

def linear_reg_with_loss(X, y, eta = 0.025, delay = 0.2):
    m, n = X.shape
    theta = np.zeros((n, 1))
    old_loss = 1e9
    iter = 0

    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.set_xlabel('Intercept')
    ax.set_ylabel('Slope')
    ax.set_zlabel('Loss')

    xx = np.linspace(-2, 2, 100)
    yy = np.linspace(-2, 2, 100)
    XX, YY = np.meshgrid(xx, yy)
    Z = np.zeros(XX.shape)
    for i in range(len(XX[0])):
        for j in range(len(YY)):
            theta = np.array([[YY[j][0], XX[0][i]]]).T
            Z[i][j] = compute_loss(X, y, theta)
    ax.plot_surface(XX, YY, Z, color = 'lightblue')
    x_values = []
    y_values = []
    z_values = []

    while True:
        iter += 1
        loss = compute_loss(X, y, theta)
        grad = compute_grad(X, y, theta)
        theta -= eta * grad
        x_values.append(theta[0][0])
        y_values.append(theta[1][0])
        z_values.append(loss)
        if iter%10 == 0:
            ax.plot3D(x_values, y_values, z_values, 'red')
            ax.set_title('Iteration: ' + str(iter))
            plt.pause(delay)
        if abs(loss - old_loss) < EPSILON:
            break
        old_loss = loss
    plt.show()
    return theta, old_loss


def linear_reg_with_contour(X, y, eta = 0.025, delay = 0.2):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('Intercept')
    ax.set_ylabel('Slope')

    xx = np.linspace(-2, 2, 100)
    yy = np.linspace(-2, 2, 100)
    XX, YY = np.meshgrid(xx, yy)
    Z = np.zeros(XX.shape)
    for i in range(len(XX[0])):
        for j in range(len(YY)):
            theta = np.array([[YY[j][0], XX[0][i]]]).T
            Z[i][j] = compute_loss(X, y, theta)
    levels = [0.01 * i * i for i in range(1, 101)]
    cp = ax.contour(XX, YY, Z, levels)
    fig.colorbar(cp)

    m, n = X.shape
    theta = np.zeros((n, 1))
    old_loss = 1e9
    iter = 0

    x_values = [0]
    y_values = [0]

    while True:
        iter += 1
        loss = compute_loss(X, y, theta)
        grad = compute_grad(X, y, theta)
        theta -= eta * grad
        x_values.append(theta[0][0])
        y_values.append(theta[1][0])
        if iter%10 == 0:
            ax.plot(x_values, y_values, 'red')
            ax.set_title('Iteration: ' + str(iter))
            plt.pause(delay)
        if abs(loss - old_loss) < EPSILON:
            break
        old_loss = loss
    plt.show()
    return theta, old_loss
    
def plot_hypothesis(X, y, theta):
    fig = plt.figure()
    plt.scatter(X[:,1], np.squeeze(y), color = 'red')
    plt.plot(X[:,1], np.squeeze(np.dot(X, theta)), color = 'green')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

if __name__ == '__main__':
    if sys.argv[1] == 'a':
        X, y = load_data()
        theta, loss = linear_reg(X, y)
        print('Theta: ', theta)
        print('Loss: ', loss)
    elif sys.argv[1] == 'b':
        X, y = load_data()
        theta, loss = linear_reg(X, y)
        plot_hypothesis(X, y, theta)
    elif sys.argv[1] == 'c':
        X, y = load_data()
        theta, loss = linear_reg_with_loss(X, y)
    elif sys.argv[1] == 'd':
        X, y = load_data()
        theta, loss = linear_reg_with_contour(X, y)