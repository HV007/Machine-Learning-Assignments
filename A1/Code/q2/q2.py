import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import time
import sys

SAMPLE_POINTS = 1000000
EPSILON = 0.001

def sample_points():
    theta = np.array([[3, 1, 2]]).T
    X0 = np.ones((SAMPLE_POINTS, 1))
    X1 = np.random.normal(3, 4, size = (SAMPLE_POINTS, 1))
    X2 = np.random.normal(-1, 4, size = (SAMPLE_POINTS, 1))
    X = np.concatenate((X0, X1, X2), axis = 1)
    y = np.dot(X, theta) + np.random.normal(0, math.sqrt(2), size = (SAMPLE_POINTS, 1))
    return X, y

def compute_grad(X, y, theta):
    m, n = X.shape
    return np.expand_dims(np.true_divide(np.sum(X * (np.dot(X, theta) - y), axis = 0), m), axis = 0).T

def compute_loss(X, y, theta):
    m, n = X.shape
    return np.dot((np.dot(X, theta) - y).T, np.dot(X, theta) - y)[0][0] / (2*m)

def sgd(X, y, batch = 100, k = 1000, eta = 0.001, plot = False):
    m, n = X.shape
    theta = np.zeros((n, 1))
    done = False
    old_loss = 0
    curr_loss = 0
    curr_batch = 0
    iteration = 0

    if plot:
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.set_xlabel('theta0')
        ax.set_ylabel('theta1')
        ax.set_zlabel('theta2')
        x_values = []
        y_values = []
        z_values = []
    while not done:
        for i in range(0, m, batch):
            if i + batch > m: continue
            curr_loss += compute_loss(X[i:i+batch, :], y[i:i+batch], theta)
            grad = compute_grad(X[i:i+batch, :], y[i:i+batch], theta)
            theta -= eta * grad
            if plot:
                x_values.append(theta[0][0])
                y_values.append(theta[1][0])
                z_values.append(theta[2][0])
            curr_batch += 1
            iteration += 1
            if plot and iteration%100 == 0:
                ax.plot3D(x_values, y_values, z_values, 'red')
                ax.set_title('Iteration: ' + str(iteration))
                plt.pause(0.2)
            if curr_batch == k:
                curr_loss /= k
                if abs(curr_loss - old_loss) < EPSILON:
                    done = True
                    break
                else :
                    old_loss = curr_loss
                    curr_batch = 0
                    curr_loss = 0
    plt.show()
    return theta, old_loss, iteration

def load_test():
    data = pd.read_csv('q2test.csv').values
    m, n = data.shape
    X = np.concatenate((np.ones((m, 1)), data[:,0:2]), axis = 1)
    y = np.expand_dims(data[:,2], axis = 1)
    return X, y

if __name__ == '__main__':
    if sys.argv[1] == 'd':
        X, y = sample_points()
        theta, loss, iteration = sgd(X, y, batch = 100, k = 1000, plot = True)
    else :
        X, y = sample_points()
        t1 = time.time()
        theta, loss, iteration = sgd(X, y, batch = 100, k = 1000, plot = False)
        t2 = time.time()
        X_test, y_test = load_test()
        test_loss = compute_loss(X_test, y_test, theta)
        actual_loss = compute_loss(X_test, y_test, np.array([[3, 1, 2]]).T)
        print('Theta: ', theta)
        print('Training loss: ', loss)
        print('Test loss: ', loss)
        print('Actual loss: ', actual_loss)
        print('Total iterations: ', iteration)
        print('Total time taken: ', t2-t1)