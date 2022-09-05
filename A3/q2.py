import sys
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

np.random.seed(3)

EPSILON = 0.0001

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return z * (1 - z)

def relu(z):
    return np.maximum(z, 0)

def relu_derivative(z):
    return np.greater(z, 0)

def get_derivative(activation):
    return sigmoid_derivative if activation == sigmoid else relu_derivative

def one_hot_encode_data(X):
    X_new = []
    for i in range(len(X)):
        temp = []
        for j in range(10):
            for k in range(1, 5 if j%2 == 0 else 14):
                if X[i][j] == k:
                    temp.append(1)
                else :
                    temp.append(0)
        X_new.append(temp)
    return np.array(X_new)

def one_hot_encode_labels(y):
    y_new = []
    for i in range(len(y)):
        temp = []
        for k in range(10):
            if y[i] == k:
                temp.append(1)
            else :
                temp.append(0)
        y_new.append(temp)
    return np.array(y_new)

def load_data(path):
    data = np.genfromtxt(path, delimiter = ',')
    X, y = data[:,:-1], data[:,-1]
    X = one_hot_encode_data(X)
    y = one_hot_encode_labels(y)
    return X, y

class NeuralNetwork:
    def __init__(self, layers, activations, features, target, batch, learning_rate = 0.1, adaptive = False):
        self.layers = [features] + layers + [target]
        self.features = features
        self.target = target
        self.batch = batch
        self.theta = [None] + [np.random.uniform(-1, 1, (self.layers[i-1] + 1, self.layers[i])) * np.sqrt(self.layers[i-1]) for i in range(1, len(self.layers))]
        self.max_epochs = 1000
        self.activations = [None] + activations
        self.activations_derivative = [None] + [get_derivative(activation) for activation in activations]
        self.learning_rate = learning_rate
        self.adaptive = adaptive
    
    def train(self, X, y):
        weights = [None for i in range(len(self.layers))]
        delta = [None for i in range(len(self.layers))]

        prev_loss = float('inf')
        total_epochs = 0

        for epoch in range(self.max_epochs):
            total_epochs += 1
            num_batches = len(X) // self.batch

            perm = np.random.permutation(len(X))
            X, y = X[perm], y[perm]

            curr_loss = 0
            curr_learning_rate = self.learning_rate / np.sqrt(epoch + 1) if self.adaptive else self.learning_rate
            for i in range(num_batches):
                X_batch = X[i * self.batch : (i+1) * self.batch, :]
                y_batch = y[i * self.batch : (i+1) * self.batch, :]

                weights[0] = np.concatenate((np.ones((self.batch, 1)), X_batch), axis = 1)
                for j in range(1, len(self.layers)):
                    weights[j] = np.concatenate((np.ones((self.batch, 1)), self.activations[j](np.dot(weights[j-1], self.theta[j]))), axis = 1)
                
                output = weights[-1][:,1:]
                
                delta[-1] = (y_batch - output) * self.activations_derivative[-1](output) / self.batch
                for j in range(len(self.layers) - 2, 0, -1):
                    delta[j] = ((np.dot(delta[j+1], self.theta[j+1].T)) * self.activations_derivative[j](weights[j]))[:,1:]
                
                for j in range(1, len(self.layers)):
                    self.theta[j] += curr_learning_rate * (np.dot(weights[j-1].T, delta[j]))
                curr_loss += np.sum((output - y_batch) ** 2) / (2 * self.batch)
            
            if abs(curr_loss - prev_loss) < EPSILON:
                print('Epochs: ', total_epochs)
                return curr_loss
            
            prev_loss = curr_loss
        print('Epochs: ', total_epochs)
        return prev_loss
    
    def predict(self, X_test):
        output = np.concatenate((np.ones((len(X_test), 1)), X_test), axis = 1)
        for j in range(1, len(self.layers)):
            output = np.concatenate((np.ones((len(X_test), 1)), self.activations[j](output @ self.theta[j])), axis = 1)
        return output[:,1:]

def get_accuracy(y, y_pred):
    return np.count_nonzero(np.argmax(y, axis = 1) == np.argmax(y_pred, axis = 1)) / len(y)

def get_confusion_matrix(y, y_pred):
    y_pred = np.argmax(y_pred, axis = 1)
    y = np.argmax(y, axis = 1)
    C = np.zeros((10, 10))
    m = len(y)
    for i in range(10):
        for j in range(10):
            for k in range(m):
                if y_pred[k] == j and y[k] == i:
                    C[i][j] += 1
    return C

def partA(train_path, test_path):
    X, y = load_data(train_path)
    X_test, y_test = load_data(test_path)
    print(X.shape, y.shape)

def partB(train_path, test_path):
    print(NeuralNetwork)

def partC(train_path, test_path):
    np.set_printoptions(suppress=True)
    X, y = load_data(train_path)
    X_test, y_test = load_data(test_path)
    # X_test = np.load('X_test.npy')
    # y_test = np.load('y_test.npy')
    train_accuracies = []
    test_accuracies = []
    times = []
    layers = [5, 10, 15, 20, 25]
    for hidden_layer in layers:
        print('Hidden Layer: ', hidden_layer)
        time1 = time()
        nn = NeuralNetwork([hidden_layer], [sigmoid, sigmoid], 85, 10, 100)
        nn.train(X, y)
        y_pred = nn.predict(X)
        y_pred_test = nn.predict(X_test)
        train_acc = get_accuracy(y, y_pred)
        test_acc = get_accuracy(y_test, y_pred_test)
        time2 = time()
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        times.append(time2 - time1)
        print('Training Accuracy: ', train_acc)
        print('Test Accuracy: ', test_acc)
        print(get_confusion_matrix(y, y_pred))
    fig, axes = plt.subplots(2)
    axes[0].plot(layers, train_accuracies, label = 'Training Accuracy')
    axes[0].plot(layers, test_accuracies, label = 'Test Accuracy')
    axes[0].legend()
    axes[1].plot(layers, times, label = 'Time Taken')
    axes[1].legend()
    plt.show()

def partD(train_path, test_path):
    np.set_printoptions(suppress=True)
    X, y = load_data(train_path)
    X_test, y_test = load_data(test_path)
    # X_test = np.load('X_test.npy')
    # y_test = np.load('y_test.npy')
    train_accuracies = []
    test_accuracies = []
    times = []
    layers = [5, 10, 15, 20, 25]
    for hidden_layer in layers:
        print('Hidden Layer: ', hidden_layer)
        time1 = time()
        nn = NeuralNetwork([hidden_layer], [sigmoid, sigmoid], 85, 10, 100, adaptive = True)
        nn.train(X, y)
        y_pred = nn.predict(X)
        y_pred_test = nn.predict(X_test)
        train_acc = get_accuracy(y, y_pred)
        test_acc = get_accuracy(y_test, y_pred_test)
        time2 = time()
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        times.append(time2 - time1)
        print('Training Accuracy: ', train_acc)
        print('Test Accuracy: ', test_acc)
        print(get_confusion_matrix(y, y_pred))
    fig, axes = plt.subplots(2)
    axes[0].plot(layers, train_accuracies, label = 'Training Accuracy')
    axes[0].plot(layers, test_accuracies, label = 'Test Accuracy')
    axes[0].legend()
    axes[1].plot(layers, times, label = 'Time Taken')
    axes[1].legend()
    plt.show()

def partE(train_path, test_path):
    np.set_printoptions(suppress=True)
    X, y = load_data(train_path)
    X_test, y_test = load_data(test_path)
    # X_test = np.load('X_test.npy')
    # y_test = np.load('y_test.npy')
    nn = NeuralNetwork([100, 100], [sigmoid, sigmoid, sigmoid], 85, 10, 100, adaptive = True)
    nn.train(X, y)
    y_pred = nn.predict(X)
    y_pred_test = nn.predict(X_test)
    train_acc = get_accuracy(y, y_pred)
    test_acc = get_accuracy(y_test, y_pred_test)
    print('Training Accuracy for sigmoid:', train_acc)
    print('Test Accuracy for sigmoid:', test_acc)
    print(get_confusion_matrix(y, y_pred))

    nn = NeuralNetwork([100, 100], [relu, relu, sigmoid], 85, 10, 100, adaptive = True)
    nn.train(X, y)
    y_pred = nn.predict(X)
    y_pred_test = nn.predict(X_test)
    train_acc = get_accuracy(y, y_pred)
    test_acc = get_accuracy(y_test, y_pred_test)
    print('Training Accuracy for relu:', train_acc)
    print('Test Accuracy for relu:', test_acc)
    print(get_confusion_matrix(y, y_pred))

def partF(train_path, test_path):
    X, y = load_data(train_path)
    X_test, y_test = load_data(test_path)
    X_test = np.load('X_test.npy')
    # y_test = np.load('y_test.npy')
    clf = MLPClassifier(random_state = 2, max_iter = 1000, hidden_layer_sizes = (100, 100), activation = 'relu', learning_rate = 'adaptive', learning_rate_init = 0.01, solver = 'sgd', shuffle = True, batch_size = 100)
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    test_acc = clf.score(X_test, y_test)
    print('Training Accuracy for MLP:', train_acc)
    print('Test Accuracy for MLP:', test_acc)

if __name__ == '__main__':
    train_path, test_path = sys.argv[1], sys.argv[2]
    if sys.argv[3] == 'a':
        partA(train_path, test_path)
    elif sys.argv[3] == 'b':
        partB(train_path, test_path)
    elif sys.argv[3] == 'c':
        partC(train_path, test_path)
    elif sys.argv[3] == 'd':
        partD(train_path, test_path)
    elif sys.argv[3] == 'e':
        partE(train_path, test_path)
    elif sys.argv[3] == 'f':
        partF(train_path, test_path)