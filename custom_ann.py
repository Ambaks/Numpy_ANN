import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time


data = pd.read_csv('mnist_train.csv')

data = np.array(data)
m, n = data.shape                       # Get rows m and columns n from our data.
np.random.shuffle(data)

data_train = data.T
X_train = data_train[1:n] / 255         # Data is expected to be within range 0, 255. The division normalizes data to 0, 1.
Y_train = data_train[0]

def init_params():
    # Here the sparting parameters are random because we are building a model from scratch. 
    # By savind the weights and biases, we can iput those as starting params to get a 90% accuracy model from jump.
    W1 = np.random.randn(10, 784) * np.sqrt(1 / 784)
    b1 = np.random.randn(10, 1)
    W2 = np.random.randn(10, 10) * np.sqrt(1 / 10)      # Multiplication by square root of 1 / array_size is to, again, normalize data for better perfomance in our models.
    b2 = np.random.randn(10, 1)
    return W1, b1, W2, b2

def ReLU(Z):
    # Any non-linear activation function we want. In this case, ReLU. 
    # If using another acticvation func such as ELU, must specify alpha (learning rate) in params.
    return np.maximum(0, Z)

def softmax(Z):
        # The activation function applied to the second hidden layer to get the output.
        e = np.exp(Z - Z.max(axis=0, keepdims=True))
        return e/e.sum(axis=0, keepdims=True)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0


def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, 1, keepdims= True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, 1, keepdims= True)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 10 == 0):
            print(f'Iteration: {i}')
            print(f'Accuracy: {round(get_accuracy(get_predictions(A2), Y)*100, 2)}%\n')
    return W1, b1, W2, b2
iteration = 200
start = time.time()
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, iteration, 0.1)
finish = time.time()
print(f'Elapsed time for {iteration} iterations: {round(finish - start, 2)}s\n')
