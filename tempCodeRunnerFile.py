import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time

# Load and preprocess data
data = pd.read_csv('mnist_train.csv')
data = np.array(data)
np.random.shuffle(data)
data_train = data.T
X_train = data_train[1:] / 255  # Normalize data
Y_train = data_train[0]

def init_params():
    # Updated example to ensure all layers have gamma and beta parameters
    layer_dims = [784, 128, 64, 32, 10]
    params = {}
    for i in range(1, len(layer_dims)):
        params[f'W{i}'] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(2 / layer_dims[i-1])
        params[f'b{i}'] = np.zeros((layer_dims[i], 1))
        params[f'gamma{i}'] = np.ones((layer_dims[i], 1))
        params[f'beta{i}'] = np.zeros((layer_dims[i], 1))
    return params

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    e = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)

def batch_norm_forward(Z, gamma, beta, epsilon=1e-8):
    mean = np.mean(Z, axis=1, keepdims=True)
    variance = np.var(Z, axis=1, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    out = gamma * Z_norm + beta
    cache = (Z, Z_norm, mean, variance, gamma, beta, epsilon)
    return out, cache

def forward_prop(X, params, use_dropout=False, dropout_rate=0.5):
    cache = {'A0': X}
    L = len(params) // 4  # Number of layers (since params also include gamma and beta for batch norm)
    for l in range(1, L):
        Z = params[f'W{l}'].dot(cache[f'A{l-1}']) + params[f'b{l}']
        Z_bn, bn_cache = batch_norm_forward(Z, params[f'gamma{l}'], params[f'beta{l}'])  # Batch norm
        A = ReLU(Z_bn)  # Apply activation
        cache[f'Z{l}'], cache[f'Z_bn{l}'], cache[f'bn_cache{l}'], cache[f'A{l}'] = Z, Z_bn, bn_cache, A
        if use_dropout:
            D = np.random.rand(*A.shape) < dropout_rate
            A = np.multiply(A, D) / dropout_rate
            cache[f'D{l}'] = D
        cache[f'A{l}'] = A

    # Output layer
    ZL = params[f'W{L}'].dot(cache[f'A{L-1}']) + params[f'b{L}']
    AL = softmax(ZL)
    cache[f'Z{L}'], cache[f'A{L}'] = ZL, AL
    return AL, cache

def batch_norm_backward(dZ_norm, cache):
    Z, Z_norm, mean, variance, gamma, beta, epsilon = cache
    m = Z.shape[1]
    dgamma = np.sum(dZ_norm * Z_norm, axis=1, keepdims=True)
    dbeta = np.sum(dZ_norm, axis=1, keepdims=True)
    dZ = (1 / m) * gamma * (m * dZ_norm - np.sum(dZ_norm, axis=1, keepdims=True) - Z_norm * np.sum(dZ_norm * Z_norm, axis=1, keepdims=True)) / np.sqrt(variance + epsilon)
    return dZ, dgamma, dbeta

def one_hot(Y):
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def backward_prop(params, cache, X, Y, alpha, use_dropout=False, dropout_rate=0.5):
    grads = {}
    m = Y.size
    L = len(params) // 4  # Ensure this reflects the correct number of layers
    Y_hot = one_hot(Y)
    AL = cache[f'A{L}']
    dZL = AL - Y_hot
    grads[f'dW{L}'] = 1 / m * dZL.dot(cache[f'A{L-1}'].T)
    grads[f'db{L}'] = 1 / m * np.sum(dZL, axis=1, keepdims=True)

    for l in range(L-1, 0, -1):
        dA = params[f'W{l+1}'].T.dot(dZL)
        if use_dropout:
            dA = np.multiply(dA, cache[f'D{l}']) / dropout_rate
        dZ_bn = dA * (cache[f'Z_bn{l}'] > 0)  # Adjust as needed for activation
        dZ, dgamma, dbeta = batch_norm_backward(dZ_bn, cache[f'bn_cache{l}'])
        grads[f'dgamma{l}'] = dgamma  # Make sure this key exists
        grads[f'dbeta{l}'] = dbeta
        grads[f'dW{l}'] = 1 / m * dZ.dot(cache[f'A{l-1}'].T)
        grads[f'db{l}'] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dZL = dZ

    for l in range(1, L+1):
        params[f'W{l}'] -= alpha * grads[f'dW{l}']
        params[f'b{l}'] -= alpha * grads[f'db{l}']
        params[f'gamma{l}'] -= alpha * grads[f'dgamma{l}']
        params[f'beta{l}'] -= alpha * grads[f'dbeta{l}']
    return params


def get_predictions(AL):
    return np.argmax(AL, axis=0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha, use_dropout=False, dropout_rate=0.5):
    params = init_params()
    for i in range(iterations):
        AL, cache = forward_prop(X, params, use_dropout, dropout_rate)
        params = backward_prop(params, cache, X, Y, alpha, use_dropout, dropout_rate)
        if i % 10 == 0:
            predictions = get_predictions(AL)
            print(f'Iteration: {i}, Accuracy: {round(get_accuracy(predictions, Y) * 100, 2)}%')
    return params

# Training the model
iterations = 1000
alpha = 0.1
start = time.time()
params = gradient_descent(X_train, Y_train, iterations, alpha, use_dropout=True, dropout_rate=0.8)
finish = time.time()
print(f'Elapsed time for {iterations} iterations: {round(finish - start, 2)}s')
