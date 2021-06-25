import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import sys

LEARNING_RATE = 0.02
TRAINING_SIZE = 0
NEURONS_HIDDEN = 128
ITERATIONS = 10_000

def to_one_hot(x):
    return np.array([1 if i == x else 0 for i in range(10)]).reshape((10, 1))

def get_data():
    """Returns MNIST Data, scaled betwen 0 and 1"""

    # Getting mnist dataset and scaling it between [0, 1]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshaping data to be numpy vectors
    x_train = [np.array(a).reshape((784, 1)) for a in x_train]
    y_train = [to_one_hot(a) for a in y_train]
    x_test = [np.array(a).reshape((784, 1)) for a in x_test]
    y_test = [to_one_hot(a) for a in y_test]

    # Returning data
    s = len(x_train) if TRAINING_SIZE == 0 else TRAINING_SIZE
    return(x_train[:s], y_train[:s]), (x_test, y_test)

def sig(x):
    return 1 / (1 + np.exp(x))

def sigg(x):
    return sig(x) * (1 - sig(x))

# Getting data
(x_train, y_train), (x_test, y_test) = get_data()

# Weight matricies
W1 = np.random.random_sample(NEURONS_HIDDEN * (784 + 1)).reshape((NEURONS_HIDDEN, 784 + 1))
W2 = np.random.random_sample(10 * (NEURONS_HIDDEN + 1)).reshape((10, NEURONS_HIDDEN + 1))

def propagate(W, x):
    return sig(W @ np.insert(x, 0, -1, axis=0))

def bpropagate(W, x):
    return sigg(W @ np.insert(x, 0, -1, axis= 0))

loss_hist = []

# Training the model
for iteration in range(ITERATIONS):

    loss = 0
    gradient = 0

    correct = 0

    for d in range(len(x_train)):

        xd, td = x_train[d], y_train[d]

        # Calculating network output
        l1 = xd
        l2 = propagate(W1, l1)
        l3 = propagate(W2, l2)

        # Updating loss / accuracy
        loss += sum([(l3[i][0] - td[i][0]) ** 2 for i in range(10)])
        correct += 1 if np.argmax(l3) == np.argmax(td) else 0

        # Updating weights to all neurons
        for neuron in range(10):
            gradient += (-1) * (td[neuron] - l3[neuron]) * bpropagate(W2[neuron], l2) * l2[neuron]

    # Updating gradient
    gradient *= 1 / (2 * len(x_train))
    W2 -= LEARNING_RATE * (1 / len(x_train)) * gradient

    # Updating loss
    loss /= (1 / len(x_train))
    loss_hist.append(loss)

    # User Info
    if iteration % 10 == 0: 
        print(f"{iteration:5d} -> {loss:16f} ({correct:5d} / {len(x_train):5d} - {(correct / len(x_train) * 100):2f}%)")

# Plotting loss
plt.plot([i for i in range(ITERATIONS)], loss_hist)
plt.show()