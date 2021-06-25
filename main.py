import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.02

def to_one_hot(x):
    return np.array([1 if i == x else 0 for i in range(10)]).reshape((10, 1))

def get_data():
    """Returns MNIST Data, scaled betwen 0 and 1"""

    # Getting mnist dataset and scaling it between [0, 1]
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print(f"{x_train[0].shape=}")
    print(f"{y_train[0].shape=}")

    # Reshaping data to be numpy vectors
    x_train = [np.array(a).reshape((784, 1)) for a in x_train]
    y_train = [to_one_hot(a) for a in y_train]
    x_test = [np.array(a).reshape((784, 1)) for a in x_test]
    y_test = [to_one_hot(a) for a in y_test]

    print(f"{x_train[0].shape=}")
    print(f"{y_train[0].shape=}")

    return(x_train, y_train), (x_test, y_test)

def sig(x):
    return 1 / (1 + np.exp(x))

def sigg(x):
    return sig(x) * (1 - sig(x))

(x_train, y_train), (x_test, y_test) = get_data()

NEURONS_HIDDEN = 128

# Weight matricies
W1 = np.ones((NEURONS_HIDDEN, 784 + 1))
W2 = np.ones((10, NEURONS_HIDDEN + 1))

def propagate(W, x):
    return sig(W @ np.insert(x, 0, -1, axis=0))

def bpropagate(W, x):
    return sigg(W @ np.insert(x, 0, -1, axis= 0))

# Training the model
for i in range(200):

    # Updating weights in the last layer

    # Calculating network output
    l1 = x_train[i]
    l2 = propagate(W1, l1)
    l3 = propagate(W2, l2)

    print(l3)

    # Updating W2

    # Calculating loss
    loss = (1 / 10) * sum((l3[k] - y_train[k]) ** 2 for k in range(10))

    print(f"{W2.shape=}")
    print(f"{l2.shape=}")
    print(f"{l3.shape=}")
    print(f"{bpropagate(W2, l2).shape=}")

    gradient_W1 = (-1) * bpropagate(W2, l2) 

    W2 = W2 - LEARNING_RATE * (1 / len(x_train)) * gradient_W1

    break