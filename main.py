import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

LEARNING_RATE = 0.05
TRAINING_SIZE = 0   # Zero for entire MNIST Dataset
ARCHITECTURE = [784, 16, 16, 10]
BATCH_SIZE = 16

def to_one_hot(x):
    """Returns the one hot encoding of the given x"""
    return np.array([1. if i == x else 0. for i in range(ARCHITECTURE[-1])]).reshape((ARCHITECTURE[-1], 1))

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
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigg(x):
    """Derivative of sigmoid activation function"""
    return sig(x) * (1 - sig(x))

# Getting data
(x_train, y_train), (x_test, y_test) = get_data()

# Weight matricies
weights = [np.random.sample(ARCHITECTURE[l] * (ARCHITECTURE[l - 1] + 1)).reshape((ARCHITECTURE[l], ARCHITECTURE[l - 1] + 1)) for l in range(1, len(ARCHITECTURE))]

def propagate(W, x):
    """Forwardpropagation"""
    return sig(W.dot(np.insert(x, 0, 1, axis=0)))

def bpropagate(W, x):
    """Backpropagation"""
    return sigg(W.dot(x))

# Storing loss for all generations for plotting
loss_hist = []

def get_loss(t, y):
    assert len(t) == len(y)
    return sum([(t[d][0] - y[d][0]) ** 2 for d in range(len(t))])
    
def sig(x):
    return 1 / (1 + np.exp(-x))

def sigg(x):
    return sig(x) * (1 - sig(x))

def propagate(x, w):

    # Adding bias neuron
    x = np.insert(x, 0, 1, axis=0)

    net = w.dot(x)
    return sig(net)

print(f"Netowrk: " + " -> ".join([str(a) for a in ARCHITECTURE]))

weight_matrices = [np.random.sample(ARCHITECTURE[l] * (ARCHITECTURE[l - 1] + 1)).reshape((ARCHITECTURE[l], ARCHITECTURE[l - 1] + 1)) for l in range(1, len(ARCHITECTURE))]
print(f"Weights: " + " -> ".join([f"{w.shape}" for w in weight_matrices]))

iteration = 0

while True:

    iteration += 1

    total_loss = 0
    correct = 0

    gradients = [np.zeros_like(w, dtype='float64') for w in weight_matrices]

    data_point_id = 0

    # Iterating over all datapoints in the training data
    for d in range(len(x_train)):

        data_point_id += 1

        # Input and target outpug
        x, t = np.array(x_train[d], dtype='float64'), y_train[d]

        od = [x]
        for weight_matrix in weight_matrices:
            od.append(propagate(od[-1], weight_matrix))

        correct += 1 if np.argmax(od[-1]) == np.argmax(t) else 0

        loss = get_loss(t, od[-1])
        total_loss += loss

        d = []

        # Backpropagation
        for l in range(len(weight_matrices) - 1, -1, -1):
            #print(f"Current layer: {l}")
            if l == len(weight_matrices) - 1: 
                
                d.insert(0, (od[-1] - t) * od[-1] * (1 - od[-1]))
                grad = d[0] @ np.transpose(np.insert(od[-2], 0, 1, axis=0))

                assert gradients[l].shape == grad.shape
                gradients[l] += grad            
            
            else: # Last layer

                oj = od[l + 1]
                dl = d[0]
                w = weight_matrices[l + 1]
                
                # Weights x delta prev layer
                wd = np.transpose(w) @ dl

                # Removing delta for bias neuron
                wd = np.delete(wd, 0, axis=0)

                # d = sum (wjl dl) * oj * (1 - oj)                
                d.insert(0, wd * oj * (1 - oj))

                # oi with bias neuron
                p2 = np.transpose(np.insert(od[l], 0, 1, axis=0))

                # grad = d * p2               
                grad = d[0] * p2

                assert grad.shape == weight_matrices[l].shape
                assert grad.shape == gradients[l].shape
                gradients[l] += grad
                
        # Updating weights at end of batch
        if data_point_id % BATCH_SIZE == 0:
            # Updating weights of all layer
            for l in range(len(weight_matrices)):
                weight_matrices[l] -= LEARNING_RATE * (1.0 / BATCH_SIZE) * gradients[l]
            gradients = [np.zeros_like(w) for w in weight_matrices]

    # Updating loss
    total_loss *= (1.0 / len(x_train))
    loss_hist.append(total_loss)

    # User Info
    print(f"Iteration: {iteration:5d} -> {total_loss:.16f} ({correct:5d} / {len(x_train):5d} - {(correct / len(x_train) * 100):5.2f}%)", end="")
    print(f"\t\tmax weight: {max([np.max(a) for a in weights]):9.5f}", end="")
    print(f"\tmin weight: {min([np.min(a) for a in weights]):9.5f}", end="")
    print(f"\tavg weight: {sum([np.sum(a) for a in weights]) / sum([a.shape[0] * a.shape[1] for a in weights]):9.5f}")
