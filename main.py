import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

LEARNING_RATE = 0.05
TRAINING_SIZE = 4000   # Zero for entire MNIST Dataset
ARCHITECTURE = [784, 32, 16, 10]
ITERATIONS = 500
BATCH_SIZE = 128

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

# Training the model
for iteration in range(ITERATIONS):

    # Storing stats for this gen
    total_loss = 0
    correct = 0
    
    # Storing gradients for weights
    gradients = [np.zeros_like(w, dtype='float64') for w in weights]

    # Iterating over all datapoints in the training data
    for d in range(len(x_train)):

        # Input and target outpug
        xd, td = np.array(x_train[d], dtype='float64'), y_train[d]

        # Stores network output for all layer
        od = [xd]
        
        # Calculates network output
        for w in weights:
            res = propagate(w, od[-1])
            od.append(res)

        # Updating loss / accuracy for given data point
        loss = sum([(od[-1][i][0] - td[i][0]) ** 2 for i in range(ARCHITECTURE[-1])])
        total_loss += loss

        # Counter for MNIST data
        correct += 1 if np.argmax(od[-1]) == np.argmax(td) else 0
        
        # Use this when using XOR data
        # correct += 1 if abs(od[-1] - td) < 0.5 else 0

        # Stores delta for neurons of prev layer
        delta_prev = None

        # Updating weight matrices between all layer
        for l in range(len(weights) - 1, -1, -1):
         
            # Updating last layer
            if l == len(weights) - 1:  

                # If using sigmoid: delta_prev = (od - td) * od * (1 - od)
                delta_prev = (od[-1] - td) * od[-1] * (1 - od[-1])

                # Grad weights: delta x output last layer, inserting 1 for bias
                grad = delta_prev @ np.transpose(np.insert(od[-2], 0, 1, axis=0))
  
                # Appending gradient
                assert grad.shape == gradients[l].shape
                gradients[l] += grad

            else:
                
                # Getting the neurons of the prev layer
                weights_prev = np.transpose(weights[l + 1])

                # Delta current layer is f'(W dot delta_prev)
                delta_curr = bpropagate(weights_prev, delta_prev)

                # Deleting delta for bias neuron
                delta_curr = np.delete(delta_curr, 0, axis=0)

                # Ouptut of neurons current layer (with added bias neuron)
                activations_curr = np.transpose(np.insert(od[l], 0, 1, axis=0))

                # Delta weights: delta neurons x output current layer
                delta_weights = delta_curr.dot(activations_curr)

                # Adding the gradients of the weights
                assert delta_weights.shape == gradients[l].shape
                gradients[l] += delta_weights

                # Storing current delta of neurons for next layer
                delta_prev = delta_curr

        # Updating weights at end of batch
        if d == len(x_train) - 1 or d % BATCH_SIZE == 0:
            # Updating weights of all layer
            for l in range(len(weights)):
                weights[l] -= LEARNING_RATE * (1.0 / BATCH_SIZE) * gradients[l]
            gradients = [np.zeros_like(w) for w in weights]

    # Updating loss
    total_loss *= (1.0 / len(x_train))
    loss_hist.append(total_loss)

    # User Info
    print(f"Iteration: {iteration:5d} -> {total_loss:.16f} ({correct:5d} / {len(x_train):5d} - {(correct / len(x_train) * 100):5.2f}%)", end="")
    print(f"\t\tmax weight: {max([np.max(a) for a in weights]):9.5f}", end="")
    print(f"\tmin weight: {min([np.min(a) for a in weights]):9.5f}", end="")
    print(f"\tavg weight: {sum([np.sum(a) for a in weights]) / sum([a.shape[0] * a.shape[1] for a in weights]):9.5f}")

# Plotting loss
# plt.plot([i for i in range(ITERATIONS)], loss_hist)
# plt.show()