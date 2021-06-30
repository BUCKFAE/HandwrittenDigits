import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 0.16
TRAINING_SIZE = 1000
ARCHITECTURE = [784, 16, 16, 10]
ITERATIONS = 1000
BATCH_SIZE = 128

def to_one_hot(x):
    return np.array([1. if i == x else 0. for i in range(10)]).reshape((10, 1))

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
    return 1 / (1 + np.exp(-x))

def sigg(x):
    return sig(x) * (1 - sig(x))

# Getting data
(x_train, y_train), (x_test, y_test) = get_data()

# Weight matricies
weights = [np.random.random_sample(ARCHITECTURE[l] * (ARCHITECTURE[l - 1] + 1)).reshape((ARCHITECTURE[l], ARCHITECTURE[l - 1] + 1)) for l in range(1, len(ARCHITECTURE))]
# print('\n'.join([f"{w.shape=}" for w in weights]))

def propagate(W, x):
    return sig(W @ np.insert(x, 0, -1, axis=0))

def bpropagate(W, x):
    return sigg(W @ np.insert(x, 0, -1, axis=0))

loss_hist = []

# Training the model
for iteration in range(ITERATIONS):

    total_loss = 0
    
    # Storing gradients
    gradients = [np.zeros_like(w) for w in weights]

    correct = 0

    for d in range(len(x_train)):

        xd, td = np.array(x_train[d], dtype='float64'), y_train[d]


        # Calculating network output
        od = [xd]
        
        for w in weights:
            res = propagate(w, od[-1])
            od.append(res)

        #if d == 1: print(f"{od[-1]=}")
        # Updating loss / accuracy
        loss = sum([(od[-1][i][0] - td[i][0]) ** 2 for i in range(ARCHITECTURE[-1])])

        # if d == 1: print(f"{od[-1]=}")

        total_loss += loss
        correct += 1 if np.argmax(od[-1]) == np.argmax(td) else 0

        for l in range(len(weights) - 1, - 1, -1):
         
            if l == len(weights) - 1:
                gradients[l] += -(loss) * bpropagate(weights[l], od[-2]) * weights[l]

            else:
    
                # TODO: Rethink this
                p = gradients[l + 1]
                p = np.delete(p, 0, axis=1)
                p = np.prod(p, axis=0)

                prev = od[l]
                #print(f"{prev.shape=}")
                #print(f"{weights[l].shape=}")
                k = bpropagate(weights[l], prev) * np.transpose(np.insert(od[l], 0, -1, axis=0))

                #print(f"{p.shape=}")
                #print(f"{k.shape=}")
                #print(f"{gradients[l].shape=}")

                # print(f"{np.max(p)=}")


                res = p @ k
                #print(f"{res.shape=}")

                # TODO Loss removed as exists in prev?
                gradients[l] += res

        if d == len(x_train) - 1 or d % BATCH_SIZE == 0:
            for l in range(len(weights)):

                weights[l] += LEARNING_RATE * (1 / len(x_train)) * gradients[l]
            gradients = [np.zeros_like(w) for w in weights]

    # Updating loss
    total_loss *= (1.0 / len(x_train))
    loss_hist.append(total_loss)

    # User Info
    print(f"{iteration:5d} -> {total_loss:.16f} ({correct:5d} / {len(x_train):5d} - {(correct / len(x_train) * 100):.2f}%)")

# Plotting loss
plt.plot([i for i in range(ITERATIONS)], loss_hist)
plt.show()