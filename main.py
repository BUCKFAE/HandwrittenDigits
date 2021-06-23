import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=20)

model.evaluate(x_test, y_test, verbose=2)

res = model(x_test).numpy()

for i in range(len(res)):
    num = np.argmax(res[i])

    if num != y_test[i]: 

        plt.imshow(x_test[i])
        plt.title(f"Prediction: {num}\nActual: {y_test[i]}")
        plt.xticks([])
        plt.yticks([])
        plt.show()
