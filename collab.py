import datetime
import os

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

from sklearn.model_selection import train_test_split

from data_loader import get_dataset
from utils import get_model_memory_usage

labels, images, length = get_dataset()

images, X_test, labels, y_test = train_test_split(images, labels, test_size=0.15)

labels = np.repeat(labels, 10, axis=0)
images = np.repeat(images, 10, axis=0)

batch_size = 20

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
print(logdir)

model = keras.Sequential([
    keras.layers.Conv2D(filters=32,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu',
                        input_shape=(192, 192, 1)),
    keras.layers.Conv2D(filters=32,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu'),
    keras.layers.Conv2D(filters=32,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2),
                              padding='valid'),
    keras.layers.Conv2D(filters=64,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu'),
    keras.layers.Conv2D(filters=64,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu'),
    keras.layers.Conv2D(filters=64,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2),
                              padding='valid'),
    keras.layers.Conv2D(filters=128,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu'),
    keras.layers.Conv2D(filters=128,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu'),
    keras.layers.Conv2D(filters=128,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu'),

    keras.layers.MaxPooling2D(pool_size=(2, 2),
                              padding='valid'),
    keras.layers.Conv2D(filters=256,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu'),
    keras.layers.Conv2D(filters=256,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu'),
    keras.layers.Conv2D(filters=256,
                        kernel_size=3,
                        strides=(1, 1),
                        activation='relu'),

    keras.layers.MaxPooling2D(pool_size=(2, 2),
                              padding='valid'),

    # first flatten
    keras.layers.AveragePooling2D(pool_size=(6),
                                  strides=None),
    keras.layers.Flatten(),
    keras.layers.Dense(8, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(x=images,
                    y=labels,
                    validation_split=0.15,
                    epochs=90,
                    verbose=1,
                    batch_size=32,
                    shuffle=True,
                    callbacks=[tensorboard_callback])

test_scalar_loss = model.evaluate(
    x=X_test,
    y=y_test,
    batch_size=32,
    verbose=1
)


print(get_model_memory_usage(32, model))

history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()




