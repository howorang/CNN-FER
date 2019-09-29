import datetime
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

from resnet import resnet50
from serialization import load_dataset

IMAGE_PATH = "data/Emotion"
OUTPUT_PATH = 'data/output/'




labels, images = load_dataset('example20190925180901')
# if 1 channel dataset


if len(images.shape) == 3:
    resotred_shape = images.shape + (1,)
    images = np.reshape(images, resotred_shape)
# if 1 channel dataset



images, X_test, labels, y_test = train_test_split(images, labels, test_size=0.15)

labels = np.repeat(labels, 5, axis=0)
images = np.repeat(images, 5, axis=0)

batch_size = 20

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
print(logdir)

model = resnet50()
model.compile(optimizer=keras.optimizers.Adam(),
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
