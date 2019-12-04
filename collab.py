import datetime
import os
from enum import Enum

import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras

from resnet import resnet50
from serialization import load_dataset
from vgg19 import vgg19


class DatasetFiles(Enum):
    RAFD = 'rafd'
    CK = 'ck'
    JAFFE = 'jaffe'
    HETERO = 'hetero'


class Network(Enum):
    VGG19 = vgg19()
    RESNET = resnet50()


class Optimizer(Enum):
    DEFAULT = keras.optimizers.Adam()
    LR = keras.optimizers.Adam(lr=0.00001)


HP_DATASET = hp.HParam('dataset', hp.Discrete({x.name for x in list(DatasetFiles)}))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete({x.name for x in list(Optimizer)}))
HP_NETWORK = hp.HParam('network', hp.Discrete({x.name for x in list(Network)}))


def run(run_name, hparams):
    labels, images = load_dataset(hparams[HP_DATASET] + '3000.h5')
    # if 1 channel dataset

    labels = [to_categorical(i - 1, 6) for i in labels]
    labels = np.array(labels)

    if len(images.shape) == 3:
        resotred_shape = images.shape + (1,)
        images = np.reshape(images, resotred_shape)
    # if 1 channel dataset

    images, X_test, labels, y_test = train_test_split(images, labels, test_size=0.15)

    labels = np.repeat(labels, 4, axis=0)
    images = np.repeat(images, 4, axis=0)

    batch_size = 20

    logdir = os.path.join("logs", run_name)

    print(logdir)

    model = Network[hparams[HP_NETWORK]].value
    model.compile(optimizer=Optimizer[hparams[HP_OPTIMIZER]].value,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(x=images,
                        y=labels,
                        validation_split=0.15,
                        epochs=30,
                        verbose=1,
                        batch_size=32,
                        shuffle=True,
                        callbacks=[tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),
                                   hp.KerasCallback(logdir, hparams)])

    test_scalar_loss = model.evaluate(
        x=X_test,
        y=y_test,
        batch_size=32,
        verbose=1
    )
    model.save(os.path.join(logdir + os.path.sep + "model"))
    print(str(test_scalar_loss), file=open(os.path.join(logdir + os.path.sep + "scalar.txt"), "a"))


for dataset in HP_DATASET.domain.values:
    for network in HP_NETWORK.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            hparams = {
                HP_DATASET: dataset,
                HP_NETWORK: network,
                HP_OPTIMIZER: optimizer,
            }
            run_name = hparams[HP_DATASET] + " " + hparams[HP_NETWORK] + " " + hparams[
                HP_OPTIMIZER] + " " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            run(run_name, hparams)
