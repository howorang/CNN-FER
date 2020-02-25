import datetime
import os
from enum import Enum
import io
import itertools

import numpy as np
import tensorflow as tf
import sklearn
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
import matplotlib.pyplot as plt


class DatasetFiles(Enum):
    RAFD = 'rafd'
    # CK = 'ck'
    # JAFFE = 'jaffe'
    # HETERO = 'hetero'


loaded_datasets = {
    DatasetFiles.RAFD: None,
    # DatasetFiles.CK: None,
    # DatasetFiles.JAFFE: None,
    # DatasetFiles.HETERO: None
}


class Network(Enum):
    VGG19 = vgg19()
    # RESNET = resnet()


class Optimizer(Enum):
    # DEFAULT = keras.optimizers.Adam()
    LR = keras.optimizers.Adam(lr=0.00001)


HP_DATASET = hp.HParam('dataset', hp.Discrete({x.name for x in list(DatasetFiles)}))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete({x.name for x in list(Optimizer)}))
HP_NETWORK = hp.HParam('network', hp.Discrete({x.name for x in list(Network)}))

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def load_or_get_dataset(dataset):
    if loaded_datasets[dataset] is None:
        loaded_datasets[dataset] = load_dataset("/content/drive/My Drive/ck/" + dataset.value + '3000.h5')
    labels, images = loaded_datasets[dataset]
    return labels, images


def run(run_name, hparams):
   # labels, images = load_dataset(hparams[HP_DATASET] + '3000.h5')
    labels, images = load_or_get_dataset(DatasetFiles[hparams[HP_DATASET]])
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

    logdir = os.path.join("/content/drive/My Drive/ck/logs", run_name)

    print(logdir)

    model = Network[hparams[HP_NETWORK]].value
    model.compile(optimizer=Optimizer[hparams[HP_OPTIMIZER]].value,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

    def log_confusion_matrix(epoch, logs):
        cm = sklearn.metrics.confusion_matrix(np.argmax(y_test, axis=1),
                                              np.argmax(model.predict(X_test), axis=1),)
        figure = plot_confusion_matrix(cm, class_names=['ANGRY', 'DISGUSTED', 'FEARFUL', 'SURPRISED', 'SAD', 'HAPPY'])
        cm_image = plot_to_image(figure)
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    history = model.fit(x=images,
                        y=labels,
                        validation_split=0.15,
                        epochs=30,
                        verbose=1,
                        batch_size=32,
                        shuffle=True,
                        callbacks=[tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),
                                   hp.KerasCallback(logdir, hparams),
                                   keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)])

    test_scalar_loss = model.evaluate(
        x=X_test,
        y=y_test,
        batch_size=32,
        verbose=1
    )
    tf.keras.models.save_model(model, os.path.join(logdir + os.path.sep + "model.h5"))
    print(str(test_scalar_loss), file=open(os.path.join(logdir + os.path.sep + "scalar.txt"), "a"))

combinations = []

for dataset in HP_DATASET.domain.values:
    for network in HP_NETWORK.domain.values:
        for optimizer in HP_OPTIMIZER.domain.values:
            if (network == 'VGG19'):
                combinations.append((dataset, network, 'LR'))
            else:
                combinations.append((dataset, network, 'DEFAULT'))
start_index = 0
for i in range(start_index, len(combinations)):
    run_params = combinations[i]
    hparams = {
        HP_DATASET: run_params[0],
        HP_NETWORK:run_params[1],
        HP_OPTIMIZER: run_params[2],
    }
    run_name = hparams[HP_DATASET] + " " + hparams[HP_NETWORK] + " " + hparams[
        HP_OPTIMIZER] + " " + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run(run_name, hparams)
