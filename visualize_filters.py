import math
import os
import tempfile

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.utils.losses import SmoothedLoss
from tf_keras_vis.utils.callbacks import Print
from resnet import resnet50
import tensorflow as tf
from tensorflow.keras import backend as K

index_emotrion = {
    1: "ANGRY",
    2: "DISGUSTED",
    3: "FEARFUL",
    4: "SUPRISED",
    5: "SAD",
    6: "HAPPY"
}


def plot_images(imgs, img_titles, dpi, title):
    plt.title(title)
    cols = rows = math.ceil(math.sqrt(len(imgs)))
    rows = cols - math.floor((((cols * rows) - len(imgs)) / cols))
    fig, axs = plt.subplots(rows, cols)
    i = 0
    for row in range(0, rows):
        if i == len(imgs):
            break
        for col in range(0, cols):
            axs[row, col].imshow(imgs[i])
            if img_titles is not None:
                axs[row, col].set_title(img_titles[i])
            i = i + 1
            if i == len(imgs):
                break
    fig.set_dpi(dpi)
    plt.show()



def visualize_conv_layer_filters(model, layer_name):
    def model_modifier(m):
        new_model = tf.keras.Model(inputs=m.inputs, outputs=[m.get_layer(name=layer_name).output])
        new_model.layers[-1].activation = tf.keras.activations.linear
        return new_model

    activation_maximization = ActivationMaximization(model, model_modifier)
    num_of_filters = 16
    filter_numbers = np.random.choice(model.get_layer(name=layer_name).output.shape[-1], num_of_filters)
    vis_images = []
    for filter_number in enumerate(filter_numbers):
        # Define loss function that is sum of a filter output.
        loss = SmoothedLoss(filter_number)

        # Generate max activation
        activation = activation_maximization(loss)
        image = activation[0].astype(np.uint8)
        vis_images.append(image)
    plot_images(vis_images, None, 400, layer_name)


def visualize_dense_layer(model, layer_name, index_label_map, itr):
    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear
    imgs = []
    labels = []
    for index, label in index_label_map.items():
        activation_maximization = ActivationMaximization(model, model_modifier)
        loss = lambda x: K.mean(x[:, index - 1])
        activation = activation_maximization(loss, steps=itr,callbacks=[Print(interval=100)])
        img = activation[0].astype(np.uint8)
        img = np.squeeze(img, 2)
        cv2.imwrite(label + ".png", img)
        imgs.append(img)
        labels.append(label)
    plot_images(imgs, labels, 400, layer_name)


model = resnet50()
model.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.load_weights('logs\\HETERO RESNET DEFAULT 20200127-082347\\model_weights.h5')
# visualize_conv_layer_filters(model, 'conv1')


def visualize(model, itr):
    visualize_dense_layer(model, 'fc1000', index_emotrion, itr)