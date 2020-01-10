import math
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from keras import activations
from tensorflow.keras.models import load_model
from vis.input_modifiers import Jitter
from vis.utils import utils
from vis.visualization import visualize_activation

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


def apply_modifications_to_model(model):
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.

    Args:
        model: The `keras.models.Model` instance.

    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = 'tmp/' + next(tempfile._get_candidate_names()) + '.h5'
    try:
        model.save(model_path)
        return load_model(model_path)
    finally:
        os.remove(model_path)


def visualize_conv_layer_filters(model, layer_name):
    layer_idx = utils.find_layer_idx(model, layer_name)
    categories = np.random.permutation(10)[:15]
    vis_images = []
    image_modifiers = [Jitter(16)]
    for idx in categories:
        img = visualize_activation(model, layer_idx, filter_indices=idx, max_iter=1000, input_modifiers=image_modifiers)
        img = np.squeeze(img, 2)
        vis_images.append(img)
    plot_images(vis_images, None, 400, layer_name)


def visualize_dense_layer(model, layer_name, index_label_map):
    layer_idx = utils.find_layer_idx(model, layer_name)
    model.layers[layer_idx].activation = activations.linear
    model = apply_modifications_to_model(model)

    plt.rcParams['figure.figsize'] = (18, 6)

    # 20 is the imagenet category for 'ouzel'
    imgs = []
    labels = []
    for index, label in index_label_map.items():
        img = visualize_activation(model, layer_idx, filter_indices=index - 1, max_iter=500,
                                   verbose=True)
        img = np.squeeze(img, 2)
        imgs.append(img)
        labels.append(label)
    plot_images(imgs, labels, 400, layer_name)


model = load_model('HETERO RESNET DEFAULT 20200107-064749.h5')
# visualize_conv_layer_filters(model, 'block1_conv2')
visualize_dense_layer(model, 'fc1000', index_emotrion)
