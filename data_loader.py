import os

import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical

import const


def get_dataset():
    labels, imgs = load_images(const.IMAGE_PATH)
    length = len(labels)
    return labels, imgs, length


def load_images(startpath):
    imgs = []
    img_labels = []
    for paths, dirs, files in os.walk(startpath):
        for f in files:
            fullpath = os.path.join(paths, f)
            label, paths = get_label_path(fullpath, f)
            for path in paths:
                img_labels.append(to_categorical(label, 8))
                imgs.append(load_and_preprocess_image(path))
    return np.array(img_labels), np.array(imgs)


def get_label_path(path, filename):
    image_path = const.OUTPUT_PATH + filename.replace('_emotion.txt', '.png')
    image_path_t = image_path.replace('.png', '_translated.png')
    image_path_m = image_path.replace('.png', '_mirrored.png')
    image_path_r = image_path.replace('.png', '_rotated.png')

    fp = open(path, "r")
    label = int(float(fp.readline()))
    fp.close()
    return label, [image_path, image_path_m, image_path_r, image_path_t]


def load_and_preprocess_image(path):
    image = load_img(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = image.convert('L')  # to grayscale
    image = image.resize((192, 192))
    arr = img_to_array(image)
    arr = np.divide(arr, 255)
    return arr