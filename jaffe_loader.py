import os

import cv2
import numpy as np
from keras.utils import to_categorical

from augmentation import load_and_preprocess_image

DATASET_PATH = "data/jaffe"
EXCLUDE_FEAR = True

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

labels = {
    'HA': 0,
    'SA': 1,
    'SU': 2,
    'AN': 3,
    'DI': 4,
    'FE': 5,
    'NE': 6
}


# 7 categories

def get_dataset():
    return load_images(DATASET_PATH)


def load_images(startpath):
    imgs = []
    img_labels = []
    for paths, dirs, files in os.walk(startpath):
        for filename in files:
            fullpath = os.path.join(paths, filename)
            label = get_label(filename)
            img_labels.append(label)
            imgs.append(load_and_preprocess_image(fullpath))
    return np.array(img_labels), np.array(imgs)


def get_label(filename):
    label = labels[filename[3:5]]
    return to_categorical(label, 7)
