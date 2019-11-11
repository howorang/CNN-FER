import os

import cv2
import numpy as np
from keras.utils import to_categorical

from ImageHandle import ImageHandle, Dataset, Emotion
from augmentation import load_and_preprocess_image
from utility_functions import from_categorigical_to_int

DATASET_PATH = "data/jaffe"
EXCLUDE_FEAR = False

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


def get_image_handles():
    startpath = DATASET_PATH
    handles = []
    for paths, dirs, files in os.walk(startpath):
        for filename in files:
            fullpath = os.path.join(paths, filename)
            label = get_label(filename)
            if (EXCLUDE_FEAR and label == 5) or (to_universal_label(label) is None):
                continue
            handles.append(
                ImageHandle(Dataset.JAFFE, get_metadata(filename)['model'], to_universal_label(label), fullpath, []))
    return Dataset.JAFFE, handles


def to_universal_label(label):
    label_to_universal = {
        0: Emotion.HAPPY,
        1: Emotion.SAD,
        2: Emotion.SURPRISED,
        3: Emotion.ANGRY,
        4: Emotion.DISGUSTED,
        5: Emotion.FEARFUL,
        6: None,  # NEUTRAL
    }
    return label_to_universal[label]


def load_images(startpath):
    imgs = []
    img_labels = []
    arr_metadata = []
    for paths, dirs, files in os.walk(startpath):
        for filename in files:
            fullpath = os.path.join(paths, filename)
            label = to_categorical(get_label(filename), 7)
            if EXCLUDE_FEAR and label[5] == 0:
                continue
            img_labels.append(label)
            imgs.append(load_and_preprocess_image(fullpath))
    return np.array(img_labels), np.array(imgs), np.array(arr_metadata)


def get_label(filename):
    return labels[filename[3:5]]


def get_metadata(filename):
    return {
        'dataset': 'jaffe',
        'model': filename[0:2],
        'emotion': filename[3:5]
    }
