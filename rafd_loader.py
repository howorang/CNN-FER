import os
import re

import cv2
import numpy as np
from keras.utils import to_categorical

from augmentation import load_and_preprocess_image

DATASET_PATH = "data/rafd"
EXCLUDE_NEUTRAL = True
LABEL_RE = re.compile('(?P<dataset_name>.*)_(?P<model>.*)_(?P<subset>.*)_(?P<gender>.*)_(?P<emotion>.*)_(?P<gaze>.*)')

labels = {
    'angry': 0,
    'contemptuous': 1,
    'disgusted': 2,
    'fearful': 3,
    'happy': 4,
    'neutral': 5,
    'surprised': 6,
    'sad': 7
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
            metadata = get_metadata(filename)
            if EXCLUDE_NEUTRAL and metadata['emotion'] == 'neutral':
                continue
            img_labels.append(to_categorical(labels[metadata['emotion']]))
            imgs.append(load_and_preprocess_image(fullpath))
    return np.array(img_labels), np.array(imgs)


def get_metadata(filename):
    search_result = LABEL_RE.search(filename)
    return {
        'model': search_result.group('model'),
        'subset': search_result.group('subset'),
        'gender': search_result.group('gender'),
        'emotion': search_result.group('emotion'),
        'gaze': search_result.group('gaze')
    }
