import os
import re

import cv2
import numpy as np
from keras.utils import to_categorical

from ImageHandle import ImageHandle, Dataset, Emotion
from augmentation import load_and_preprocess_image

DATASET_PATH = "data/ck"
IMAGE_PATH = DATASET_PATH + "/cohn-kanade-images"
EMOTION_PATH = DATASET_PATH + "/Emotion"

FILENAME_RE = re.compile('(?P<model>.*)_(?P<session>.*)_(?P<frame>.*)')


# 8 categories

def get_dataset():
    return load_images(DATASET_PATH + "/Emotion")


def get_image_handles():
    startpath = DATASET_PATH + "/Emotion"
    handles = []
    for paths, dirs, files in os.walk(startpath):
        for filename in files:
            fullpath = os.path.join(paths, filename)
            label = _get_label(fullpath)
            image_path = _get_image_file_path(fullpath)
            if _to_universal_label(label) is None:
                continue
            handles.append(ImageHandle(Dataset.CK, _get_model(filename), _to_universal_label(label), image_path, []))
    return handles


def load_images(startpath):
    imgs = []
    img_labels = []
    for paths, dirs, files in os.walk(startpath):
        for f in files:
            fullpath = os.path.join(paths, f)
            label = to_categorical(_get_label(fullpath), 8)
            img_labels.append(label)
            image_path = _get_image_file_path(fullpath)
            imgs.append(load_and_preprocess_image(image_path))
    return np.array(img_labels), np.array(imgs)


def _to_universal_label(label):
    label_to_universal = {
        0: None,  # NEUTRAL
        1: Emotion.ANGRY,
        2: None,  # CONTEMPT
        3: Emotion.DISGUSTED,
        4: Emotion.FEARFUL,
        5: Emotion.HAPPY,
        6: Emotion.SAD,
        7: Emotion.SURPRISED
    }
    return label_to_universal[label]


def _get_model(filename):
    search_result = FILENAME_RE.search(filename)
    return search_result.group('model')


def _get_label(label_path):
    fp = open(label_path, "r")
    label = int(float(fp.readline()))
    fp.close()
    return label


def _get_image_file_path(label_file_path):
    return label_file_path.replace(EMOTION_PATH, IMAGE_PATH).replace("_emotion.txt", ".png")
