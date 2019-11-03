import os
import re

import cv2
import numpy as np
from keras.utils import to_categorical

from ImageHandle import Dataset, ImageHandle, Emotion
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
    return _load_images(DATASET_PATH)


def get_image_handles():
    startpath = DATASET_PATH
    handles = []
    for paths, dirs, files in os.walk(startpath):
        for filename in files:
            fullpath = os.path.join(paths, filename)
            metadata = _get_metadata(filename)
            emotion_label = metadata['emotion']
            if (EXCLUDE_NEUTRAL and emotion_label == 'neutral') or (_to_universal_label(emotion_label) is None):
                continue
            handles.append(
                ImageHandle(Dataset.RAFD, metadata['model'], _to_universal_label(emotion_label), fullpath, []))
    return handles


def _to_universal_label(label):
    label_to_universal = {
        0: Emotion.ANGRY,
        1: None,  # CONTEMPTUOUS
        2: Emotion.DISGUSTED,
        3: Emotion.FEARFUL,
        4: Emotion.HAPPY,
        5: None,  # NEUTRAL
        6: Emotion.SURPRISED,
        7: Emotion.SAD
    }
    return label_to_universal[label]


def _load_images(startpath):
    imgs = []
    img_labels = []
    for paths, dirs, files in os.walk(startpath):
        for filename in files:
            fullpath = os.path.join(paths, filename)
            metadata = _get_metadata(filename)
            if EXCLUDE_NEUTRAL and metadata['emotion'] == 'neutral':
                continue
            img_labels.append(to_categorical(labels[metadata['emotion']]))
            imgs.append(load_and_preprocess_image(fullpath))
    return np.array(img_labels), np.array(imgs)


def _get_metadata(filename):
    search_result = LABEL_RE.search(filename)
    return {
        'dataset': 'rafd',
        'model': search_result.group('model'),
        'subset': search_result.group('subset'),
        'gender': search_result.group('gender'),
        'emotion': search_result.group('emotion'),
        'gaze': search_result.group('gaze')
    }
