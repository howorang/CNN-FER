import os

import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from keras_preprocessing.image import load_img

DATASET_PATH = "data/ck"
IMAGE_PATH = DATASET_PATH + "/cohn-kanade-images"
EMOTION_PATH = DATASET_PATH + "/Emotion"


def get_dataset():
    return load_images(DATASET_PATH + "/Emotion")


def load_images(startpath):
    imgs = []
    img_labels = []
    for paths, dirs, files in os.walk(startpath):
        for f in files:
            fullpath = os.path.join(paths, f)
            label = get_label(fullpath)
            img_labels.append(label)
            image_path = get_image_file_path(fullpath, f)
            imgs.append(load_and_preprocess_image(image_path))
    return np.array(img_labels), np.array(imgs)


def get_label(label_path):
    fp = open(label_path, "r")
    label = int(float(fp.readline()))
    fp.close()
    return to_categorical(label, 8)


def get_image_file_path(label_file_path, label_filename):
    return label_file_path.replace(EMOTION_PATH, IMAGE_PATH) + label_filename.replace("_emotion.txt", "png")


def load_and_preprocess_image(path):
    image = load_img(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = image.convert('L')  # to grayscale
    image = image.resize((192, 192))
    arr = img_to_array(image)
    arr = np.divide(arr, 255)
    return arr
