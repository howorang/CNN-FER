import os
import tensorflow as tf
import cv2
import numpy as np

label_dict = {
    0: "neutral",
    1: "anger",
    2: "contempt",
    3: "disgust",
    4: "fear",
    5: "happy",
    6: "sadness",
    7: "surprise"
}

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def get_dataset():
    labels, imgs = load_images("data/Emotion")
    length = len(labels);
    labels = tf.one_hot(labels, 8)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    image_ds = tf.data.Dataset.from_tensor_slices(imgs)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return image_label_ds, length


def load_images(startpath):
    imgs = []
    img_labels = []
    for paths, dirs, files in os.walk(startpath):
        for f in files:
            fullpath = os.path.join(paths, f)
            label, path = get_label_path(fullpath)
            img_labels.append(label)
            imgs.append(load_and_preprocess_image(path))
    return img_labels, imgs


def get_label_path(path):
    image_path = path.replace('data/Emotion', 'data/cohn-kanade-images').replace('_emotion.txt', '.png')
    fp = open(path, "r")
    label = int(float(fp.readline()))
    fp.close()
    return label, image_path


def load_and_preprocess_image(path):
    return preprocess_image(path)


def preprocess_image(image):
    image = cv2.imread(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))


    for (x, y, w, h) in faces:
        crop_img = image[y:y + h, x:x + w]
        cv2.imshow('image', crop_img)
        np_image_data = np.asarray(crop_img)

        tf.image.decode_image(np_image_data)
    return image
