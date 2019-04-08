import os
import tensorflow as tf
import matplotlib.pyplot as plt

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


def load_images(startpath):
    img_paths = []
    img_labels = []
    for paths, dirs, files in os.walk(startpath):
        for f in files:
            fullpath = os.path.join(paths, f)
            label, path = get_label_path(fullpath)
            img_labels.append(label)
            img_paths.append(path)
    return img_labels, img_paths


def get_label_path(path):
    image_path = path.replace('data/Emotion', 'data/cohn-kanade-images').replace('_emotion.txt', '.png')
    fp = open(path, "r")
    label = int(float(fp.readline()))
    fp.close()
    return label, image_path


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image


labels, paths = load_images("data/Emotion")

path_ds = tf.data.Dataset.from_tensor_slices(paths)
image_ds = path_ds.map(load_and_preprocess_image)

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))

image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

