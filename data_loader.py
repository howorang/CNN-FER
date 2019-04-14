import os
import tensorflow as tf
import numpy as np

# import matplotlib.pyplot as plt

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


# tf.enable_eager_execution()
def get_dataset():
    labels, imgs = load_images("data/Emotion")
    ds = tf.data.Dataset.from_tensor_slices((imgs, labels))
    return ds, len(labels)


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
    image = tf.read_file(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image

# plt.figure(figsize=(8, 8))
# for n, image in enumerate(image_ds.take(4)):
#     plt.subplot(2, 2, n + 1)
#     plt.imshow(image)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()
