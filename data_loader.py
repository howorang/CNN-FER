import os
import tensorflow as tf

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
            label, path, path2, path3, path4 = get_label_path(fullpath, f)
            img_labels.append(label)
            img_labels.append(label)
            img_labels.append(label)
            img_labels.append(label)
            imgs.append(load_and_preprocess_image(path))
            imgs.append(load_and_preprocess_image(path2))
            imgs.append(load_and_preprocess_image(path3))
            imgs.append(load_and_preprocess_image(path4))
    return img_labels, imgs


def get_label_path(path, filename):
    image_path = '\\data\\output\\' + filename.replace('_emotion.txt', '.png')
    image_path_t = image_path.replace('.png', '_translated.png')
    image_path_m = image_path.replace('.png', '_mirrored.png')
    image_path_r = image_path.replace('.png', '_rotated.png')

    fp = open(path, "r")
    label = int(float(fp.readline()))
    fp.close()
    return label, image_path, image_path_m, image_path_r, image_path_t


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image
