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

def load_images(startpath):
    for paths, dirs, files in os.walk(startpath):
        for f in files:
            fullpath = os.path.join(paths, f)
            load_file_into_tf(fullpath)


def load_file_into_tf(path):
    image_path = path.replace('data/Emotion', 'data/cohn-kanade-images').replace('_emotion.txt', '.png')
    fp = open(path, "r")
    label = label_dict.get(int(float(fp.readline())))
    fp.close()
    image = tf.image.decode_png(image_path)
    return label, image


load_images("data/Emotion")
