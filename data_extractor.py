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

def load_images(startpath, newpath):
    imgs = []
    img_labels = []
    for paths, dirs, files in os.walk(startpath):
        for f in files:
            fullpath = os.path.join(paths, f)
            preprocess_image(get_final_image_path(fullpath), newpath, f.replace('_emotion.txt', '.png'))

def get_final_image_path(path):
    image_path = path.replace('data/Emotion', 'data/cohn-kanade-images').replace('_emotion.txt', '.png')
    fp = open(path, "r")
    fp.close()
    return image_path


def preprocess_image(image, newpath, filename):
    read_image = cv2.imread(image)
    if read_image is not None:
        gray = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))


        for (x, y, w, h) in faces:
            crop_img = read_image[y:y + h, x:x + w]
            path_to_write = newpath + filename
            cv2.imwrite(path_to_write, crop_img)

load_images("data/Emotion", "data/new/")