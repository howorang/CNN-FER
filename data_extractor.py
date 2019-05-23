import os
import cv2
import numpy as np
import random

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


def crop_faces(startpath, newpath):
    for paths, dirs, files in os.walk(startpath):
        for f in files:
            fullpath = os.path.join(paths, f)
            crop_face(get_final_image_path(fullpath), newpath, f.replace('_emotion.txt', '.png'))


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


def for_each_image(func, path):
    for paths, dirs, files in os.walk(path):
        for f in files:
            fullpath = os.path.join(paths, f)
            func(fullpath)


def mirror(fullpath):
    source_image = cv2.imread(fullpath)
    height, width, channels = source_image.shape
    mirrored_image = np.zeros((height, width, 3), np.uint8)
    cv2.flip(source_image, 1, mirrored_image)
    cv2.imwrite(fullpath.replace(".png", "_mirrored.png"), mirrored_image)


def mirror_images(source_path):
    for paths, dirs, files in os.walk(source_path):
        for f in files:
            fullpath = os.path.join(paths, f)
            mirror(fullpath)


def rotate_images(source_path):
    for paths, dirs, files in os.walk(source_path):
        for f in files:
            fullpath = os.path.join(paths, f)
            mirror(fullpath)


crop_faces("data/Emotion", "data/new/")
mirror_images("data/new")
import os
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


def crop_faces(startpath, newpath):
    for paths, dirs, files in os.walk(startpath):
        for f in files:
            fullpath = os.path.join(paths, f)
            crop_face(get_final_image_path(fullpath), newpath, f.replace('_emotion.txt', '.png'))


def get_final_image_path(path):
    image_path = path.replace('data/Emotion', 'data/cohn-kanade-images').replace('_emotion.txt', '.png')
    fp = open(path, "r")
    fp.close()
    return image_path


def crop_face(image, newpath, filename):
    read_image = cv2.imread(image)
    if read_image is not None:
        gray = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))

        for (x, y, w, h) in faces:
            crop_img = read_image[y:y + h, x:x + w]
            path_to_write = newpath + filename
            cv2.imwrite(path_to_write, crop_img)


def for_each_image(func, path):
    for paths, dirs, files in os.walk(path):
        for f in files:
            fullpath = os.path.join(paths, f)
            func(fullpath)


def mirror(fullpath):
    source_image = cv2.imread(fullpath)
    height, width, channels = source_image.shape
    mirrored_image = np.zeros((height, width, 3), np.uint8)
    cv2.flip(source_image, 1, mirrored_image)
    cv2.imwrite(fullpath.replace(".png", "_mirrored.png"), mirrored_image)


def rotate(fullpath):
    source_image = cv2.imread(fullpath)
    height, width, channels = source_image.shape
    angle = random.randint(0, 7)
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    dest_image = cv2.warpAffine(source_image, rotation_matrix)
    cv2.imwrite(fullpath.replace(".png", "_rotated.png"), dest_image)



crop_faces("data/Emotion", "data/new/")
mirror_images("data/new")

for_each_image(mirror, "data/new")
for_each_image(rotate, "data/new")
