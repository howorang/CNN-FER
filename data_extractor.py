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


def crop_face(image, newpath, filename):
    read_image = cv2.imread(image)
    if read_image is not None:
        gray = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))

        for (x, y, w, h) in faces:
            crop_img = read_image[y:y + h, x:x + w]
            path_to_write = newpath + filename
            cv2.imwrite(path_to_write, crop_img)


def for_each_image(func, source_path, dest_path):
    for paths, dirs, files in os.walk(source_path):
        for f in files:
            fullpath = os.path.join(paths, f)
            func(fullpath, source_path, dest_path)


def mirror(fullpath, source_path, dest_path):
    source_image = cv2.imread(fullpath)
    height, width, channels = source_image.shape
    mirrored_image = np.zeros((height, width, 3), np.uint8)
    cv2.flip(source_image, 1, mirrored_image)
    cv2.imwrite(fullpath.replace(source_path, dest_path).replace(".png", "_mirrored.png"), mirrored_image)


def rotate(fullpath, source_path, dest_path):
    source_image = cv2.imread(fullpath)
    height, width, channels = source_image.shape
    angle = random.randint(0, 7) * (-1 if random.randint(0, 1) == 0 else 1)
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    dest_image = cv2.warpAffine(source_image, rotation_matrix, (width, height))
    cv2.imwrite(fullpath.replace(source_path, dest_path).replace(".png", "_rotated.png"), dest_image)


def translate(fullpath, source_path, dest_path):
    x = random.randint(0, 10)
    y = random.randint(0, 10)
    M = np.float32([[1, 0, x], [0, 1, y]])
    source_image = cv2.imread(fullpath)
    height, width, channels = source_image.shape
    dst = cv2.warpAffine(source_image, M, (width, height))
    cv2.imwrite(fullpath.replace(source_path, dest_path).replace(".png", "_translated.png"), dst)


crop_faces("data/Emotion", "data/croped/")

for_each_image(mirror, "data/croped", "data/mirrored")
for_each_image(rotate, "data/croped", "data/rotated")
for_each_image(translate, "data/croped", "data/translated")
