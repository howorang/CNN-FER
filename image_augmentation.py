import cv2
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def crop_face(source_image):
    gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 3, minSize=(100, 100))

    for (x, y, w, h) in faces:
        crop_img = source_image[y:y + h, x:x + w]
        return crop_img


def for_each_image(func, source_path, dest_path):
    for paths, dirs, files in os.walk(source_path):
        for f in files:
            fullpath = os.path.join(paths, f)
            func(fullpath, source_path, dest_path)