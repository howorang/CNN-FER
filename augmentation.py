import random

import cv2
import numpy as np

from augmentation_ops import ops

NUMBER_OF_AUGMENTATION_STEPS = 3
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def augment_random_images_randomly(labels, imgs, target_count):
    count = len(labels)
    tmp_labels = labels.tolist()
    tmp_imgs = imgs.tolist()
    for i in range(count, target_count):
        label, img = choose_random_image(labels, imgs)
        augmented_img = augment_randomly(img)
        tmp_labels.append(label)
        tmp_imgs.append(augmented_img)
    return np.array(tmp_labels), np.array(tmp_imgs)


def choose_random_image(labels, imgs):
    dataset_length = len(labels)
    random_index = random.randint(0, dataset_length - 1)
    return labels[random_index], imgs[random_index]


def augment_randomly(img):
    pipe = generate_random_changes_pipe(NUMBER_OF_AUGMENTATION_STEPS)
    return execute_pipe(img, pipe)


def generate_random_changes_pipe(number_of_steps):
    pipeline = []
    for i in range(number_of_steps):
        pipeline.append(ops[random.randint(0, len(ops) - 1)])
    return pipeline


def execute_pipe(source_img, pipe):
    for op in pipe:
        source_img = op(source_img)
    return source_img


def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = crop_face(image)
    image = cv2.resize(image, (192, 192))
    return image


def crop_face(source_image):
    faces = faceCascade.detectMultiScale(source_image, 1.1, 3, minSize=(100, 100))
    for (x, y, w, h) in faces:
        crop_img = source_image[y:y + h, x:x + w]
        return crop_img
