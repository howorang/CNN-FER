import random

import cv2
import numpy as np


def mirror(source_image):
    height, width, channels = source_image.shape
    mirrored_image = np.zeros((height, width, 3), np.uint8)
    cv2.flip(source_image, 1, mirrored_image)
    return mirrored_image


def rotate(source_image):
    degrees = random.randint(0, 7) * (-1 if random.randint(0, 1) == 0 else 1)
    height, width, channels = source_image.shape
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)
    rotated_image = cv2.warpAffine(source_image, rotation_matrix, (width, height))
    return rotated_image


def translate(source_image):
    x = random.randint(0, 10)
    y = random.randint(0, 10)
    M = np.float32([[1, 0, x], [0, 1, y]])
    height, width, channels = source_image.shape
    translated_image = cv2.warpAffine(source_image, M, (width, height))
    return translated_image


def random_shapes(source_image):
    source_image = source_image.copy()
    shape_size = random.randint(1, 20)
    height, width, channels = source_image.shape
    if random.randint(0, 1) == 0:
        cv2.circle(img=source_image,
                   center=(random.randint(0, width - shape_size), random.randint(0, height - shape_size)),
                   radius=shape_size,
                   color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                   )
    else:
        origin = (random.randint(0, width - shape_size), random.randint(0, height - shape_size))
        cv2.rectangle(img=source_image,
                      pt1=origin,
                      pt2=(origin[0] + shape_size, origin[1] + shape_size),
                      color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                      )
        return source_image


ops = [translate, rotate, mirror, random_shapes]


def generate_random_changes_pipe(number_of_steps):
    pipeline = []
    for _ in number_of_steps:
        pipeline.append(ops[random.randint(0, len(ops) - 1)])
    return pipeline


def execute_pipe(source_img, pipe):
    for op in pipe:
        source_img = op(source_img)
