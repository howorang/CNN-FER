import random

import cv2
import numpy as np

DEBUG_MODE = True


def mirror(source_image):
    height, width, channels = source_image.shape
    mirrored_image = np.zeros((height, width, 3), np.uint8)
    cv2.flip(source_image, 1, mirrored_image)
    if DEBUG_MODE:
        cv2.imshow('unmirrored', source_image)
        cv2.imshow('mirrored', mirrored_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mirrored_image


def rotate(source_image):
    degrees = random.randint(0, 7) * (-1 if random.randint(0, 1) == 0 else 1)
    height, width, channels = source_image.shape
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)
    rotated_image = cv2.warpAffine(source_image, rotation_matrix, (width, height))
    if DEBUG_MODE:
        cv2.imshow('unrotated', source_image)
        cv2.imshow('rotated', rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return rotated_image


def translate(source_image):
    x = random.randint(0, 10)
    y = random.randint(0, 10)
    M = np.float32([[1, 0, x], [0, 1, y]])
    height, width, channels = source_image.shape
    translated_image = cv2.warpAffine(source_image, M, (width, height))
    if DEBUG_MODE:
        cv2.imshow('untranslated', source_image)
        cv2.imshow('translated', translated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return translated_image


def random_shapes(source_image):
    channged_picture = source_image.copy()
    shape_size = random.randint(1, 20)
    height, width, channels = channged_picture.shape
    if random.randint(0, 1) == 0:
        cv2.circle(img=channged_picture,
                   center=(random.randint(0, width - shape_size), random.randint(0, height - shape_size)),
                   radius=shape_size,
                   color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                   thickness=-1
                   )
    else:
        origin = (random.randint(0, width - shape_size), random.randint(0, height - shape_size))
        cv2.rectangle(img=channged_picture,
                      pt1=origin,
                      pt2=(origin[0] + shape_size, origin[1] + shape_size),
                      color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
                      thickness=-1
                      )
    if DEBUG_MODE:
        cv2.imshow('unshaped', source_image)
        cv2.imshow('shaped', channged_picture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return channged_picture


ops = [translate, rotate, mirror, random_shapes]
