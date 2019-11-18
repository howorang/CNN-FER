# Equal dataset distribution
# Equal emotion + dataset distribution
import math

import numpy as np

from ImageHandle import Emotion
import random
from augmentation import load_and_preprocess_image, augment_randomly


def get_dataset(loaders, target_size):
    dataset_handles = {}
    target_handles = []
    target_images = []
    target_labels = []
    for loader in loaders:
        dataset, handles = loader.get_image_handles()
        dataset_handles[dataset] = handles
    dataset_emotions_count = math.ceil(target_size / len(loaders) / len(Emotion))
    result_size = dataset_emotions_count * len(loaders) * len(Emotion)
    i = 0
    while i < result_size:
        for dataset in dataset_handles.keys():
            for emotion in list(Emotion):
                target_handles.append(get_image(dataset, emotion, dataset_handles))
                i += 1
    i = 0
    for handle in target_handles:
        target_images.append(augment_randomly(load_and_preprocess_image(handle.path)))
        target_labels.append(handle.emotion.value)
        i += 1
        print("Rendering " + str(i) + " out of " + str(len(target_handles)))
    return np.array(target_labels), np.array(target_images)


def get_image(dataset, emotion, dataset_handles):
    handles = dataset_handles[dataset]
    eligible_handles = [i for i in handles if (not i.was_used and i.emotion == emotion)]
    if len(eligible_handles) > 0:
        image = random.choice(eligible_handles)
        image.was_used = True
        return image
    else:
        for image in [i for i in handles if i.emotion == emotion]:
            image.was_used = False
        return get_image(dataset, emotion, dataset_handles)
