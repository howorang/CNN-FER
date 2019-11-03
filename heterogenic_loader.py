import random

import ck_loader
import jaffe_loader


def run():
    create_dataset([ck_loader, jaffe_loader])


def create_dataset(dataset_loaders, target_size):
    dataset_handles = _load_handles(dataset_loaders)
    per_dataset_image_count = target_size / len(dataset_handles)
    for dataset in dataset_handles.keys():
        if len(dataset_handles[dataset]) < per_dataset_image_count:
            new_images = _extend_dataset(dataset_handles[dataset], per_dataset_image_count)
            dataset_handles[dataset].append(new_images)


def _load_handles(dataset_loaders):
    dataset_handles = {}
    for loader in dataset_loaders:
        handles = loader.get_image_handles()
        dataset_handles[handles[0].dataset] = handles
    return dataset_handles


def _extend_dataset(image_handles, per_dataset_image_count):
    new_images = []
    for i in range(0, per_dataset_image_count):
        new_images.append(_get_new_image(image_handles))
    return new_images


def _get_new_image(image_handles):
    image_handle = random.choice(image_handles)
