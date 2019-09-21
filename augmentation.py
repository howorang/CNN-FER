import random

from augmentation_ops import ops

NUMBER_OF_AUGMENTATION_STEPS = 3


def augment_random_images_randomly(labels, imgs, target_count):
    count = len(labels)
    for i in range(count, target_count):
        label, img = choose_random_image(labels, imgs)
        augmented_img = augment_randomly(img)
        labels.append(label)
        imgs.append(augmented_img)
    return labels, imgs


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
