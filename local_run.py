from augmentation import augment_random_images_randomly
from ck_loader import get_dataset
from serialization import save_dataset

labels, imgs = get_dataset()
labels, imgs = augment_random_images_randomly(labels, imgs, 10000)
save_dataset("example", labels, imgs)