import numpy as np

from augmentation import augment_random_images_randomly
from jaffe_loader import get_dataset
from serialization import save_dataset

labels, imgs = get_dataset()
labels, imgs = augment_random_images_randomly(labels, imgs, 2000)
imgs = np.divide(imgs, 255)
save_dataset("jaffe", labels, imgs)