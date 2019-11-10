import cv2
import numpy as np

import ck_loader
import jaffe_loader
import rafd_loader
from heterogenic_loader import get_dataset
from serialization import save_dataset

labels, imgs = get_dataset([jaffe_loader, rafd_loader, ck_loader], 1000)
imgs = np.divide(imgs, 255)
save_dataset("example", labels, imgs)