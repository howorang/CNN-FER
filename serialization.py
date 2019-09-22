import datetime

import h5py
import numpy as np


def save_dataset(path, labels, imgs):
    filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    f = h5py.File(path + filename, 'w')
    f.create_dataset("labels", data=labels)
    f.create_dataset("imgs", data=imgs)
    f.close()


def load_dataset(path):
    f = h5py.File(path, 'r')
    labels = f['labels'].value
    imgs = f['imgs'].value
    f.close()
    if len(imgs.shape) == 3:
        resotred_shape = imgs.shape + (1,)
        images = np.reshape(imgs, resotred_shape)
    return labels, imgs
