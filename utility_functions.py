import numpy as np


def from_categorigical_to_int(label: np.ndarray):
    for i in range(0, label.size):
        if label[i] == 1:
            return 1
