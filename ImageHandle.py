from dataclasses import dataclass
from enum import Enum

from augmentation_ops import AugmentationOp


class Dataset(Enum):
    CK = 1,
    JAFFE = 2,
    RAFD = 3


# Emotion Enum
# It is an universal enum that omits emotions such as:
# contemptuous and neutral

class Emotion(Enum):
    ANGRY = 1,
    DISGUSTED = 2,
    FEARFUL = 3,
    SURPRISED = 4,
    SAD = 5,
    HAPPY = 6


class Augmentation:
    operation: AugmentationOp
    params: {}


@dataclass
class ImageHandle:
    dataset: Dataset
    model: str
    emotion: Emotion
    path: str
    augmentations: []
