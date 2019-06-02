import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

label_dict = {
    0: "neutral",
    1: "anger",
    2: "contempt",
    3: "disgust",
    4: "fear",
    5: "happy",
    6: "sadness",
    7: "surprise"
}


def get_dataset():
    labels, imgs = load_images("data/Emotion")
    length = len(labels)
    return labels, imgs, length


def load_images(startpath):
    imgs = []
    img_labels = []
    for paths, dirs, files in os.walk(startpath):
        for f in files:
            fullpath = os.path.join(paths, f)
            label, paths = get_label_path(fullpath, f)
            for path in paths:
                img_labels.append(to_categorical(label, 8))
                imgs.append(load_and_preprocess_image(path))
    return np.array(img_labels), np.array(imgs)


def get_label_path(path, filename):
    image_path = 'data/output/' + filename.replace('_emotion.txt', '.png')
    image_path_t = image_path.replace('.png', '_translated.png')
    image_path_m = image_path.replace('.png', '_mirrored.png')
    image_path_r = image_path.replace('.png', '_rotated.png')

    fp = open(path, "r")
    label = int(float(fp.readline()))
    fp.close()
    return label, [image_path, image_path_m, image_path_r, image_path_t]


def load_and_preprocess_image(path):
    image = load_img(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = image.convert('L')  # to grayscale
    image = image.resize((192, 192))
    arr = img_to_array(image)
    arr = np.divide(arr, 255)
    return arr


labels, images, length = get_dataset()
labels = labels.repeat(10)
images = images.repeat(10)
length = length * 10
print(length)
VAL_SIZE = 0.15
TST_SIZE = 0.15
TRN_SIZE = 0.70


batch_size = 20

model = keras.Sequential([
    keras.layers.Flatten(batch_size=batch_size),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x=images,
                    y=labels,
                    validation_split=0.15,
                    epochs=1000,
                    verbose=1,
                    batch_size=32,
                    shuffle=True)

history_dict = history.history
history_dict.keys()

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
