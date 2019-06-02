import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import os

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
    labels, imgs = load_images("/content/data/Emotion")
    length = len(labels);
    labels = tf.one_hot(labels, 8)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(labels, tf.int64))
    image_ds = tf.data.Dataset.from_tensor_slices(imgs)
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))
    return image_label_ds, length


def load_images(startpath):
    imgs = []
    img_labels = []
    for paths, dirs, files in os.walk(startpath):
        for f in files:
            fullpath = os.path.join(paths, f)
            label, path, path2, path3, path4 = get_label_path(fullpath, f)
            img_labels.append(label)
            img_labels.append(label)
            img_labels.append(label)
            img_labels.append(label)
            imgs.append(load_and_preprocess_image(path))
            imgs.append(load_and_preprocess_image(path2))
            imgs.append(load_and_preprocess_image(path3))
            imgs.append(load_and_preprocess_image(path4))
    return img_labels, imgs


def get_label_path(path, filename):
    image_path = '/content/data/' + filename.replace('_emotion.txt', '.png')
    image_path_t = image_path.replace('.png', '_translated.png')
    image_path_m = image_path.replace('.png', '_mirrored.png')
    image_path_r = image_path.replace('.png', '_rotated.png')

    fp = open(path, "r")
    label = int(float(fp.readline()))
    fp.close()
    return label, image_path, image_path_m, image_path_r, image_path_t


def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.resize_images(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range
    return image

dataset, length = get_dataset()
dataset = dataset.repeat(10)
dataset = dataset.shuffle(buffer_size=20)
length = length * 10
print(length)
VAL_SIZE = 0.15
TST_SIZE = 0.15
TRN_SIZE = 0.70

train_set_count = int(TRN_SIZE * length)
train_dataset = dataset.take(train_set_count)
test_dataset = dataset.skip(train_set_count)

validation_set_count = int(VAL_SIZE * length)
val_dataset = test_dataset.skip(validation_set_count)
test_set_count = int(TST_SIZE * length)
test_dataset = test_dataset.take(test_set_count)

batch_size = 20

train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

model = keras.Sequential([
    keras.layers.Flatten(batch_size=batch_size),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_dataset.make_one_shot_iterator(),
                    validation_data=val_dataset.make_one_shot_iterator(),
                    epochs=20,
                    verbose=1,
                    steps_per_epoch=batch_size)

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
