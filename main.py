import data_loader
import tensorflow as tf
from tensorflow import keras

dataset, length = data_loader.get_dataset()
dataset = dataset.repeat(10)
dataset = dataset.shuffle(buffer_size=20)
length = length * 10


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
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(192, 3)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
batches_count = int(length / 20)

history = model.fit(x=train_dataset.make_one_shot_iterator(),
                    validation_data=val_dataset.make_one_shot_iterator(),
                    epochs=40,
                    batch_size=20,
                    verbose=1,
                    steps_per_epoch=1,
                    validation_steps=1)
