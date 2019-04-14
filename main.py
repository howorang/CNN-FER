import data_loader
import tensorflow as tf
import matplotlib.pyplot as plt
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
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

test = train_dataset.make_one_shot_iterator().get_next()

batch_size = 80
steps_per_epoch_val = int(train_set_count / batch_size)
history = model.fit(train_dataset.make_one_shot_iterator(),
                    validation_data=val_dataset.make_one_shot_iterator(),
                    epochs=10,
                    batch_size=batch_size,
                    verbose=1,
                    steps_per_epoch=steps_per_epoch_val,
                    validation_steps=2)

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
