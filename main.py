import data_loader
import tensorflow as tf

dataset, length = data_loader.get_dataset()

VAL_SIZE = 0.15
TST_SIZE = 0.15
TRN_SIZE = 0.70




dataset = dataset.shuffle(buffer_size=20)
train_dataset = dataset.take(int(TRN_SIZE * length))
test_dataset = dataset.skip(int(TRN_SIZE * length))

val_dataset = test_dataset.skip(int(VAL_SIZE * length))
test_dataset = test_dataset.take(int(TST_SIZE * length))