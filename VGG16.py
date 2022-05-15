import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tqdm import tqdm

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


data_augmentation = keras.Sequential([
     # layers.RandomFlip(mode="horizontal"),
     # layers.RandomRotation(factor=0.2),
     layers.Normalization(mean=(0.4914, 0.4822, 0.4465), variance=(0.2023, 0.1994, 0.2010))
     ]
)

model_Resnet= keras.applications.vgg16.VGG16(weights='imagenet', input_shape=(32, 32, 3), include_top=False)

inputs = layers.Input(shape=(32,32,3))
x = data_augmentation(inputs)
x = model_Resnet(x)
x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(units=128, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.3)(x)
x = layers.Dense(units=64, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.3)(x)
outputs= layers.Dense(units=10)(x)

model=keras.Model( inputs=inputs, outputs=outputs)

## Hyperparameter setting and optimization ###
batch_size = 64
num_epochs = 20
learning_rate = 1e-4

(x_train,y_train),(x_test, y_test)=cifar10.load_data()
x_train = x_train.astype("float32")/255.
x_test = x_test.astype("float32")/255.
# Prepare the training dataset.
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)
# Prepare the test dataset.
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=len(x_test)).batch(batch_size)

optimizer = tf.keras.optimizers.Adam(learning_rate)
loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
acc_metric = keras.metrics.SparseCategoricalAccuracy()


@tf.function
def train_step(x, y):

  with tf.GradientTape() as tape:

    y_hat = model(x)

    loss = loss_func(y,y_hat)

  grads = tape.gradient(loss, model.trainable_variables)

  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  acc_metric.update_state(y, y_hat)
  accuracy =  acc_metric.result()
  return loss, accuracy


@tf.function
def test_step(x,y):
    y_hat = model(x, training=False)
    loss = loss_func(y, y_hat)
    acc_metric.update_state(y, y_hat)
    accuracy = acc_metric.result()
    return loss , accuracy

for epoch in tqdm(range(num_epochs)):
    print("\nEpoch [%d/%d]" % (epoch+1,num_epochs),)

    for (x_batch_train, y_batch_train) in train_dataset:
        loss , accuracy = train_step(x_batch_train, y_batch_train)

    print("training loss: " + str(np.mean(loss)) ," - training accuracy: " + str(accuracy.numpy()))

for (x_batch_test, y_batch_test) in test_dataset:
    loss , accuracy = test_step(x_batch_test, y_batch_test)

print('test  - loss: ' + str(np.mean(loss)) , '-  accuracy: ' + str(accuracy.numpy()))
