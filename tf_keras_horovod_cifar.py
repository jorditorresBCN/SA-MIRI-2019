import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import backend as K
import horovod.tensorflow.keras as hvd
#import horovod.tensorflow as hvd_tf

import numpy as np
import argparse
import math
import time

from cifar import load_cifar
import custom_callbacks

# Horovod: initialize Horovod.
hvd.init()

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help="epochs by GPU")
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--learning_rate', type=float, default=0.01)

args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs
# Horovod: modified learning rate
lr = args.learning_rate * hvd.size()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

train_ds, test_ds = load_cifar(batch_size)

model = tf.keras.applications.resnet50.ResNet50(include_top=True, weights=None,
            input_shape=(32, 32, 3), classes=10)

opt = tf.keras.optimizers.SGD(lr)
opt = hvd.DistributedOptimizer(opt)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              experimental_run_tf_function=False,
              metrics=['accuracy'])

time_callback = custom_callbacks.TimeHistory()
avg_cb = custom_callbacks.MetricAverageCallback()

# Horovod: broadcast initial variable states from rank 0 to all other processes.
# This is necessary to ensure consistent initialization of all workers when
# training is started with random weights or restored from a checkpoint.
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
             time_callback,
             avg_cb]

model.fit(train_ds,
          steps_per_epoch=math.ceil(50000/batch_size),
          epochs=epochs,
          verbose=0,
          callbacks=callbacks
          )


if hvd.rank() == 0:
    times = avg_cb.reduced_time
# tentative code version SA-MIRI Lab8
    print("TIME: ", times)
    print("Average time: ", sum(times)/len(times))
    print("Average time without epoch 1: ", sum(times[1:])/len(times[1:]))
