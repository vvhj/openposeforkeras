#!/usr/bin/env python

# point to python env

# which gpus to run on
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
sys.path.append("..")
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard, TerminateOnNaN
from keras import backend as K
from train_common import train
from train_common import prepare, validate, save_network_input_output, test_augmentation_speed
from ds_generators import DataGeneratorClient, DataIterator
from config import COCOSourceConfig, GetConfig
import math

use_client_gen = False
batch_size = 10 # set batch size
gamma = 0.333
stepsize = 121746*17 
task = sys.argv[1] if len(sys.argv)>1 else "train"
config_name = sys.argv[2] if len(sys.argv)>2 else "Canonical"
experiment_name = sys.argv[3] if len(sys.argv)>3 else None
if experiment_name=='': experiment_name=None
epoch = int(sys.argv[4]) if len(sys.argv)>4 and sys.argv[4]!='' else None
base_lr = 2e-5

config = GetConfig(config_name)

# point to data files
train_client = DataIterator(config, COCOSourceConfig("D:/big3down/coco2017data/coco_train_dataset.h5"), shuffle=True,
                            augment=True, batch_size=batch_size)
val_client = DataIterator(config, COCOSourceConfig("D:/big3down/coco2017data/coco_val_dataset.h5"), shuffle=False, augment=False,
                          batch_size=batch_size)

train_samples = train_client.num_samples()
val_samples = val_client.num_samples()

model, iterations_per_epoch, validation_steps, epoch, metrics_id, callbacks_list = \
    prepare(config=config, config_name=config_name, exp_id=experiment_name, train_samples = train_samples, val_samples = val_samples, batch_size=batch_size, epoch=epoch)
#-----------------------------------------
def step_decay(epoch):
    steps = epoch * iterations_per_epoch * batch_size
    lrate = base_lr * math.pow(gamma, math.floor(steps/stepsize))
    print("Epoch:", epoch, "Learning rate:", lrate)
    return lrate
LOGS_DIR = 'model/log'
TRAINING_LOG = 'model/csv/openpoes1.csv'
WEIGHT_DIR = 'model'
WEIGHTS_SAVE = 'temp.h5'
lrate = LearningRateScheduler(step_decay)
checkpoint = ModelCheckpoint(WEIGHT_DIR + '/' + WEIGHTS_SAVE, monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=1)
csv_logger = CSVLogger(TRAINING_LOG, append=True)
tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0, write_graph=True, write_images=False)
tnan = TerminateOnNaN()
max_iter = 100
callbacks_list = [lrate, checkpoint, csv_logger, tb, tnan]
#-----------------------------------------
#multisgd = MultiSGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False, lr_mult=lr_mult)
model.fit_generator(train_client.gen(),
                    steps_per_epoch=iterations_per_epoch,
                    epochs=max_iter,
                    callbacks=callbacks_list,
                    validation_data=val_client.gen(),
                    validation_steps=validation_steps,
                    verbose=1
                    )
model.save("openpose.h5")
