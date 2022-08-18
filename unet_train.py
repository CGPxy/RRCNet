#coding=utf-8
import matplotlib

import matplotlib.pyplot as plt
import argparse
import numpy as np  
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from PIL import Image  
import cv2
import random
import os
from tqdm import tqdm  
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import tensorflow as tf
from model_net import *
from loss_function import *

matplotlib.use("Agg")


def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    else:
        img = cv2.imread(path)
        img = np.array(img,dtype="float") / 255.0
    return img

filepath ='/media/dy/Data_2T/CGP/Unet_Segnet/data/new_Dataset_B/1/Train_images/Augementa/'

def get_train_val(val_rate=0.20):
    train_url = []
    train_set = []
    val_set = []
    for pic in os.listdir(filepath + 'Train_images/'):
        train_url.append(pic)
    random.shuffle(train_url)
    total_num = len(train_url)
    val_num = int(val_rate * total_num)
    for i in range(len(train_url)):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


# data for training
def generateData(batch_size, data=[]):
    # print 'generateData...'
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'Train_images/' + url)
            img = img_to_array(img)
            train_data.append(img)
            label = load_img(filepath + 'Train_labels/' + url, grayscale=True)
            label = img_to_array(label)
            train_label.append(label)
            if batch % batch_size == 0:
                # print 'get enough bacth!\n'
                train_data = np.array(train_data)
                train_label = np.array(train_label)
                yield (train_data, {"out2":train_label,"out3":train_label,"out4":train_label,"out5":train_label,"out6":train_label,"out7":train_label})
                train_data = []
                train_label = []
                batch = 0


# data for validation
def generateValidData(batch_size, data=[]):
    # print 'generateValidData...'
    while True:
        valid_data = []
        valid_label = []
        batch = 0
        for i in (range(len(data))):
            url = data[i]
            batch += 1
            img = load_img(filepath + 'Train_images/' + url)
            img = img_to_array(img)
            valid_data.append(img)
            label = load_img(filepath + 'Train_labels/' + url, grayscale=True)
            label = img_to_array(label)
            valid_label.append(label)
            if batch % batch_size == 0:
                valid_data = np.array(valid_data)
                valid_label = np.array(valid_label)
                yield (valid_data, {"out2":valid_label,"out3":valid_label,"out4":valid_label,"out5":valid_label,"out6":valid_label,"out7":valid_label})
                valid_data = []
                valid_label = []
                batch = 0


def BCE():
    def dice(y_true, y_pred):
        return tf.keras.metrics.binary_crossentropy(y_true, y_pred)
    return dice


from typing import Any, Callable, Dict, List, Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
  """Counts and returns model FLOPs.
  Args:
    model: A model instance.
    inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
      shape specifications to getting corresponding concrete function.
    output_path: A file path to write the profiling results to.
  Returns:
    The model's FLOPs.
  """
  if hasattr(model, 'inputs'):
    try:
      # Get input shape and set batch size to 1.
      if model.inputs:
        inputs = [
            tf.TensorSpec([1] + input.shape[1:], input.dtype)
            for input in model.inputs
        ]
        concrete_func = tf.function(model).get_concrete_function(inputs)
      # If model.inputs is invalid, try to use the input to get concrete
      # function for model.call (subclass model).
      else:
        concrete_func = tf.function(model.call).get_concrete_function(
            **inputs_kwargs)
      frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

      # Calculate FLOPs.
      run_meta = tf.compat.v1.RunMetadata()
      opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
      if output_path is not None:
        opts['output'] = f'file:outfile={output_path}'
      else:
        opts['output'] = 'none'
      flops = tf.compat.v1.profiler.profile(
          graph=frozen_func.graph, run_meta=run_meta, options=opts)
      return flops.total_float_ops
    except Exception as e:  # pylint: disable=broad-except
      logging.info(
          'Failed to count model FLOPs with error %s, because the build() '
          'methods in keras layers were not called. This is probably because '
          'the model was not feed any input, e.g., the max train step already '
          'reached before this run.', e)
      return None
  return None



def train(args): 
    EPOCHS = 50
    BS = 12


    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    # Open a strategy scope.
    with strategy.scope():
        model = RRCNet()
        model.summary()
        flops = try_count_flops(model)
        print(flops/1000000000,"GFlops")

    model.compile(loss={'out2': BCE(), 'out3': BCE(), 'out4': BCE(), 'out5': BCE(), 'out6': BCE(), 'out7': BCE()}, loss_weights={'out2': 2.0, 'out3': 1.0, 'out4': 1.0, 'out5': 1.0, 'out6': 1.0, 'out7': 1.0}, metrics=["accuracy"], optimizer=Adam(lr=1e-3)) # binary_focal_loss

    checkpointer = ModelCheckpoint(os.path.join(
        args['save_dir'], 'model_{epoch:03d}.hdf5'), monitor='val_acc', save_best_only=False, mode='max')

    tensorboard = TensorBoard(log_dir='./Ours/logs/Ablation/Segnet_DS/4/', histogram_freq=0, write_graph=True, write_images=True)

    train_set,val_set = get_train_val()
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = model.fit_generator(generator=generateData(BS,train_set),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,  
                    validation_data=generateValidData(BS,val_set),validation_steps=valid_numb//BS,callbacks=[checkpointer, tensorboard])   #,max_q_size=1


def args_parse():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--save_dir", default="/media/dy/Data_2T/CGP/Unet_Segnet/method/BASNet/Ours/model/Ablation/Segnet_DS/4/",
                    help="path to output model")
    args = vars(ap.parse_args()) 
    return args


if __name__=='__main__':  
    args = args_parse()
    train(args)  
