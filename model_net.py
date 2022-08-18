# coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from PIL import Image
import cv2
import random
import os
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model,Sequential
from tensorflow.python.layers import utils
from tensorflow.keras import regularizers
import tensorflow_addons as  tfa


img_w = 384  
img_h = 384

def ConvBlock0(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same",dilation_rate=(2,2))(data)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)

    return LeakyReLU1
def ConvBlock1(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    return LeakyReLU1

def ConvBlock2(data, filte, rate):
    conv1 = Conv2D(filte, (3, 3), padding="same", dilation_rate=rate)(data)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    return LeakyReLU1

def ConvBlock(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    conv2 = Conv2D(filte, (3, 3), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)
    return LeakyReLU2



def updata1(data, skipdata, filte):
    shape = K.int_shape(skipdata)
    shape1 = K.int_shape(data)
    data1 = UpSampling2D((shape[1] // shape1[1], shape[2] // shape1[2]))(data)
    concatenate = Concatenate()([skipdata, data1])
    concatenate = ConvBlock(data=concatenate, filte=filte)

    return concatenate


def soout(data, strides, name):
    up = UpSampling2D(size=strides)(data)
    outconv01 = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(up)
    outconv02 = Activation('sigmoid', name=name)(outconv01)

    return outconv02

def RRCNet():

    inputs = Input(shape=(img_w, img_h, 3))

    # encoder
    conv_1 = Convolution2D(64, (3, 3), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (3, 3), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = tf.nn.max_pool_with_argmax(conv_2, ksize=(3, 3), strides=(2, 2), padding='SAME')

    conv_3 = Convolution2D(128, (3, 3), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (3, 3), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = tf.nn.max_pool_with_argmax(conv_4, ksize=(3, 3), strides=(2, 2), padding='SAME')

    conv_5 = Convolution2D(256, (3, 3), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (3, 3), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (3, 3), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = tf.nn.max_pool_with_argmax(conv_7, ksize=(3, 3), strides=(2, 2), padding='SAME')

    conv_8 = Convolution2D(512, (3, 3), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (3, 3), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (3, 3), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = tf.nn.max_pool_with_argmax(conv_10, ksize=(3, 3), strides=(2, 2), padding='SAME')

    conv_11 = Convolution2D(512, (3, 3), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (3, 3), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (3, 3), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = tf.nn.max_pool_with_argmax(conv_13, ksize=(3, 3), strides=(2, 2), padding='SAME')
    print("Build enceder done..")

    # between encoder and decoder
    conv_14 = Convolution2D(512, (3, 3), padding="same")(pool_5)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (3, 3), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (3, 3), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)
    out7 = soout(data=conv_16, strides=(32,32), name='out7')

    # decoder
    unpool_1 = tfa.layers.MaxUnpooling2D((2, 2))(conv_16, mask_5)
    concat_1 = Concatenate()([unpool_1, conv_13])

    conv_17 = Convolution2D(512, (3, 3), padding="same")(concat_1)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (3, 3), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(512, (3, 3), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)
    out6 = soout(data=conv_19, strides=(16,16), name='out6')

    unpool_2 = tfa.layers.MaxUnpooling2D((2, 2))(conv_19, mask_4)
    concat_2 = Concatenate()([unpool_2, conv_10])

    conv_20 = Convolution2D(512, (3, 3), padding="same")(concat_2)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(512, (3, 3), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(256, (3, 3), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)
    out5 = soout(data=conv_22, strides=(8,8), name='out5')

    unpool_3 = tfa.layers.MaxUnpooling2D((2, 2))(conv_22, mask_3)
    concat_3 = Concatenate()([unpool_3, conv_7])

    conv_23 = Convolution2D(256, (3, 3), padding="same")(concat_3)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(256, (3, 3), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)
    conv_25 = Convolution2D(128, (3, 3), padding="same")(conv_24)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)
    out4 = soout(data=conv_25, strides=(4,4), name='out4')

    unpool_4 = tfa.layers.MaxUnpooling2D((2, 2))(conv_25, mask_2)
    concat_4 = Concatenate()([unpool_4, conv_4])

    conv_26 = Convolution2D(128, (3, 3), padding="same")(concat_4)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Activation("relu")(conv_26)
    conv_27 = Convolution2D(64, (3, 3), padding="same")(conv_26)
    conv_27 = BatchNormalization()(conv_27)
    conv_27 = Activation("relu")(conv_27)
    out3 = soout(data=conv_27, strides=(2,2), name='out3')



    unpool_5 = tfa.layers.MaxUnpooling2D((2, 2))(conv_27, mask_1)
    concat_5 = Concatenate()([unpool_5, conv_2])

    conv_28 = Convolution2D(64, (3, 3), padding="same")(concat_5)
    conv_28 = BatchNormalization()(conv_28)
    conv_28 = Activation("relu")(conv_28)
    conv_29 = Convolution2D(1, (1, 1), padding="valid")(conv_28)
    out2 = Activation('sigmoid', name='out2')(conv_29)


    out0, out = MRNet(data=out2)

    out1 = FRNet(data=out0)

    model = Model(inputs=inputs, outputs=[out0, out1, out2,out3,out4,out5,out6,out7])
    return model


def MRNet(data):

    Conv1 = ConvBlock1(filte=64, data=data)

    Conv2 = ConvBlock2(data=Conv1, filte=64, rate=(2,2))

    Conv3 = ConvBlock2(data=Conv2, filte=64, rate=(3,3))

    Conv4 = ConvBlock2(data=Conv3, filte=64, rate=(4,4))


    Conv7 = ConvBlock2(data=Conv4, filte=64, rate=(3,3))
    concatenate2 = Concatenate()([Conv7, Conv3])
    Conv7 = ConvBlock2(data=concatenate2, filte=64, rate=(3,3))

    Conv8 = ConvBlock2(data=Conv7, filte=64, rate=(2,2))
    concatenate3 = Concatenate()([Conv8, Conv2])
    Conv8 = ConvBlock2(data=concatenate3, filte=64, rate=(2,2))

    Conv9 = ConvBlock1(data=Conv8, filte=64)
    concatenate4 = Concatenate()([Conv9, Conv1])
    Conv9 = ConvBlock1(data=concatenate4, filte=64)

    outconv = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(Conv9)
    out = Add()([data, outconv])
    out0 = Activation('sigmoid',name='out0')(out)

    return out0, out
    # out1 = Activation('sigmoid', name='out1')(outconv)

    # return out1


def FRNet(data):

    Conv1 = ConvBlock1(filte=64, data=data)

    pool1 = MaxPooling2D(pool_size=(2, 2))(Conv1)
    Conv2 = ConvBlock1(data=pool1, filte=64)
    
    pool2 = MaxPooling2D(pool_size=(2, 2))(Conv2)
    Conv3 = ConvBlock1(data=pool2, filte=128)

    pool3 = MaxPooling2D(pool_size=(2, 2))(Conv3)
    Conv4 = ConvBlock1(data=pool3, filte=128)

    pool4 = MaxPooling2D(pool_size=(2, 2))(Conv4)
    Conv5 = ConvBlock1(data=pool4, filte=256)


    up1 = UpSampling2D((2, 2))(Conv5)
    concatenate1 = Concatenate()([up1, Conv4])
    Conv6 = ConvBlock1(data=concatenate1, filte=128)

    up2 = UpSampling2D((2, 2))(Conv6)
    concatenate2 = Concatenate()([up2, Conv3])
    Conv7 = ConvBlock1(data=concatenate2, filte=128)

    up3 = UpSampling2D((2, 2))(Conv7)
    concatenate3 = Concatenate()([up3, Conv2])
    Conv8 = ConvBlock1(data=concatenate3, filte=64)

    up4 = UpSampling2D((2, 2))(Conv8)
    concatenate4 = Concatenate()([up4, Conv1])
    Conv9 = ConvBlock1(data=concatenate4, filte=64)

    outconv = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(Conv9)
    out1 = Subtract()([data, outconv])
    out1 = Activation('sigmoid',name='out1')(out1)
    return out1