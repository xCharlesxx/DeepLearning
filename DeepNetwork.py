from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

import numpy as np 
import os 
import random

#Keras
#Shape = Shape of input data
#Dropout = Fraction rate of input inits to 0 at each update during training time, which prevents overfitting (0-1)
def build_knet(shapex, shapey, dropout):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(shapex, shapey),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(dropout + 0.3))

    #Output Layer
    model.add(Dense(2, activation='softmax'))

    learning_rate = 1e-4
    decay = 1e-6
    opt = keras.optimizers.adam(lr=learning_rate, decay=decay)

    model.compile(loss='categorical_crossentropy', 
                  optimizer=opt, 
                  metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir='logs/stage1')

    training_data_dir = "training_data"


    return 0

#Tensorflow
def build_net(input, info, num_action):
    mconv1 = layers.conv2d(tf.transpose(input, [0, 2, 3, 1]),
                            num_outputs=16,
                            kernel_size=8,
                            stride=4,
                            scope='mconv1')
    mconv2 = layers.conv2d(mconv1,
                            num_outputs=32,
                            kernel_size=4,
                            stride=2,
                            scope='mconv2')
    info_fc = layers.fully_connected(layers.flatten(info),
                                    num_outputs=256,
                                    activation_fn=tf.tanh,
                                    scope='info_fc')

    # Compute spatial actions, non spatial actions and value
    feat_fc = tf.concat([layers.flatten(mconv2), info_fc], axis=1)
    feat_fc = layers.fully_connected(feat_fc,
                                    num_outputs=256,
                                    activation_fn=tf.nn.relu,
                                    scope='feat_fc')

    return spatial_action, non_spatial_action, value


