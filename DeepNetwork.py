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
import csv
import ast
import re
import random

#Keras
#Shape = Shape of input data
#Dropout = Fraction rate of input inits to 0 at each update during training time, which prevents overfitting (0-1)
def build_knet(shapex, shapey, dropout):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(24, 24, 1),
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

    training_data_dir = "training_data1"
    epochs = 30



    all_files = os.listdir(training_data_dir)
    all_files_size = len([num for num in all_files])
    inputs=[]
    outputs=[]
    counter = 0
    print('Extracting files...')
    for file in all_files:
        print("{}/{}".format(counter,all_files_size), end='\r')
        counter+=1
        full_path = os.path.join(training_data_dir,file)
        # Extract file code
        with open (full_path) as csv_file:
            reader = csv.reader(csv_file)
            count = 0
            for row in reader:
                if count == 0:
                    inputrow =[]
                    for rowseg in row:
                        inputrow.append(ast.literal_eval(re.sub(r'(\d?)(?:\r\n)*\s+', r'\1, ', rowseg)))
                    inputs.append(inputrow)
                if count == 2:
                    outputs.append(row)
                count+=1

    inputs = np.reshape(inputs, (-1,24,24))
    batch_size = 10
    inputs = np.expand_dims(inputs, axis=3)
    #xtrain = inputs ytrain = outputs
    xtest = np.array(inputs[:int(len(inputs)*0.1)])
    ytest = np.array(outputs[:int(len(outputs)*0.1)])

    xtrain = np.array(inputs[int(len(inputs)*0.1):])
    ytrain = np.array(outputs[int(len(outputs)*0.1):])

    #xtrain = np.reshape(xtrain, (-1,24,24,1))

    model.fit(xtrain, ytrain, 
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(xtest,ytest), 
                shuffle=False, verbose=1, callbacks=[tensorboard])

    model.save("MoveToBeaconCNN-{}-epochs-{}-batches-{}-dataSetSize".format(epochs, batch_size, all_files_size))
    print('Finished Training')
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


