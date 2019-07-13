from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as layers

import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.callbacks import TensorBoard
from keras_transformer import get_model

import numpy as np 
import os 
import csv
import ast
import re
import random
from decimal import Decimal
from Constants import const


def get_training_data(training_data_dir):
    all_files = os.listdir(training_data_dir)
    all_files_size = len([num for num in all_files])
    inputs=[]
    outputs=[]
    counter = 0
    print('Extracting files...')
    for file in all_files:
        print("{}/{}".format(counter+1,all_files_size), end='\r')
        counter+=1
        full_path = os.path.join(training_data_dir,file)
        # Extract file code
        with open (full_path) as csv_file:
            reader = csv.reader(csv_file)
            count = 0
            inputrows =[]
            for row in reader:
                if count == const.InputSize():
                    outputs.append(row)
                else:
                    inputrows.append(row)
                count+=1
            inputs.append(inputrows)

    print("{}/{}".format(counter,all_files_size))
    inputs = np.expand_dims(inputs, axis=3)

    inputs = np.reshape(inputs, (-1,const.InputSize(),const.InputSize(),1))
    outputs = np.reshape(outputs, (-1,const.OutputSize()))

    inputs = inputs.astype(np.int)
    outputs = outputs.astype(np.float)

    return [inputs,outputs]


#Keras
#Shape = Shape of input data
#Dropout = Fraction rate of input inits to 0 at each update during training time, which prevents overfitting (0-1)
def build_knet():

    dropout = 0.2
    learning_rate = 1e-4
    decay = 1e-6
    padding = 'same'
    loss_function = 'mean_squared_error'
    metrics = 'accuracy'
    epochs = 10
    batch_size = 2
    verbose = 1
    #Percent of data to be split for validation
    validation = 0.1
    
    training_data_dir = "training_data"
    tensorboard = TensorBoard(log_dir='logs/stage1')
    activation = 'tanh'

    model = Sequential()
    #model.add(Conv2D(32, (3, 3), padding=padding,
    #                 input_shape=(const.InputSize(), const.InputSize(), 1),
    #                 activation=activation))
    #model.add(Conv2D(32, (3, 3), activation=activation))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    ##model.add(Dropout(dropout))

    #model.add(Conv2D(64, (3, 3), padding=padding,
    #                 activation=activation))
    #model.add(Conv2D(64, (3, 3), activation=activation))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    ##model.add(Dropout(dropout))

    #model.add(Conv2D(128, (3, 3), padding=padding,
    #                 activation=activation))
    #model.add(Conv2D(128, (3, 3), activation=activation))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(dropout))


    #model.add(Dense(1280, activation=activation))
    model.add(Flatten())
    model.add(Dense(512, activation=activation))
   # model.add(Dense(320, activation=activation))
    model.add(Dense(160, activation=activation))
   # model.add(Dense(80, activation=activation))
    model.add(Dense(40, activation=activation))
   # model.add(Dense(20, activation=activation))
    model.add(Dense(10, activation=activation))
  #  model.add(Dense(5, activation=activation))
    #model.add(Dropout(dropout + 0.3))

    #Output Layer
    model.add(Dense(2, activation=activation))

    
    opt = ks.optimizers.adam(lr=learning_rate, decay=decay)

    model.compile(loss=loss_function, 
                  optimizer=opt, 
                  metrics=[metrics])

    file_size = len([num for num in os.listdir(training_data_dir)])
    TD = get_training_data(training_data_dir)
    #for sets in range(0,50):
        #print(inputs[sets])
        #print(outputs[sets])

    #xtrain = np.reshape(xtrain, (-1,const.InputSize(),const.InputSize(),1))

    model.fit(TD[0], TD[1], 
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1, 
                shuffle=False, verbose=verbose)
    

    #prediction = model.predict(TD[0])

    #for loop in range(0,150):
    #    dec1 = np.around(prediction[loop][0],2)
    #    dec2 = np.around(prediction[loop][1],2)
    #    print(str([dec1,dec2]) + '      ' + str(TD[1][loop]))

    model.save("{}-{}-epochs-{}-batches-{}-dataSetSize".format(loss_function, epochs, batch_size, file_size))
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


def build_transformer():

    td = get_training_data("training_data")
    model = get_model(
    token_num=(const.inputsize(), const.inputsize()),
    embed_dim=1,
    encoder_num=3,
    decoder_num=2,
    head_num=3,
    hidden_dim=120,
    attention_activation='relu',
    feed_forward_activation='relu',
    dropout_rate=0.05,
    embed_weights=np.random.random((13, 30))
    )
    
    
    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy')
    model.summary()

     #train the model
    model.fit(
        x=[np.asarray(encoder_inputs * 1000), np.asarray(decoder_inputs * 1000)],
        y=np.asarray(decoder_outputs * 1000),
        epochs=5)

    return 0

