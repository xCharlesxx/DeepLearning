from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pysc2
import numpy
import cv2
import pandas
from pprint import pprint

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

import csv
import random

from absl import app

class MoveToBeacon(base_agent.BaseAgent):
    loaded = False
        #Pysc2 defs
    def get_obs(self, obs):
        return {self.screen: obs['screen'],
                self.available_actions: obs['available_actions']}

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
         obs.observation.single_select[0].unit_type == unit_type):
          return True
        
      #Tensorflow defs
    def build():
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        self.saver = tf.train.Saver(variables, max_to_keep=100)
        self.init_op = tf.variables_initializer(variables)
        train_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
        self.train_summary_op = tf.summary.merge(train_summaries)

    def save(self, path, step=None):
        os.makedirs(path, exist_ok=True)
        step = step or self.train_step
        print("Save agent to %s, step %d" % (path, step))
        ckpt_path = os.path.join(path, 'model.ckpt')
        self.saver.save(self.sess, ckpt_path, global_step=step)

    def load(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        self.train_step = int(ckpt.model_checkpoint_path.split('-')[-1])
        print("Load agent at step %d" % self.train_step)

    def loadK(self, path):
        self.model = ks.models.load_model(path)

    def step(self, obs):
        super(MoveToBeacon, self).step(obs)

        if MoveToBeacon.loaded == False:
            self.loadK("MoveToBeaconCNN-10-epochs-Attempt2")
            MoveToBeacon.loaded = True

        #If maring is selected, use DNN
        if self.unit_type_is_selected(obs, units.Terran.Marine):
            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
                input = obs.observation.feature_minimap[5]
                stencil = obs.observation.feature_minimap[3]
                newInput = numpy.zeros((24,24),int)
                counterx = 0
                countery = 0
                for numx, x in enumerate(stencil):
                    for numy, y in enumerate(x): 
                        if (y == 1):
                            newInput[counterx][countery] = input[numx][numy]
                            counterx+=1
                        if (counterx == 24):
                            countery+=1
                            counterx=0

                for x in newInput:
                        output = ""
                        for i in x:
                            output+=str(i)
                            output+=" "
                        print(output)
                print("\n")

                newInput = numpy.expand_dims(newInput, axis=2)
                prediction = self.model.predict([newInput.reshape([-1,24,24,1])])
                outputx = prediction[0][0] * 80
                outputy = prediction[0][1] * 80
                print('Network Predicts: {},{}'.format(outputx,outputy))
                return actions.FUNCTIONS.Attack_screen("now", (outputx,outputy))
        #Select Marine
        else: 
            marine = self.get_units_by_type(obs, units.Terran.Marine)
            if len(marine) > 0: 
                return actions.FUNCTIONS.select_point('select_all_type', (marine[0].x,
                                                                    marine[0].y))

        #features.MinimapFeatures.
        return actions.FUNCTIONS.no_op()

class GenerateMoveToBeaconTestData(base_agent.BaseAgent):
        #Pysc2 defs
    packagedInput = numpy.empty((24*24), int)
    packagedOutput = numpy.empty(2, float)
    packageCounter = 1017
    def get_obs(self, obs):
        return {self.screen: obs['screen'],
                self.available_actions: obs['available_actions']}

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
         obs.observation.single_select[0].unit_type == unit_type):
          return True

    def step(self, obs):
        super(GenerateMoveToBeaconTestData, self).step(obs)


       # Extract file code
       # with open ('training_data/0.csv') as csv_file:
       #     reader = csv.reader(csv_file)
       #     count = 0
       #     for row in reader:
       #         if count == 0:
       #             GenerateMoveToBeaconTestData.packagedInput = row
       #         if count == 2:
       #             GenerateMoveToBeaconTestData.packagedOutput = row
       #         count+=1


        #If previous action was successful, record as training data
        if obs.reward > 0:
            #counter = 0
            #output = ""
            #for x in GenerateMoveToBeaconTestData.packagedInput:
            #    output+=str(x)
            #    output+=" "
            #    counter+=1 
            #    if (counter == 24):
            #        print(output)
            #        output = ""
            #        counter = 0
            #print(GenerateMoveToBeaconTestData.packagedOutput)
            fileName = 'training_data/' + str(GenerateMoveToBeaconTestData.packageCounter) + '.csv'
            with open(fileName, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(GenerateMoveToBeaconTestData.packagedInput)
                writer.writerow(GenerateMoveToBeaconTestData.packagedOutput)
            GenerateMoveToBeaconTestData.packageCounter+=1


        input = obs.observation.feature_minimap[5]
        stencil = obs.observation.feature_minimap[3]
        #24x24 is refined input data size
        newInput = [0] * (24*24)
        counter = 0
        #Use camera stencil to grab relevent data
        for numx, x in enumerate(stencil):
            for numy, y in enumerate(x): 
                if (y == 1):
                    newInput[counter] = input[numx][numy]
                    counter+=1


        #Screen is not 80x80 but ~80x60 but 80x80 for simplicity
        outputx = random.randint(0,80)
        outputy = random.randint(0,80)

        GenerateMoveToBeaconTestData.packagedInput = newInput
        #/80 to get a number between 0 and 1 as outputs for DNN
        GenerateMoveToBeaconTestData.packagedOutput = [outputx/80, outputy/80]
        if self.unit_type_is_selected(obs, units.Terran.Marine):
            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
                return actions.FUNCTIONS.Attack_screen("now", (outputx,outputy))
        #Select Marine
        else: 
            marine = self.get_units_by_type(obs, units.Terran.Marine)
            if len(marine) > 0: 
                return actions.FUNCTIONS.select_point('select_all_type', (marine[0].x,
                                                                    marine[0].y))


        #features.MinimapFeatures.
        return actions.FUNCTIONS.no_op()