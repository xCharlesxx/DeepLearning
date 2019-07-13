from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras as ks
import numpy as np

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
from decimal import Decimal
from Constants import const
from absl import app

class DefeatEnemies(base_agent.BaseAgent):
    loaded = False
    def get_obs(self, obs):
        return {self.screen: obs['screen'],
                self.available_actions: obs['available_actions']}

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
            if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.multi_select) > 0): 
            if (obs.observation.multi_select[0].unit_type == unit_type):
                return True
        elif (len(obs.observation.single_select) > 0):
                if (obs.observation.single_select[0].unit_type == unit_type):
                    return True
    def loadK(self, path):
        self.model = ks.models.load_model(path)

    def step(self, obs):
        super(DefeatEnemies, self).step(obs)

        if DefeatEnemies.loaded == False:
            self.loadK("8Dense-10-epochs-2-batches-3000-dataSetSize-98%")
            DefeatEnemies.loaded = True


        if self.unit_type_is_selected(obs, units.Terran.Marine):
            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):


                #84x84 Detailed
                #input = obs.observation.feature_screen[6]

                #31x31 Simplified
                input = obs.observation.feature_minimap[5]
                stencil = obs.observation.feature_minimap[3]
                newInput = numpy.zeros((const.InputSize(),const.InputSize()),int)
                counterx = 0
                countery = 0
                for numy, y in enumerate(stencil):
                    for numx, x in enumerate(y): 
                        if (x == 1):
                            newInput[countery][counterx] = input[numy][numx]
                            counterx+=1
                        if (counterx == const.InputSize()):
                            countery+=1
                            counterx=0

                #for x in obs.observation.feature_screen[6]:
                #        output = ""
                #        for i in x:
                #            output+=str(i)
                #            output+=""
                #        print(output)
                #print("\n")

                newInput = numpy.expand_dims(newInput, axis=2)
                newInput = newInput.reshape([-1,const.InputSize(),const.InputSize(),1])


                self.model.fit(TD[0], TD[1], 
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1, 
                shuffle=False, verbose=verbose)
                #for marines in obs.observation.multi_select:
                #    obs.observation.multi_select[0]
                #    prediction = self.model.predict(newInput)


                outputx = prediction[0][0] * const.ScreenSize()
                outputy = prediction[0][1] * const.ScreenSize()
                return actions.FUNCTIONS.Attack_screen("now", (outputx,outputy))

        #Select Marine
        else: 
            marine = self.get_units_by_type(obs, units.Terran.Marine)
            if len(marine) > 0: 
                return actions.FUNCTIONS.select_point('select_all_type', (marine[0].x,
                                                                    marine[0].y))

        return actions.FUNCTIONS.no_op()


class RandomAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""
  inputActionPairs = np.empty([1])
  def getSimplifiedInput(self, obs):
    #31x31 Simplified
    input = obs.observation.feature_minimap[5]
    stencil = obs.observation.feature_minimap[3]
    newInput = numpy.zeros((const.InputSize(),const.InputSize()),int)
    counterx = 0
    countery = 0
    for numy, y in enumerate(stencil):
        for numx, x in enumerate(y): 
            if (x == 1):
                newInput[countery][counterx] = input[numy][numx]
                counterx+=1
            if (counterx == const.InputSize()):
                countery+=1
                counterx=0
    return newInput
    

  def step(self, obs):
    super(RandomAgent, self).step(obs)

    input = self.getSimplifiedInput(obs)
    for x in input:
            output = ""
            for i in x:
                output+=str(i)
                output+=""
            print(output)
    print("\n")



    while True:
        function_id = numpy.random.choice(obs.observation.available_actions)

        args = [[numpy.random.randint(0, size) for size in arg.sizes] 
                for arg in self.action_spec.functions[function_id].args]

        print(function_id)
        print(args)
        if ((str)(np.array(args).shape) == "(2,)"):
            break

    return actions.FunctionCall(function_id, args)
