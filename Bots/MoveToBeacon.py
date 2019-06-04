from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pysc2
import numpy
from pprint import pprint

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units


from absl import app

class MoveToBeacon(base_agent.BaseAgent):

    def get_obs(self, obs):
        return {self.screen: obs['screen'],
                self.available_actions: obs['available_actions']}

    def step(self, obs):
        super(MoveToBeacon, self).step(obs)

        
        #input = self.get_obs(obs)
        input = obs.observation.feature_minimap
        pprint(inputlist)


        #choice = numpy.random.choice(obs.observation.available_actions) 
        #args = [[numpy.random.randint(0, size) for size in arg.sizes]
        #        for arg in self.action_spec.functions[choice].args]
        #return actions.FunctionCall(choice, args)
        #input = obs.observation.
        #return actions.FUNCTIONS_AVAILABLE[numpy.random.randint(0, len(actions.FUNCTIONS_AVAILABLE))]

   # function_id = numpy.random.choice(obs.observation.available_actions)
   # args = [[numpy.random.randint(0, size) for size in arg.sizes]
   #         for arg in self.action_spec.functions[function_id].args]
   # return actions.FunctionCall(function_id, args)


        #Select Marine
        marine = [unit for unit in obs.observation.feature_units
                if unit.unit_type == units.Terran.Marine]

        if len(marine) > 0: 
          return actions.FUNCTIONS.select_point('select_all_type', (marine[0].x,
                                                                    marine[0].y))

        #features.MinimapFeatures.
        return actions.FUNCTIONS.no_op()