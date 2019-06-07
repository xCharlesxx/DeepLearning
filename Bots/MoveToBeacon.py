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
from pprint import pprint

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units


from absl import app

class MoveToBeacon(base_agent.BaseAgent):
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

    def step(self, obs):
        super(MoveToBeacon, self).step(obs)

        
        #input = self.get_obs(obs) 7x64x64
        input = obs.observation.feature_minimap
        inputx = tf.keras.utils.normalize(input[1])
        inputy = tf.keras.utils.normalize(input[2])
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        #model.fit(inputx, inputy, epochs=3)
        outputx = 0 
        outputy = 0
        #Reward = objectives completed since last step
        reward = obs.reward

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
        if self.unit_type_is_selected(obs, units.Terran.Marine):
            if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                return actions.FUNCTIONS.Attack_minimap("now", (outputx,outputy))
        #Select Marine
        else: 
            marine = self.get_units_by_type(obs, units.Terran.Marine)
            if len(marine) > 0: 
                return actions.FUNCTIONS.select_point('select_all_type', (marine[0].x,
                                                                    marine[0].y))

        #features.MinimapFeatures.
        return actions.FUNCTIONS.no_op()