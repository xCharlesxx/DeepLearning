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
from decimal import Decimal
from Constants import const
from absl import app

class DefeatEnemies(base_agent.BaseAgent):

    def step(self, obs):
        super(MoveToBeacon, self).step(obs)

        return actions.FUNCTIONS.no_op()
