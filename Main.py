from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import keras as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pysc2
import numpy
import Bots

from Bots.MoveToBeacon import MoveToBeacon, GenerateMoveToBeaconTestData
from Bots.Overmind.Overmindx00 import Overmindx00
from DeepNetwork import build_knet, build_transformer
from Bots.DefeatEnemies import RandomAgent, DefeatEnemies
from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features, units

from absl import app

def main(unused_argv):
    #build_knet()
    #build_transformer()

    #Agent
    agent = RandomAgent()
    try: 
        while True:
            with sc2_env.SC2Env(False,
                map_name = 'DefeatZerglingsAndBanelings',
                players= [
                        sc2_env.Agent(sc2_env.Race.zerg)#,
                        #sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.very_easy)
                         ], 
                agent_interface_format=features.AgentInterfaceFormat(
                    #What resolution the player sees the world at 
                    feature_dimensions=features.Dimensions(screen=84, minimap=84),
                    #More indepth unit information
                    use_feature_units=True),
                #Steps default is 8 per frame (168APM)
                step_mul=1,#175
                #Max steps per game (0 is infinite)
                game_steps_per_episode=0,
                #visualize pysc2 input layers 
                visualize=False, 
                #Play-back-time
                realtime=True, 
                #Fog of War
                disable_fog=False
           ) as env:
                run_loop.run_loop([agent], env)
    except KeyboardInterrupt:
        pass



if __name__ == "__main__":
      app.run(main)