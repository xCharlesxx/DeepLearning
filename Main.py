from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow 
import pysc2
import numpy

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features


from absl import app

class Overmind(base_agent.BaseAgent):
     def step(self, obs):
         super(Overmind, self).step(obs)



         return actions.FUNCTIONS.no_op()


def main(unused_argv):
    #Agent
    agent = Overmind()
    #Map name
    map = 'AbyssalReef' 
    #Steps default is 8 per frame
    steps = 16
    #visualize map 
    vis = False
    try: 
        while True:
            with sc2_env.SC2Env(False,
                map_name = map,
                players=[sc2_env.Agent(sc2_env.Race.zerg), 
                         sc2_env.Bot(sc2_env.Race.zerg, 
                                     sc2_env.Difficulty.very_easy)], 
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64), 
                    use_feature_units=True),
                step_mul=steps,
                game_steps_per_episode=0,
                visualize=vis) as env:

             agent.setup(env.observation_spec(), env.action_spec())

             timesteps = env.reset()
             agent.reset()

             while True:
                 step_actions = [agent.step(timesteps[0])]
                 if timesteps[0].last():
                     break
                 timesteps = env.step(step_actions)

    except KeyboardInterrupt: 
      pass

if __name__ == "__main__":
      app.run(main)