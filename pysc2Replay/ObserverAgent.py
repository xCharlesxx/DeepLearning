#!/usr/bin/env python

import pickle
import numpy as np
import math

from pysc2.lib import actions as sc_action
from pysc2.lib import static_data
from pysc2.agents import base_agent
from Constants import const

class ObserverAgent(base_agent.BaseAgent):
    #Camera offset
    cy_cx = [0,0]
    unit_dict = {}
    def __init__(self):
        self.states = []
        self.count = 0;
        self.unit_dict = {
                #select point
                '2': self.select_point,
                #select rect 
                '3': self.double_select_point,
                #smart minimap 
                '452': self.single_select_point,
                #attack minimap 
                '13': self.single_select_point, 
                #smart screen 
                '451': self.single_select_point,
                #attack screen 
                '12': self.single_select_point, 
                '14': self.single_select_point,
                #Inject larve
                '204': self.single_select_point,
                #Burrow down
                '103': self.single_q,
                #Burrow up
                '117': self.single_q,
                #Blinding cloud 
                '179': self.single_select_point,
                #Caustic spray 
                '184': self.single_select_point,
                #Contaminate 
                '188': self.single_select_point,
                #Corrosive bile 
                '189': self.single_select_point,
                #Explode 
                '191': self.single_q,
                #fungal growth 
                '194': self.single_select_point,
                #infested terrans 
                '203': self.single_select_point,
                #transfuse 
                '242': self.single_select_point,
                #neural parasite 
                '212': self.single_select_point,
                #parasidic bomb 
                '215': self.single_select_point,
                #spawn changeling
                '228': self.single_q,
                #spawn locusts
                '229': self.single_select_point,
                #viper consume 
                '243': self.single_select_point,
                #hold position quick 
                '274': self.single_q, 
                #stop quick
                '453': self.single_q,
                #select army 
                '7': self.single_q
                    }

    def calculate_offset(self, height_map): 
        for numy, y in enumerate(height_map):
            for numx, x in enumerate(y): 
                if (x != 0):
                    self.cy_cx = [numy, numx]
                    return

    def stencil(self, _stencil, _raw_list, _new_width, _new_height):
        input = _raw_list
        stencil = _stencil
        newInput = np.zeros((_new_height,_new_width),int)
        counterx = 0
        countery = 0
        for numy, y in enumerate(stencil):
            for numx, x in enumerate(y): 
                if (x != 0):
                    newInput[countery][counterx] = input[numy][numx]
                    counterx+=1
                if (counterx == _new_width):
                    countery+=1
                    counterx=0
        return newInput

    def select_point(self, args):
        string = "[[" + str(format(args[0][0])) + "], "
        string += "[" + str(args[1][0]) + ", "
        string += str(args[1][1]) + "]]"
        return string
    def single_select_point(self, args):
        string = "[[" + '0' + "], "
        string += "[" + str(args[1][0]) + ", "
        string += str(args[1][1]) + "]]"
        return string
    def double_select_point(self, args):
        string = "[[" + '0' + "], "
        string += "[" + str(args[1][0]) + ", "
        string += str(args[1][1]) + "], "
        string += "[" + str(args[1][0]) + ", "
        string += str(args[1][1]) + "]]"
        return string
    def single_q(self, args):
        return "[0]"
    def default(self, args):
        return "Unknown"

    def extract_args(self, id, args):
        if not args: 
            return "[]"

        #Burrow down for various units translate to quick burrow up
        if (int(id) > 103 and int(id) < 117): 
            id = '103'

        #Same for burrow up
        if (int(id) > 117 and int(id) < 140):
            id = '117'

        func = self.unit_dict.get(id, self.default)
        return func(args)


    def step(self, time_step, info, acts):

        #print(self.count, end='\r')
        #self.count += 1
        state = {}
        for action in acts:
            state["Actions:"] += self.extract_args(format(action.function), action.arguments)
            break
        



        #state["minimap"] = [
        #    time_step.observation["feature_minimap"][0] / 255,                  # height_map
        #    time_step.observation["feature_minimap"][1] / 2,                    # visibility
        #    time_step.observation["feature_minimap"][2],                        # creep
        #    time_step.observation["feature_minimap"][3],                        # camera
        #    (time_step.observation["feature_minimap"][5] == 1).astype(int),     # own_units
        #    (time_step.observation["feature_minimap"][5] == 3).astype(int),     # neutral_units
        #    (time_step.observation["feature_minimap"][5] == 4).astype(int),     # enemy_units
        #    time_step.observation["feature_minimap"][6]                         # selected
        #]
        #height_map = time_step.observation.feature_screen[0]
        ##self.calculate_offset(height_map)
        ##Remove all Zero lines 
        #height_map = height_map[~np.all(height_map == 0, axis=1)]
        ##Remove all Zeros in remaining lines
        #height_map = [x[x != 0] for x in height_map]

        #height = 0
        #width = 0
        #for x in height_map:
        #    output = ""
        #    width = 0
        #    for i in x:
        #        output+=str(i)
        #        output+=""
        #        width += 1
        #    #print(output)
        #    height += 1
        ##print("\n")   

        #print("width: ")
        #print(width)
        #print("height: ")
        #print(height)
        #print("\n")
        #return 
        #unit_type = self.stencil(time_step.observation.feature_screen[0], time_step.observation.feature_screen[6], width, height)  
     
        #height = 0
        #width = 0
        #for x in unit_type:
        #    output = ""
        #    width = 0
        #    for i in x:
        #        output+=str(i)
        #        output+=" "
        #        width += 1
        #    #print(output)
        #    height += 1
        ##print("\n") 
        
        #print("width unit: ")
        #print(width)
        #print("height unit: ")
        #print(height)
        #print("\n")  

        #unit_type_compressed = np.zeros(unit_type.shape, dtype=np.float)
        #for y in range(len(unit_type)):
        #    for x in range(len(unit_type[y])):
        #        if unit_type[y][x] > 0 and unit_type[y][x] in static_data.UNIT_TYPES:
        #            unit_type_compressed[y][x] = static_data.UNIT_TYPES.index(unit_type[y][x]) / len(static_data.UNIT_TYPES)

        #for x in unit_type_compressed:
        #    output = ""
        #    for i in x:
        #        output+=str(i)
        #        output+=""
        #    print(output)
        #print("\n")  
        #hit_points = time_step.observation["feature_screen"][8]
        #hit_points_logged = np.zeros(hit_points.shape, dtype=np.float)
        #for y in range(len(hit_points)):
        #    for x in range(len(hit_points[y])):
        #        if hit_points[y][x] > 0:
        #            hit_points_logged[y][x] = math.log(hit_points[y][x]) / 4

        #state["screen"] = [
        #    time_step.observation["feature_screen"][0] / 255,               # height_map
        #    time_step.observation["feature_screen"][1] / 2,                 # visibility
        #    time_step.observation["feature_screen"][2],                     # creep
        #    time_step.observation["feature_screen"][3],                     # power
        #    (time_step.observation["feature_screen"][5] == 1).astype(int),  # own_units
        #    (time_step.observation["feature_screen"][5] == 3).astype(int),  # neutral_units
        #    (time_step.observation["feature_screen"][5] == 4).astype(int),  # enemy_units
        #    unit_type_compressed,                                   # unit_type
        #    time_step.observation["feature_screen"][7],                     # selected
        #    hit_points_logged,                                      # hit_points
        #    time_step.observation["feature_screen"][9] / 255,               # energy
        #    time_step.observation["feature_screen"][10] / 255,              # shields
        #    time_step.observation["feature_screen"][11]                     # unit_density
        #]



        ## Binary encoding of available actions
        #'''
        #state["game_loop"] = time_step.observation["game_loop"]
        #state["player"] = time_step.observation["player"]
        
        #state["available_actions"] = np.zeros(len(sc_action.FUNCTIONS))
        #for i in time_step.observation["available_actions"]:
        #    state["available_actions"][i] = 1.0
        #'''

        self.states.append(state)


class NothingAgent(base_agent.BaseAgent):
    def step(self, obs):
        #print(obs.observation.available_actions)
        if (1 in obs.observation.available_actions):
            return sc_action.FUNCTIONS.move_camera([const.MiniMapSize()/2,const.MiniMapSize()/2])
        return sc_action.FUNCTIONS.no_op()