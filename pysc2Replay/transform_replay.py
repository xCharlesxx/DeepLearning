from pysc2.lib import features, point, remote_controller, actions
from absl import app, flags
from pysc2.env.environment import TimeStep, StepType
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
import importlib
import glob
from random import randint
import pickle
from multiprocessing import Process
from tqdm import tqdm
import math
import random
import numpy as np
import multiprocessing
import sys, os
from Constants import const

cpus = multiprocessing.cpu_count()

FLAGS = flags.FLAGS
FLAGS(sys.argv)
flags.DEFINE_string("replays", "D:\Games\StarCraft II\Replays\Test", "Path to the replay files.")
flags.DEFINE_string("agent", "ObserverAgent.ObserverAgent", "Path to an agent.")
flags.DEFINE_integer("procs", cpus, "Number of processes.", lower_bound=1)
flags.DEFINE_integer("start", 0, "Start at replay no.", lower_bound=0)
flags.DEFINE_integer("batch", 1, "Size of replay batch for each process", lower_bound=1, upper_bound=512)

class Parser:
    def __init__(self,
                 replay_file_path,
                 agent,
                 player_id=1,
                 screen_size_px=(84, 84),
                 minimap_size_px=(84, 84),
                 discount=1.):
                 #frames_per_game=1):


        print("Parsing " + replay_file_path)

        self.replay_file_name = replay_file_path.split("/")[-1].split(".")[0]
        self.agent = agent
        self.discount = discount
        #self.frames_per_game = frames_per_game

        self.run_config = run_configs.get()
        self.sc2_proc = self.run_config.start()
        self.controller = self.sc2_proc.controller

        replay_data = self.run_config.replay_data(self.replay_file_name + '.SC2Replay')
        ping = self.controller.ping()
        self.info = self.controller.replay_info(replay_data)
        if not self._valid_replay(self.info, ping):
            raise Exception("{} is not a valid replay file!".format(self.replay_file_name + '.SC2Replay'))

        screen_size_px = point.Point(*screen_size_px)
        minimap_size_px = point.Point(*minimap_size_px)
        interface = sc_pb.InterfaceOptions(
            raw=True, score=True,
            feature_layer=sc_pb.SpatialCameraSetup(width=350,allow_cheating_layers=False,crop_to_playable_area=True),show_cloaked=True, raw_affects_selection=True,raw_crop_to_playable_area=True)
        screen_size_px.assign_to(interface.feature_layer.resolution)
        minimap_size_px.assign_to(interface.feature_layer.minimap_resolution)

        map_data = None
        if self.info.local_map_path:
            map_data = self.run_config.map_data(self.info.local_map_path)

        self._episode_length = self.info.game_duration_loops
        self._episode_steps = 0

        self.controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            map_data=map_data,
            options=interface,
            observed_player_id=player_id))

        self._state = StepType.FIRST

    @staticmethod
    def _valid_replay(info, ping):
        """Make sure the replay isn't corrupt, and is worth looking at."""
        if (info.HasField("error") or
                    info.base_build != ping.base_build or  # different game version
                    info.game_duration_loops < 10 or
                    len(info.player_info) != 2):
            # Probably corrupt, or just not interesting.
            return False
#   for p in info.player_info:
#       if p.player_apm < 10 or p.player_mmr < 1000:
#           # Low APM = player just standing around.
#           # Low MMR = corrupt replay or player who is weak.
#           return False
        return True

    def start(self):
        print("Hello we are in Start")


        step_mul = 50
        _features = features.features_from_game_info(self.controller.game_info())
        
        while True:
            #Takes one step through the replay
            self.controller.step(step_mul)
            #Converts visual data into abstract data
            obs = self.controller.observe()
            #if (len(obs.actions) != 0):
                #print(obs.actions)
            agent_obs = _features.transform_obs(obs)

            if obs.player_result: # Episide over.
                self._state = StepType.LAST
                discount = 0
            else:
                discount = self.discount

            self._episode_steps += step_mul

            step = TimeStep(step_type=self._state, reward=0,
                            discount=discount, observation=agent_obs)

            self.agent.step(step, self.info)
            screenpoint = (84, 84)
            screenpoint = point.Point(*screenpoint)

            #self.controller.actions(sc_pb.RequestAction(actions=[actions.FUNCTIONS.move_camera(screenpoint)]))
            if obs.player_result:
                break

            self._state = StepType.MID

        print("Saving data")
        print(self.info)
        print(self.agent.states)
        pickle.dump({"info" : self.info, "state" : self.agent.states}, open("D:/Charlie/DeepLearning/data/" + "Me" + ".p", "wb"))
        print("Data successfully saved")
        self.agent.states = []
        print("Data flushed")
        #self.controller.
        print("Done")

def parse_replay(replay_batch, agent_module, agent_cls):
    for replay in replay_batch:
        try:
            parser = Parser(replay, agent_cls())
            print("Got to Start")
            parser.start()
        except Exception as e:
            print("Exception was made")
            print(e)

def main(unused):
    agent_module, agent_name = FLAGS.agent.rsplit(".", 1)
    agent_cls = getattr(importlib.import_module(agent_module), agent_name)
    processes = 1#FLAGS.procs
    replay_folder = FLAGS.replays
    batch_size = 1#FLAGS.batch

    truePath = os.path.join(replay_folder, '*.SC2Replay')
    replays = glob.glob(truePath, recursive=True)

    start = FLAGS.start
    #Split replays into batches 
    for i in tqdm(range(math.ceil(len(replays)/processes/batch_size))):
        procs = []
        #Skip to start pos
        x = i * processes * batch_size
        if x < start:
            continue
        #For each processor 
        for p in range(processes):
            #Get num replay start
            xp1 = x + p * batch_size
            #Get num replay end
            xp2 = xp1 + batch_size
            xp2 = min(xp2, len(replays))
            #Give each processor their replays for this batch
            p = Process(target=parse_replay, args=(replays[xp1:xp2], agent_module, agent_cls))
            p.start()
            #Add process to list
            procs.append(p)
            #If at the end of replays, break
            if xp2 == len(replays):
                break
        #Does something? 
        for p in procs:
            p.join()


if __name__ == "__main__":
    app.run(main)