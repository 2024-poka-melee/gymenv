from melee import enums
#from melee_env.env import MeleeEnv
from melee_env.myenv import MeleeEnv
#from melee_env.agents.basic import *
from melee_env.agents.mybasic import *
import argparse
from DQNAgent import DQNAgent
from melee_env.agents.util import *

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import argparse

parser = argparse.ArgumentParser(description="Example melee-env demonstration.")
parser.add_argument("--iso", default="/home/vlab/SSBM/ssbm.iso", type=str, 
    help="Path to your NTSC 1.02/PAL SSBM Melee ISO")
args = parser.parse_args()

players = [Rest(), Random(enums.Character.FOX)] #CPU(enums.Character.KIRBY, 5)]

episodes = 1; reward = 0

env = MeleeEnv(args.iso, players, fast_forward=True, ai_starts_game=True)
env.start()

for episode in range(episodes):
    # gamestate, done = env.setup(enums.Stage.BATTLEFIELD)
    obs, done = env.reset(enums.Stage.BATTLEFIELD)
    while not done:
        action = [0, 0]
        for i in range(len(players)):
            action[i] = players[i].act(obs)
        obs, reward, done, info = env.step(*action)
    env.close()