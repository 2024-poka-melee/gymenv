from abc import ABC, abstractmethod
from melee import enums
import numpy as np
from melee_env.agents.util import *
import code

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import namedtuple, deque


class Agent(ABC):
    def __init__(self):
        self.agent_type = "AI"
        self.controller = None
        self.port = None  # this is also in controller, maybe redundant?
        self.action = 0
        self.press_start = False
        self.self_observation = None
        self.current_frame = 0

    @abstractmethod
    def act(self):
        pass

class DQN(nn.Module):
    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_obs, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, n_actions)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        data = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*data)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class DQNAgent(Agent):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = 45
        self.n_obs = 12
        
        self.policy_net = DQN(self.n_obs, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_obs, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.character = enums.Character.FOX
        
        self.BATCH_SIZE = 512
        self.GAMMA = 0.9
        self.EPS_START = 0.1 #0.9
        # self.EPS_END = 0.05
        #self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace()
        self.action = 0
        self.steps_done = 0
        self.replay_buffer = ReplayMemory(100000)
    
    def convert_state(self, state):
        # state, done = env.step()
        obs, reward, done, info = self.observation_space(state)
        obs = torch.tensor(obs, device=self.device)
        obs = torch.reshape(obs, (1, 12))
        return obs
    
    def select_action(self, gamestate):
        sample = random.random()
        eps_threshold = self.EPS_START
        #eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            state = self.convert_state(gamestate)
            with torch.no_grad():
                # policy_net(staet).max(1): torch.return_types.max(values=tensor([0.0286], grad_fn=<MaxBackward0>),indices=tensor([0]))
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.action_space.sample()]], device=self.device, dtype=torch.long)
    
    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = map(torch.cat, self.replay_buffer.sample(self.BATCH_SIZE))
        
        Q = self.policy_net(states).gather(1, actions)
        nextQ = self.target_net(next_states).max(1)[0] # 행벡터
        
        target = rewards + (1 - dones) * nextQ * self.GAMMA
        target = target.reshape(self.BATCH_SIZE, 1)
        
        #loss = nn.SmoothL1Loss(Q, target)
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q.to(torch.float32), target.to(torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        
        """for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)"""
        if self.steps_done % 1000 == 0:
            # print("sync!!") 
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]
            self.target_net.load_state_dict(target_net_state_dict)
        
    @from_action_space       # translate the action from action_space to controller input
    def act(self, state):
        obs = self.convert_state(state)
        with torch.no_grad():
            q_values = self.policy_net(obs)
            self.action = q_values.argmax(dim=1, keepdim=True)
        return self.action
    
    @from_action_space
    def apply(self, action):
        self.action = action
        return self.action