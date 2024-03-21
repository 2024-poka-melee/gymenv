from abc import ABC, abstractmethod
from melee import enums
import numpy as np
from melee_env.agents.util import *
import code
import time

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

class AgentChooseCharacter(Agent):
    def __init__(self, character):
        super().__init__()
        self.character = character
    

class Random(AgentChooseCharacter):
    def __init__(self, character):
        super().__init__(character)
        self.action_space = ActionSpace()
    
    def act(self, observation):
        action = self.action_space.sample()
        return action


class Rest(Agent):
    # adapted from AltF4's tutorial video: https://www.youtube.com/watch?v=1R723AS1P-0
    # This agent will target the nearest player, move to them, and rest
    def __init__(self):
        super().__init__()
        self.character = enums.Character.JIGGLYPUFF

        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace()
        self.action = 0
    
    def act(self, observation):
        curr_position = observation[self.port-1, :2]
        try:
            positions_centered = observation[:, :2] - curr_position
        except:
            code.interact(local=locals())

        # distance formula
        distances = np.sqrt(np.sum(positions_centered**2, axis=1))
        closest_sort = np.argsort(np.sqrt(np.sum(positions_centered**2, axis=1)))  

        actions = observation[:, 2]
        actions_by_closest = actions[closest_sort]

        # select closest player who isn't dead
        closest = 0
        for i in range(len(observation)):
            if actions_by_closest[i] >= 14 and i != 0:
                closest = closest_sort[i]
                break

        if closest == self.port-1:  # nothing to target
            action = 0

        elif distances[closest] < 4:
            action = 23  # Rest

        else:  
            if np.abs(positions_centered[closest, 0]) < np.abs(positions_centered[closest, 1]):
                # closer in X than in Y - prefer jump
            
                if observation[closest, 1] > curr_position[1]:
                    if self.action == 1:
                        action = 0  # re-input jump
                    else:
                        action = 1
                else:
                    if self.action == 5:
                        action = 0
                    else:
                        action = 5  # reinput down to fall through platforms
            else:
                # closer in Y than in X - prefer run/drift
                if observation[closest, 0] < curr_position[0]:
                    action = 7  # move left
                else:
                    action = 3  # move right
                    
        self.action = action
        return self.action
