import numpy as np
import random 

class GridWorld:

    def __init__(self, grid_size, state_reward, jump):
        # grid_size = (height, width)
        # state_reward = {state: reward} take any action in this state will get a fix reward immediately
        # state : tuple(i,j)
        # jump: {state:state}
        self.height = grid_size[0]
        self.width = grid_size[1]
        self.all_states = [(i,j) for i in range(self.height) for j in range(self.width)]
        self.reward = np.zeros(grid_size)
        self.jump = jump 
        for i,j in state_reward.items():
            self.reward[i] = j
    
    
    def get_next_state(self, state, action):
        if state in self.jump.keys():
            return self.jump[state]
        if action == 0:
            if state[0] == 0:
                return state
            state = tuple([state[0]-1,state[1]])
        if action == 1:
            if state[1]+1 == self.width:
                return state
            state = tuple([state[0],state[1]+1])
        if action == 2:
            if state[0]+1 == self.height:
                return state
            state = tuple([state[0]+1,state[1]])
        if action == 3:
            if state[1] == 0:
                return state
            state = tuple([state[0],state[1]-1])
        return state


    def get_reward(self, state, action):
        if state in self.jump.keys():
            return self.reward[state]
        if self.get_next_state(state,action) == state:
            return -1
        return 0