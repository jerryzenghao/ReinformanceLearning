import numpy as np
import random 

STATE_SPACE_SIZE = (10,10)
START_STATE = (0,0)
ACTIONS = (0,1,2,3)


class GridWorld:

    def __init__(self,reward_matrix):
        self.reward_matrix = reward_matrix
        self.wall = {(2,1),(2,2),(2,3),(2,4),(2,6),(2,7),(2,8), (3,4), (4,4), (5,4),(6,4), (7,4)}
        for i in self.wall:
            reward_matrix[i] = None
        self.state_size = STATE_SPACE_SIZE
        
    
    def get_next_state(self, s, a):
        if self.reward_matrix[s] > 0:
            return (0,0)
        if a == 0:
            n_s = (s[0]-1,s[1])
        elif a == 1:
            n_s = (s[0],s[1]+1)
        elif a == 2:
            n_s = (s[0]+1,s[1])
        else:
            n_s = (s[0],s[1]-1)
        if n_s not in self.wall and 0<=n_s[0]<10 and 0<=n_s[1]<10:
            return n_s
        return s

    def get_reward(self,s):
        return self.reward_matrix[s] 
        

        
    

