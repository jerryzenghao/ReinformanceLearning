import numpy as np
import random 

class Agent:
    def __init__(self, env, gamma = 0.9):
        self.env = env
        self.gamma = gamma
        self.policy = [[[0.25,0.25,0.25,0.25] for i in range(env.state_size[0])] for j in range(env.state_size[1])]
        self.value = np.zeros(env.state_size)
        for i in env.wall:
            self.value[i] = None
            self.policy[i[0]][i[1]] = None
        
        
        for i in range(1,9):
            self.policy[0][i] = [0,1/3,1/3,1/3]
            self.policy[-1][i] = [1/3,1/3,0,1/3]
            self.policy[i][0] = [1/3,1/3,1/3,0]
            self.policy[i][-1] = [1/3,0,1/3,1/3]
        self.policy[0][0] = [0,0.5,0.5,0]
        self.policy[0][-1] = [0,0,0.5,0.5]
        self.policy[-1][0] = [0.5,0.5,0,0]
        self.policy[-1][-1] = [0.5,0,0,0.5]
                
        
    def get_policy(self,s):
        
        return self.policy[s[0]][s[1]]





class DPAgent(Agent):
    
    def policy_evaluation(self):
        
        while True:
            delta = 0
            v_new = np.zeros(self.env.state_size)
            for i in range(self.env.state_size[0]):
                for j in range(self.env.state_size[1]):
                    s = (i,j)
                    if s in self.env.wall:
                        v_new[s] = None
                        continue
                    policy = self.get_policy(s)
                    r = self.env.get_reward(s)
                    for a in range(4):
                        s_n = self.env.get_next_state(s,a)
                        v_new[s] += policy[a]*(r + self.gamma*self.value[s_n])

                    delta = max(delta, abs(self.value[s]-v_new[s]))
            
            self.value = v_new
            if delta < 1e-6:
                print('Policy Evaluation:\n', np.round(self.value,2))
                return

                    
                    


    def policy_improvement(self):
        
        stable = True
        for i in range(self.env.state_size[0]):
            for j in range(self.env.state_size[1]):
                s = (i,j)
                if s in self.env.wall:
                    continue
                policy = self.get_policy(s)
                nmax = 0
                vmax = - np.inf
                v = []
                r = self.env.get_reward(s)
                for a in range(4):
                    pi = policy[a]
                    s_n = self.env.get_next_state(s,a)
                    v.append(r + self.gamma*self.value[s_n])
                    if a == 0 or v[a] > vmax:
                        vmax = v[a]
                        nmax = 1
                    elif v[a] == vmax:
                        nmax += 1
                new_policy = [0,0,0,0]
                for a in range(4):
                    if vmax == v[a]:
                        new_policy[a] = 1.0/nmax
                    else:
                        new_policy[a] = 0.0
                if policy != new_policy:
                    stable = False
                self.policy[s[0]][s[1]] = new_policy
                
        return stable
        

                    
    
    def policy_iteration(self):
        
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                print('Policy stable.')
                return



class TDAgent(Agent):
    pass