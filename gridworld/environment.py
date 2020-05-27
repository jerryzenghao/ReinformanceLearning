import numpy as np
import random 

class Agent:
    def __init__(self, env,discount_rate = 0.9):
        self.env = env
        #self.state = init_state
        self.policy = [[[0.25,0.25,0.25,0.25] for _ in range(self.env.height)] for _ in range(self.env.width)] # (W,H,4) list
        self.action_value = np.random.rand(self.env.height, self.env.width, 4)*0.01
        self.value = np.random.rand(self.env.height,self.env.width)*0.01
        self.discount_rate = discount_rate

    def get_policy(self,state):
        
        return self.policy[state[0]][state[1]]
    
    def get_action(self,state):
        action = [0,1,2,3]
        policy = self.get_policy(state)
        return np.random.choice(action,size=1,p=policy).item()




class DPAgent(Agent):
    
    def policy_evaluation(self):
        while True:
            delta = 0
            v_new = np.zeros((self.env.height,self.env.width))
            for i in range(self.env.height):
                for j in range(self.env.width):
                    s = (i,j)
                    v = self.value[s]
                    policy = self.get_policy(s)
                    for a in range(4):
                        pi = policy[a]
                        r = self.env.get_reward(s,a)
                        v_next = self.value[self.env.get_next_state(s,a)]
                        v_new[s] += pi*(r+self.discount_rate*v_next)
                    delta = max(delta, abs(v-v_new[s]))
            self.value = v_new
            
            if delta < 0.001:
                return self.value


    def policy_improvement(self):
        stable = True
        for i in range(self.env.height):
            for j in range(self.env.width):
                s = (i,j)
                policy = self.get_policy(s)
                nmax = 0
                v = []
                for a in range(4):
                    pi = policy[a]
                    r = self.env.get_reward(s,a)
                    vn = self.value[self.env.get_next_state(s,a)]
                    v.append(r + self.discount_rate*vn)
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
                return self.value, self.policy

class QlearningAgent(Agent):
    def Q_learning(self,alpha, epsi=0.3):
        action = [0,1,2,3]
        Q_new = np.random.rand(self.env.height,self.env.width,4)*0.01
        for i in range(2000):
            s = random.choice(self.env.all_states)
            for j in range(100):
                # choose A by epsilon-greedy
                coin = np.random.binomial(1,1-epsi)
                if coin:
                    A = np.argmax(Q_new[s])
                else:
                    A = random.choice(action)
                r = self.env.get_reward(s,A)
                s_next = self.env.get_next_state(s,A)
                Q_new[s][A] += alpha*(r+self.discount_rate* max(Q_new[s_next])-Q_new[s][A])
                s = s_next
        self.action_value = Q_new
        self.value = Q_new.max(axis = 2)
        # get policy
        for i in self.env.all_states:
            nmax = 0
            Qmax = max(self.action_value[i])
            new_policy = [0,0,0,0]
            for a in range(4):
                if Qmax - self.action_value[i][a] < 0.2:
                    nmax += 1
                    new_policy[a] = 1.0

            for a in range(4):
                    self.policy[i[0]][i[1]][a] = new_policy[a]/nmax 


        return  self.value,self.policy # self.action_value,
                



class TDAgent(Agent):
    def temporal_difference(self,alpha):
        v_new = np.random.rand(self.env.height,self.env.width)*0.01
        for i in range(2000):
            s = random.choice(self.env.all_states)
            for j in range(100):
                a = self.get_action(s)
                r = self.env.get_reward(s,a)
                s_new = self.env.get_next_state(s,a)
                v_new[s] += alpha*(r+self.discount_rate*v_new[s_new]-v_new[s])
                s = s_new
        self.value = v_new
        return self.value