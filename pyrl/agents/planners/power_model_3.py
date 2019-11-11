import numpy as np
from fitted_q_iteration import FittedQIteration

import julia
j = julia.Julia(compiled_modules=False)
julia_model = j.include("power_model_qty.jl")

# State representation : bid 24, 48, 2, 1 hours + timestamp (0-720)

class BatchModel():
    def __init__(self, **kwargs):
        # Set up parameters
        self.max_experiences = 700
        self.episode_len = 720
        self.action_mapping = {}


    def model_init(self):
        self.stateSpace = 5
        self.numActions = 0
        self.populate_actions()


        # Initialize storage for training data
        self.experiences = np.zeros((self.max_experiences, self.numActions, self.stateSpace))
        self.transitions = np.zeros((self.max_experiences, self.numActions, self.stateSpace))
        self.rewards = np.zeros((self.max_experiences, self.numActions))
        self.exp_index = np.zeros(self.numActions)

    def populate_actions(self):
        b1 = 1.0
        self.numActions = 0
        while b1 <= 5:
            b2 = 1.0
            while b2 <= 5:
                self.action_mapping[self.numActions] = [b1,b2]
                b2+=0.5
                self.numActions += 1
            b1+=0.5

    def getStateSpace(self):
        state_ranges = np.array([[1.0,5.0],[1.0,5.0],[1.0,24.0],[1.0,30.0],[0,1.5]])
        return state_ranges, self.numActions

    def sampleStateActions(self, num_requested):
        sample = []
        pred = []
        for a in range(self.numActions):
            rnd = list(range(int(min(self.exp_index[a], self.experiences.shape[0]))))
            np.random.shuffle(rnd)
            num_available = int(min(self.exp_index[a], num_requested))
            action_sample = self.experiences[rnd[:num_available],a]

            sample += [action_sample]
            pState = self.transitions[rnd[:num_available],a]
            pRewards = self.rewards[rnd[:num_available],a]

            
            pred += [[pState, pRewards, [1.0]*num_available]]
        return sample, pred

    def step(self, state, action):
        prev_ts = (state[3]-1)*24 + state[2]
        rewards, qty = julia_model(self.action_mapping[action][0], self.action_mapping[action][1],prev_ts)
        rewards_n = (rewards-np.average(rewards))
        rewards_n /= np.std(rewards_n)
        reward = rewards_n[0]
        cur_ts = prev_ts + 1
        done = False
        self.bids[cur_ts-1] = action
        self.bid_rewards[cur_ts-1] = reward
        self.qty[cur_ts-1] = np.sum(qty)
        prev_bid = np.argmax([self.bid_rewards[cur_ts-48], self.bid_rewards[cur_ts-24], self.bid_rewards[cur_ts-1], self.bid_rewards[cur_ts-2]])
        prev_bid_mapping = {0:cur_ts-48, 1:cur_ts-24, 2:cur_ts-1, 3:cur_ts-2}
        a = self.bids[prev_bid_mapping[prev_bid]]
        hour, day = 1, state[3]
        if (state[2] == 24):
            day = state[3]+1
            hour = 1
        else:
            hour = state[2]+1
        qty = (self.qty[cur_ts-48] + self.qty[cur_ts-24])/4
        newState = [self.action_mapping[a][0], self.action_mapping[a][1], hour, day, qty]        

        if cur_ts == 720:
            done = True
        return newState,reward, done

    # only for eval
    def step_2(self, state, action):
        prev_ts = (state[3]-1)*24 + state[2]
        rewards,qty = julia_model(self.action_mapping[action][0], self.action_mapping[action][1],prev_ts)
        rewards_n = (rewards-np.average(rewards))
        rewards_n /= np.std(rewards_n)
        reward = rewards_n[0]
        cur_ts = prev_ts + 1
        done = False
        self.bids[cur_ts-1] = action
        self.bid_rewards[cur_ts-1] = reward
        self.qty[cur_ts-1] = np.sum(qty)
        prev_bid = np.argmax([self.bid_rewards[cur_ts-48], self.bid_rewards[cur_ts-24], self.bid_rewards[cur_ts-1], self.bid_rewards[cur_ts-2]])
        prev_bid_mapping = {0:cur_ts-48, 1:cur_ts-24, 2:cur_ts-1, 3:cur_ts-2}
        a = self.bids[prev_bid_mapping[prev_bid]]
        hour, day = 1, state[3]
        if (state[2] == 24):
            day = state[3]+1
            hour = 1
        else:
            hour = state[2]+1
        qty = (self.qty[cur_ts-48] + self.qty[cur_ts-24] + self.qty[cur_ts-1] + self.qty[cur_ts-2])/4            
        newState = [self.action_mapping[a][0], self.action_mapping[a][1], hour, day, qty]       

        if cur_ts == 720:
            done = True
        return newState,rewards, done


    def sample_action(self):
        return np.random.randint(self.numActions)

    def reset(self):
        self.bids = np.random.randint(self.numActions, size=self.episode_len+1)
        self.bid_rewards = np.zeros(self.episode_len+1)
        self.qty = np.array([0.75]*(self.episode_len+1))
        a = self.sample_action()
        return [self.action_mapping[a][0], self.action_mapping[a][1], np.random.randint(24)+1, np.random.randint(30)+1, 0.75]

    def updateExperience(self, lastState, action):
        newState, reward, done = self.step(lastState, action)
        index = int(self.exp_index[action] % self.max_experiences)
        self.experiences[index,action, :] = lastState
        self.rewards[index, action] = reward
        self.transitions[index, action, :] = newState
        self.exp_index[action]+=1
        return newState, done

    def populateExperience(self):
        num_exp = 0
        state = self.reset()
        while num_exp < 6000:
            state, done = self.updateExperience(state, self.sample_action())
            if done:
                state = self.reset()
                print('Added episode', num_exp)
            num_exp+=1



model = BatchModel()
model.model_init()
model.populateExperience()

fitted_q = FittedQIteration(model)
fitted_q.planner_init()
fitted_q.updatePlan()
fitted_q.test()