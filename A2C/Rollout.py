import numpy as np
import scipy.signal

class Rollout:

    def discount(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

    def __init__(self):
        self.rewards = []
        self.observation = []
        self.action_selected = []
        self.h_state = []
        self.c_state = []
        self.values = []

    def add(self, reward, observation, action_selected, value, h_state, c_state):
        self.rewards.append(reward)
        self.observation.append(observation)
        self.action_selected.append(action_selected)
        self.values.append(value)
        self.h_state.append(h_state)
        self.c_state.append(c_state)

    def make_data(self, finalReward, gamma, LAMBDA):
        feed = {}
        self.rewards = np.array(self.rewards)
        self.values = np.array(self.values + [finalReward])
        rewards_plus_v = np.append(self.rewards, [finalReward])
        batch_r = self.discount(rewards_plus_v, gamma)[:-1]
        delta_t = self.rewards + gamma * self.values[1:] - self.values[:-1]
        batch_adv = self.discount(delta_t, gamma * LAMBDA)

        feed['rewards'] = batch_r
        feed['observations'] = np.array(self.observation)
        feed['action_selected'] = np.array(self.action_selected)
        feed['h_state'] = np.array(self.h_state)
        feed['c_state'] = np.array(self.c_state)
        feed['advantage'] = batch_adv


        self.rewards = []
        self.observation = []
        self.action_selected = []
        self.h_state = []
        self.c_state = []
        self.values = []

        return feed
