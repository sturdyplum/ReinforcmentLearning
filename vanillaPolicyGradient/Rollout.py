import numpy as np

class Rollout:

    def __init__(self):
        self.rewards = []
        self.observation = []
        self.action_selected = []
        self.h_state = []
        self.c_state = []

    def add(self, reward, observation, action_selected, h_state, c_state):
        self.rewards.append(reward)
        self.observation.append(observation)
        self.action_selected.append(action_selected)
        self.h_state.append(h_state)
        self.c_state.append(c_state)

    def make_data(self, finalReward, discount):
        feed = {}
        discounted_rewards = []
        total = finalReward

        for reward in self.rewards[::-1]:
            total *= discount
            total += reward
            discounted_rewards.append(total)
        feed['rewards'] = np.array(discounted_rewards[::-1])

        feed['observations'] = np.array(self.observation)
        feed['action_selected'] = np.array(self.action_selected)
        feed['h_state'] = np.array(self.h_state)
        feed['c_state'] = np.array(self.c_state)

        self.rewards = []
        self.observation = []
        self.action_selected = []
        self.h_state = []
        self.c_state = []

        return feed
