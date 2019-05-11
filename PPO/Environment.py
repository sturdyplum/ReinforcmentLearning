import sys
sys.path.append('../')
sys.path.append('../Games')
import gym
from Chase import Chase
import threading
from Rollout import Rollout
import tensorflow as tf
import numpy as np
import utils as U
import time


class Environment(threading.Thread):

    LOCK = threading.Lock()
    training_queue = []
    games = 0
    wait = 0

    def __init__(self, name_env, agent, summary_writer, renderer, customEnv):
        threading.Thread.__init__(self)
        if not customEnv:
            self.env = gym.make(name_env)
        else:
            self.env = Chase(100,100)
        self.agent = agent
        self.n_step = 40
        self.rollout = Rollout()
        self.summary_writer = summary_writer
        self.renderer = renderer
        self.canGo = False
        self.customEnv = customEnv

    def run(self):
        while True:
            done = False
            if not self.customEnv:
                observation = U.preprocess(self.env.reset())
            else:
                observation = self.env.reset()
            steps = 0
            score = 0

            h_state = np.zeros(self.agent.state_shape)
            c_state = np.zeros(self.agent.state_shape)

            while not done:
                steps += 1
                # print("STEP", steps)
                old_h_state = h_state
                old_c_state = c_state
                if self.renderer:
                    self.env.render()
                action, value, h_state, c_state, policy = self.agent.act(observation, h_state, c_state)
                old_obs = observation
                observation, reward, done, _ = self.env.step(action)
                if not self.customEnv:
                    observation = U.preprocess(observation)
                score += reward
                if not self.renderer:
                    self.rollout.add(reward, old_obs, np.eye(self.agent.output_shape[0])[action], value, old_h_state, old_c_state, policy)
                    if (steps % self.n_step == 0) or done:
                        last_reward = 0
                        if not done:
                            last_reward = self.agent.getValue(observation, h_state, c_state)
                        batch = self.rollout.make_data(last_reward, self.agent.discount, self.agent.LAMBDA)
                        with Environment.LOCK:
                            Environment.training_queue.append(batch)
                            Environment.wait -= 1
                        while not self.canGo:
                            time.sleep(.0001)
                        self.canGo = False
                else:
                    time.sleep(.002)

            if not self.renderer:
                sum = tf.Summary()
                sum.value.add(tag='score', simple_value=score)
                with Environment.LOCK:
                        Environment.games = Environment.games + 1
                        local_games = Environment.games

                self.summary_writer.add_summary(sum, local_games)
                self.env.close()
