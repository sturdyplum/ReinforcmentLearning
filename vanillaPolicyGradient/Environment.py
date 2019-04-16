import gym
import threading
from Rollout import Rollout
import tensorflow as tf
import numpy as np

class Environment(threading.Thread):

    LOCK = threading.Lock()
    training_queue = []
    games = 0

    def __init__(self, name_env, agent, summary_writer):
        threading.Thread.__init__(self)
        self.env = gym.make(name_env)
        self.agent = agent
        self.n_step = 8
        self.rollout = Rollout()
        self.summary_writer = summary_writer

    def run(self):
        done = False
        observation = self.env.reset()
        steps = 0
        score = 0
        while not done:
            steps += 1
            action = self.agent.act(observation)
            old_obs = observation
            observation, reward, done, _ = self.env.step(action)
            score += reward
            self.rollout.add(reward, old_obs, np.eye(self.agent.output_shape[0])[action])
            if (steps % self.n_step == 0) or done:
                last_reward = 0
                if not done:
                    last_reward = self.agent.getValue(observation)
                batch = self.rollout.make_data(last_reward, self.agent.discount)
                with Environment.LOCK:
                    Environment.training_queue.append(batch)

        sum = tf.Summary()
        sum.value.add(tag='score', simple_value=score)
        with Environment.LOCK:
                Environment.games = Environment.games + 1
                local_games = Environment.games

        self.summary_writer.add_summary(sum, local_games)
