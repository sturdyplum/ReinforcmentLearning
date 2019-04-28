import sys
sys.path.append('../Models')
from AtariModels import CNN

import tensorflow as tf
import tensorflow.keras.backend as K
import sys

import numpy as np

class PPO:

    def __init__(self, network, input_shape, output_shape, summary_writer):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.training_counter = 0
        self.learning_rate = 1e-3
        self.discount = .99
        self.LAMBDA = 1
        self.clip_range = .2
        self.summary_writer = summary_writer
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        self.input, self.value, self.policy, self.h_state, self.c_state, self.h_state_out, self.c_state_out, self.state_shape = CNN('PPO', input_shape, output_shape, network)
        self.buildLoss('PPO')
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

    def act(self, observation, h_state, c_state):
        with self.session.as_default():
            policy, value, h_state, c_state  = self.session.run([self.policy, self.value, self.h_state_out, self.c_state_out], feed_dict = {
                self.input:np.array([observation]),
                self.h_state:np.array([h_state]),
                self.c_state:np.array([c_state])
            })
            policy = policy[0]
            return np.random.choice(len(policy), p=policy), value, h_state[0], c_state[0], policy

    def getValue(self, observation, h_state, c_state):
        with self.session.as_default():
            return self.session.run(self.value, feed_dict = {
                self.input:np.array([observation]),
                self.h_state:np.array([h_state]),
                self.c_state:np.array([c_state])
            })

    def buildLoss(self, variable_scope):
        with tf.variable_scope(variable_scope):
            self.action_selected = tf.placeholder(tf.float32, [None, self.output_shape[0]])
            self.target_value = tf.placeholder(tf.float32,[None])
            self.advantage = tf.placeholder(tf.float32, [None])
            self.old_prob = tf.placeholder(tf.float32, [None, self.output_shape[0]])
            # advantage = self.target_value - self.value
            action_probability = tf.reduce_sum(self.action_selected * self.policy, axis=1)
            old_action_probability = tf.reduce_sum(self.action_selected * self.old_prob, axis=1)
            log_prob = tf.log(action_probability + 1e-10)
            old_log_prob = tf.log(old_action_probability + 1e-10)

            ratio = tf.exp(log_prob - old_log_prob)
            clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_range, 1 + self.clip_range)
            policy_loss = -tf.minimum(ratio * self.advantage, clipped_ratio * self.advantage)

            value_loss = tf.square(self.target_value - self.value)

            entropy = tf.reduce_sum(self.policy * tf.log(self.policy + 1e-10), axis=-1)

            loss = tf.reduce_mean(policy_loss + .5 * value_loss + .01 * entropy)

            opt = tf.train.AdamOptimizer(self.learning_rate)

            grads, vars = zip(*opt.compute_gradients(loss))
            grads, glob_norm = tf.clip_by_global_norm(grads, 50.0)
            self.train_op = opt.apply_gradients(zip(grads, vars))

            summary = []
            summary.append(tf.summary.scalar(
                'policy_loss', tf.reduce_mean(policy_loss)))
            summary.append(tf.summary.scalar(
                'glob_norm', glob_norm))
            summary.append(tf.summary.scalar(
                'value_loss', tf.reduce_mean(value_loss)))
            summary.append(tf.summary.scalar(
                'entropy_loss', tf.reduce_mean(entropy)))
            summary.append(tf.summary.scalar(
                'loss', tf.reduce_mean(loss)))

            self.summary_op = tf.summary.merge(summary)

    def updateModel(self, queue, epochs):

        feed = {}
        for batch in queue:
            if self.input not in feed:
                feed[self.input] = batch['observations']
                feed[self.target_value] = batch['rewards']
                feed[self.action_selected] = batch['action_selected']
                feed[self.h_state] = batch['h_state']
                feed[self.c_state] = batch['c_state']
                feed[self.advantage] = batch['advantage']
                feed[self.old_prob] = batch['old_prob']
            else:
                feed[self.input] = np.concatenate((feed[self.input], batch['observations']))
                feed[self.target_value]= np.concatenate((feed[self.target_value], batch['rewards']))
                feed[self.action_selected] = np.concatenate((feed[self.action_selected],batch['action_selected']))
                feed[self.h_state] = np.concatenate((feed[self.h_state],batch['h_state']))
                feed[self.c_state] = np.concatenate((feed[self.c_state],batch['c_state']))
                feed[self.advantage] = np.concatenate((feed[self.advantage], batch['advantage']))
                feed[self.old_prob] = np.concatenate((feed[self.old_prob], batch['old_prob']))

        for _ in range(epochs):
            _, summary = self.session.run([self.train_op,self.summary_op], feed_dict = feed)
            self.training_counter +=  1
            self.summary_writer.add_summary(summary, self.training_counter)
            self.summary_writer.flush()
