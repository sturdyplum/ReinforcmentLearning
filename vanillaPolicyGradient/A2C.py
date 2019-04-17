import sys
sys.path.append('../Models')
from AtariModels import CNN

import tensorflow as tf
import tensorflow.keras.backend as K
import sys

import numpy as np

class A2C:

    def __init__(self, network, input_shape, output_shape, summary_writer):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.training_counter = 0
        self.learning_rate = 1e-3
        self.discount = .99
        self.LAMBDA = 1
        self.summary_writer = summary_writer
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        self.input, self.value, self.policy, self.h_state, self.c_state, self.state_shape = CNN('A2C', input_shape, output_shape, network)
        self.buildLoss('A2C')
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()

    def act(self, observation, h_state, c_state):
        with self.session.as_default():
            policy, value, h_state, c_state  = self.session.run([self.policy, self.value, self.h_state, self.c_state], feed_dict = {
                self.input:np.array([observation]),
                self.h_state:np.array([h_state]),
                self.c_state:np.array([c_state])
            })
            policy = policy[0]
            return np.random.choice(len(policy), p=policy), value, h_state[0], c_state[0]

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
            # advantage = self.target_value - self.value

            action_probability = tf.reduce_sum(self.action_selected * self.policy, axis=1)
            policy_loss = -tf.log(action_probability + 1e-10) * self.advantage

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

    def updateModel(self, queue):

        feed = {}
        for batch in queue:
            if self.input not in batch:
                feed[self.input] = batch['observations']
                feed[self.target_value] = batch['rewards']
                feed[self.action_selected] = batch['action_selected']
                feed[self.h_state] = batch['h_state']
                feed[self.c_state] = batch['c_state']
                feed[self.advantage] = batch['advantage']
            else:
                feed[self.input] = np.concatenate((feed[self.input], batch['observations']))
                feed[self.target_value]= np.concatenate((feed[self.target_value], batch['rewards']))
                feed[self.action_selected] = np.concatenate((feed[self.action_selected],batch['action_selected']))
                feed[self.h_state] = np.concatenate((feed[self.h_state],batch['h_state']))
                feed[self.c_state] = np.concatenate((feed[self.c_state],batch['c_state']))
                feed[self.advantage] = np.concatenate((feed[self.advantage], batch['advantage']))

        # print(?"PPPPPPPPPPPPPPPPPP", self.session.run(self.c_state), feed_dict=feed)
        # print(self.session.run(self.h_state), feed_dict=feed)
        # print(feed[self.input].shape, feed[self.h_state].shape,'ooooooooooooooo')
        _, summary = self.session.run([self.train_op,self.summary_op], feed_dict = feed)
        self.training_counter +=  1
        self.summary_writer.add_summary(summary, self.training_counter)
        self.summary_writer.flush()
