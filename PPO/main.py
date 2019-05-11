import sys
sys.path.append('./Models')
sys.path.append('./Games')
import numpy as np
from PPO import PPO
from Environment import Environment
from Chase import Chase
import gym
import datetime
import tensorflow as tf
import time
import utils as U
num_parallel = 10
name_env = 'Chase'
network = 'LSTM'
epochs = 5

def createSummaryWriter():
    TBDIR = ''
    date = datetime.datetime.now()
    stamp = date.strftime('%Y.%m.%d_%H.%M')
    title = name_env + "_" + stamp + "_x" + str(num_parallel)
    TBDIR = './tb/' + title
    return tf.summary.FileWriter(TBDIR)
customEnv = True
def main():
    if not customEnv:
        env = gym.make(name_env)
        input_shape = env.observation_space.shape
        input_shape = U.preprocess(np.zeros(input_shape)).shape
        output_shape = (env.action_space.n,)
    else:
        env = Chase(100,100)
        input_shape = env.inputShape()
        output_shape = (env.outputShape(),)
    sw = createSummaryWriter()
    agent = PPO(network, input_shape, output_shape,sw)
    renderer = Environment(name_env, agent, sw, True, customEnv)
    renderer.daemon = True
    renderer.start()
    environments = [Environment(name_env, agent, sw, False, customEnv) for x in range(num_parallel)]
    Environment.wait = num_parallel
    for env in environments:
        env.daemon = True
        env.start()
    while True:
        if Environment.wait == 0:
            agent.updateModel(Environment.training_queue, epochs)
            Environment.training_queue = []
            Environment.wait = num_parallel
            for env in environments:
                env.canGo = True

        else:
            time.sleep(.001)

    for env in environments:
        env.join()



if __name__ == '__main__':
    main()
