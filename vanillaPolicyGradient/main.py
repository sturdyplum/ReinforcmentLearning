from A2C import A2C
from Environment import Environment
import gym
import datetime
import tensorflow as tf

num_parallel = 1
name_env = 'CartPole-v0'
network = 'fcn'

def createSummaryWriter():
    TBDIR = ''
    date = datetime.datetime.now()
    stamp = date.strftime('%Y.%m.%d_%H.%M')
    title = name_env + "_" + stamp + "_x" + str(num_parallel)
    TBDIR = './tb/' + title
    return tf.summary.FileWriter(TBDIR)

def main():
    env = gym.make(name_env)
    input_shape = env.observation_space.shape
    output_shape = (env.action_space.n,)
    sw = createSummaryWriter()
    agent = A2C(network, input_shape, output_shape,sw)
    while True:
        environments = [Environment(name_env, agent, sw) for _ in range(num_parallel)]
        for env in environments:
            env.daemon = True
            env.start()

        for env in environments:
            env.join()

        agent.updateModel(Environment.training_queue)

if __name__ == '__main__':
    main()
