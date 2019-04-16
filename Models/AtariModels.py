import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten


def CNN(variable_scope, input_shape, output_shape, network):
     with tf.variable_scope(variable_scope):
         input = tf.placeholder(tf.float32, [None] + [x for x in input_shape])
         if network == 'CNN':
             conv1 = Conv2D(32, kernel_size=8, strides=(4,4),padding='same',activation='relu',name='conv1')(input)
             conv2 = Conv2D(64, kernel_size=4, strides=(2,2),padding='same',activation='relu',name='conv2')(conv1)
             conv3 = Conv2D(64, kernel_size=3, strides=(1,1),padding='same',activation='relu',name='conv3')(conv2)
             flatConv3 = Flatten(name='flatConv3')(conv3)
             fc1 = Dense(512,activation='relu')(flatConv3)

         elif network == 'fcn':
             fc1 = Dense(32, activation='relu')(input)
             fc1 = Dense(32, activation='relu')(fc1)
             fc1 = Dense(32, activation='relu')(fc1)

         value = Dense(1, activation='linear')(fc1)
         value = tf.squeeze(value)
         policy = Dense(output_shape[0], activation = 'softmax')(fc1)

         return input, value, policy
