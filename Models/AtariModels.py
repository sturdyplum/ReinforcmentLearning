import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Flatten

def Conv2DLSTM(inputs, state, output_channels, kernel_size, forgetBias = 1.0):
        hidden, cell = state
        inputConv = Conv2D(4 * output_channels, kernel_size = kernel_size, strides=(1,1), padding='same')(inputs)
        hiddenConv = Conv2D(4 * output_channels, kernel_size = kernel_size, strides=(1,1), padding='same')(hidden)
        nextHidden = Add()([inputConv, hiddenConv])

        gates = tf.split(value = nextHidden, num_or_size_splits=4, axis=3)
        inputGate, nextInput, forgetGate, outputGate = gates
        nextCell = tf.sigmoid(forgetGate + forgetBias) * cell
        nextCell += tf.sigmoid(inputGate) * tf.tanh(nextInput)
        output = tf.tanh(nextCell) * tf.sigmoid(outputGate)

        return [output, output, nextCell]


        # return [inputs, hidden, cell]


def CNN(variable_scope, input_shape, output_shape, network):
     with tf.variable_scope(variable_scope):
         input = tf.placeholder(tf.float32, [None] + list(input_shape))
         if network == 'LSTM':
             conv1 = Conv2D(32, kernel_size=8, strides=(4,4),padding='same',activation='relu',name='conv1')(input)
             conv2 = Conv2D(64, kernel_size=4, strides=(2,2),padding='same',activation='relu',name='conv2')(conv1)
             conv3 = Conv2D(64, kernel_size=3, strides=(1,1),padding='same',activation='relu',name='conv3')(conv2)

             state_shape = (conv3.shape[1], conv3.shape[2], 16)
             h_state = tf.placeholder(tf.float32, [None] + list(state_shape))
             c_state = tf.placeholder(tf.float32, [None] + list(state_shape))

             lstm, h_state, c_state = Conv2DLSTM(conv3,(h_state, c_state),16,3)
             flatConv3 = Flatten(name='flatConv3')(conv3)
             fc1 = Dense(512,activation='relu')(flatConv3)
         elif network == 'CNN':
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


         if network == 'LSTM':
             return input, value, policy, h_state, c_state, state_shape
         return input, value, policy
