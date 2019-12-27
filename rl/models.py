import numpy as np
import tensorflow as tf

from collections import namedtuple
from utils import merge_last_two_dims

DQNNetworkType = namedtuple('dqn_network', ['q_values'])
RainbowNetworkType = namedtuple(
    'c51_network', ['q_values', 'logits', 'probabilities'])

class SimpleDQNNetwork(tf.keras.Model):

    def __init__(self, num_actions, name=None):

        """Creates the layers used for calculating Q-values.
        Args:
            num_actions: int, number of actions.
            name: str, used to create scope for network parameters.
        """
        super(SimpleDQNNetwork, self).__init__(name=name)

        self.num_actions = num_actions
        # Defining layers.
        activation_fn = tf.keras.activations.relu
        # Setting names of the layers manually to make variable names more similar
        # with tf.slim variable names/checkpoints.
        self.conv1 = tf.keras.layers.Conv2D(32, [8, 8], strides=4, padding='same',
                                            activation=activation_fn, name='Conv')
        self.conv2 = tf.keras.layers.Conv2D(64, [4, 4], strides=2, padding='same',
                                            activation=activation_fn, name='Conv')
        self.conv3 = tf.keras.layers.Conv2D(64, [3, 3], strides=1, padding='same',
                                            activation=activation_fn, name='Conv')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation=activation_fn,
                                            name='fully_connected')
        self.dense2 = tf.keras.layers.Dense(num_actions, name='fully_connected')

    def call(self, state,):
        """
        Args:
        state: Tensor, input tensor.
        Returns:
        collections.namedtuple, output ops (graph mode) or output tensors (eager).
        
        """
        #TODO Make rgb proper
        #HACK pass 
        

        x = tf.cast(state, tf.float32)
        x = tf.div(x, 255.)
        x = merge_last_two_dims(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)

        return DQNNetworkType(self.dense2(x))



class RainbowNetwork(tf.keras.Model):
    def __init__(self, num_actions, num_atoms, support, name=None):
        """Creates the layers used calculating return distributions.
        Args:
            num_actions: int, number of actions.
            num_atoms: int, the number of buckets of the value function distribution.
            support: tf.linspace, the support of the Q-value distribution.
            name: str, used to crete scope for network parameters.
        """
        super(RainbowNetwork, self).__init__(name=name)
        activation_fn = tf.keras.activations.relu
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.support = support
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
        # Defining layers.
        self.conv1 = tf.keras.layers.Conv2D(
            32, [8, 8], strides=4, padding='same', activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='Conv')
        self.conv2 = tf.keras.layers.Conv2D(
            64, [4, 4], strides=2, padding='same', activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='Conv')
        self.conv3 = tf.keras.layers.Conv2D(
            64, [3, 3], strides=1, padding='same', activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='Conv')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            512, activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='fully_connected')
        self.dense2 = tf.keras.layers.Dense(
            num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
            name='fully_connected')

    def call(self, state):
        """Creates the output tensor/op given the state tensor as input.
        See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
        information on this. Note that tf.keras.Model implements `call` which is
        wrapped by `__call__` function by tf.keras.Model.
        Args:
            state: Tensor, input tensor.
        Returns:
            collections.namedtuple, output ops (graph mode) or output tensors (eager).
        """
        x = tf.cast(state, tf.float32)
        x = tf.div(x, 255.)
        x = merge_last_two_dims(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
        probabilities = tf.keras.activations.softmax(logits)
        q_values = tf.reduce_sum(self.support * probabilities, axis=2)
        return RainbowNetworkType(q_values, logits, probabilities)