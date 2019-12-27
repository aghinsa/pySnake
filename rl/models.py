import tensorflow as tf
from collections import namedtuple
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim

DQNNetworkType = namedtuple('dqn_network', ['q_values'])

def simple_dqn_network(num_actions, network_type, state,):
    """The convolutional network used to compute the agent's Q-values.
    Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    Returns:
    net: _network_type object containing the tensors output by the network.
    """
    
    net = tf.cast(state, tf.float32)
    net = tf.div(net, 255.)
    net = contrib_slim.conv2d(net, 32, [8, 8], stride=4)
    net = contrib_slim.conv2d(net, 64, [4, 4], stride=2)
    net = contrib_slim.conv2d(net, 64, [3, 3], stride=1)
    net = contrib_slim.flatten(net)
    net = contrib_slim.fully_connected(net, 512)
    q_values = contrib_slim.fully_connected(net, num_actions, activation_fn=None)
    return network_type(q_values)

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
        
        x = tf.cast(state, tf.float32)
        x = tf.div(x, 255.)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)

        return DQNNetworkType(self.dense2(x))
