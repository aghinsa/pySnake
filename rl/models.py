import tensorflow as tf
from collections import namedtuple
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim

DQNNetworkType = namedtuple('dqn_network', ['q_values'])


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
        def infer_shape(x):
            x = tf.convert_to_tensor(x)

            # If unknown rank, return dynamic shape
            if x.shape.dims is None:
                return tf.shape(x)

            static_shape = x.shape.as_list()
            dynamic_shape = tf.shape(x)

            ret = []
            for i in range(len(static_shape)):
                dim = static_shape[i]
                if dim is None:
                    dim = dynamic_shape[i]
                ret.append(dim)

            return ret

        def merge_last_two_dims(tensor):
            shape = infer_shape(tensor)
            shape[-2] *= shape[-1]
            shape.pop(-1)
            return tf.reshape(tensor, shape)

        x = tf.cast(state, tf.float32)
        x = tf.div(x, 255.)
        x = merge_last_two_dims(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)

        return DQNNetworkType(self.dense2(x))
