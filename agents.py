import gym
import configs
import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim as contrib_slim
from dopamine.replay_memory import prioritized_replay_buffer, circular_replay_buffer
from dopamine.replay_memory import prioritized_replay_buffer, circular_replay_buffer
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains.run_experiment import Runner
from dopamine.discrete_domains.gym_lib import create_gym_environment
from dopamine.discrete_domains import atari_lib
from dopamine.agents.rainbow import rainbow_agent
from models import SimpleDQNNetwork, RainbowNetwork


class SnakeDQNAgent(dqn_agent.DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_replay_buffer(self, use_staging):

        """Creates the replay buffer used by the agent.
        Args:
        use_staging: bool, if True, uses a staging area to prefetch data for
            faster training.
        Returns:
        A WrapperReplayBuffer object.
        """
        return circular_replay_buffer.WrappedReplayBuffer(
            replay_capacity=configs.REPLAY_CAPACITY,
            batch_size=configs.BATCH_SIZE,
            observation_shape=self.observation_shape,
            stack_size=configs.STACK_SIZE,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype,
        )


class SnakeRainbowAgent(rainbow_agent.RainbowAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewards = [] # For visualizer

    def step(self, reward, observation):
        self.rewards.append(reward)
        return super().step(reward, observation)
    def get_rewards(self):
        return [np.cumsum(self.rewards)]

    def _build_replay_buffer(self, use_staging):

        """Creates the replay buffer used by the agent.
        Args:
        use_staging: bool, if True, uses a staging area to prefetch data for
            faster training.
        Returns:
        A WrapperReplayBuffer object.
        """
        return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
            replay_capacity=configs.REPLAY_CAPACITY,
            batch_size=configs.BATCH_SIZE,
            observation_shape=self.observation_shape,
            stack_size=configs.STACK_SIZE,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype,
        )

    def reload_checkpoint(self, checkpoint_path, use_legacy_checkpoint=False):
        if use_legacy_checkpoint:
            variables_to_restore = atari_lib.maybe_transform_variable_names(
                tf.all_variables(), legacy_checkpoint_load=True)
        else:
            global_vars = set([x.name for x in tf.global_variables()])
            ckpt_vars = [
                '{}:0'.format(name)
                for name, _ in tf.train.list_variables(checkpoint_path)
            ]
            include_vars = list(global_vars.intersection(set(ckpt_vars)))
            variables_to_restore = contrib_slim.get_variables_to_restore(
                include=include_vars)
        if variables_to_restore:
            reloader = tf.train.Saver(var_list=variables_to_restore)
            reloader.restore(self._sess, checkpoint_path)
            tf.logging.info('Done restoring from %s', checkpoint_path)
        else:
            tf.logging.info('Nothing to restore!')

    def get_probabilities(self):
        return self._sess.run(tf.squeeze(self._net_outputs.probabilities),
                            {self.state_ph: self.state})