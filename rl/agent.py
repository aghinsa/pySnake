import gym
import numpy as np
import tensorflow as tf

from dopamine.replay_memory import prioritized_replay_buffer,circular_replay_buffer
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains.run_experiment import Runner
from dopamine.discrete_domains.gym_lib import create_gym_environment
from models import SimpleDQNNetwork


STACK_SIZE = 4
GAMMA = 0.9
REPLAY_CAPACITY = 10000
BATCH_SIZE = 32

class SnakeDQNAgent(dqn_agent.DQNAgent):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def _build_replay_buffer(self,use_staging):

        """Creates the replay buffer used by the agent.
        Args:
        use_staging: bool, if True, uses a staging area to prefetch data for
            faster training.
        Returns:
        A WrapperReplayBuffer object.
        """
        return circular_replay_buffer.WrappedReplayBuffer(
            replay_capacity = REPLAY_CAPACITY,
            batch_size = BATCH_SIZE,
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype
        )




env = create_gym_environment(
                        environment_name="gym_snake_classic:SnakeClassic",
                        version = 'v0'
                            )
memory_buffer = prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
    observation_shape=env.observation_space.shape,
    stack_size=STACK_SIZE,
    replay_capacity=100,
    batch_size=32,
    gamma=GAMMA,
    )
print(env.action_space.n)
sess = tf.Session()



def _agent_fn(sess,env,summary_writer):
    AGENT = SnakeDQNAgent(
    sess=sess,
    num_actions = env.action_space.n,
    observation_shape = env.observation_space.shape,
    stack_size = STACK_SIZE,
    network = SimpleDQNNetwork,
    gamma=GAMMA,
    tf_device = '/gpu:0' ,
    summary_writer=summary_writer
    )
    return AGENT

def _env_fn(*args):
    return env

runner = Runner(
            base_dir = '_tmp_agent_dir/',
            create_agent_fn = _agent_fn,
            create_environment_fn= _env_fn,
            checkpoint_file_prefix='ckpt',
            logging_file_prefix='log',
            log_every_n=10,
            num_iterations=200,
            training_steps=2500,
            evaluation_steps=1250,
            max_steps_per_episode=10000
               )
runner.run_experiment()



