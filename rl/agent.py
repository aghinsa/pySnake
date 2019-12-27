import gym
import numpy as np
import tensorflow as tf

from dopamine.replay_memory import prioritized_replay_buffer
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains.run_experiment import TrainRunner
from models import SimpleDQNNetwork


class SnakeDQNAgent(dqn_agent.DQNAgent):
    def __init__(self,memory,*args,**kwargs):
        self.memory=memory
        super().__init__(*args,**kwargs)

    def _build_replay_buffer(self,*args,**kwargs):
        return self.memory



STACK_SIZE = 4
GAMMA = 0.9

env = gym.make("gym_snake_classic:SnakeClassic-v0")
memory_buffer = prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
    observation_shape=env.observation_space.shape,
    stack_size=STACK_SIZE,
    replay_capacity=100,
    batch_size=32,
    gamma=GAMMA,
    )
print(env.action_space.n)
sess = tf.Session()

AGENT = SnakeDQNAgent(
    memory=memory_buffer,
    sess=sess,
    num_actions = env.action_space.n,
    observation_shape = env.observation_space.shape,
    stack_size = STACK_SIZE,
    network = SimpleDQNNetwork,
    gamma=GAMMA,
    tf_device = '/gpu:0' 
    )

def _agent_fn(*args):
    return AGENT
def _env_fn(*args):
    return env

runner = TrainRunner(
            base_dir = '_tmp_agent_dir/',
            create_agent_fn = _agent_fn,
            create_environment_fn= _env_fn,
            checkpoint_file_prefix='ckpt',
            logging_file_prefix='log',
            log_every_n=1,
            num_iterations=200,
            training_steps=250000,
            evaluation_steps=125000,
            max_steps_per_episode=27000
               )



