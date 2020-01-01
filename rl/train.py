import gym
import configs
import numpy as np
import tensorflow as tf

from dopamine.replay_memory import prioritized_replay_buffer, circular_replay_buffer
from dopamine.agents.dqn import dqn_agent
from dopamine.discrete_domains.run_experiment import Runner
from dopamine.discrete_domains.gym_lib import create_gym_environment
from dopamine.discrete_domains import atari_lib
from dopamine.agents.rainbow import rainbow_agent
from models import SimpleDQNNetwork, RainbowNetwork
from utils import SnakeRunner

from agents import SnakeRainbowAgent

STACK_SIZE = configs.STACK_SIZE
GAMMA = configs.GAMMA
REPLAY_CAPACITY = configs.REPLAY_CAPACITY
BATCH_SIZE = configs.BATCH_SIZE


sess = tf.Session()


env = create_gym_environment(
    environment_name="gym_snake_classic:SnakeClassic", version="v0"
)


def _agent_fn(sess, env, summary_writer):
    # AGENT = SnakeDQNAgent(
    # sess=sess,
    # num_actions = env.action_space.n,
    # observation_shape = env.observation_space.shape,
    # stack_size = STACK_SIZE,
    # network = SimpleDQNNetwork,
    # gamma=0.99,
    # update_horizon=1,
    # min_replay_history=configs.MIN_REPLAY_HISTORY,
    # update_period=4,
    # target_update_period=configs.TARGET_UPDATE_PERIOD,
    # epsilon_fn=dqn_agent.linearly_decaying_epsilon,
    # epsilon_train=0.01,
    # epsilon_eval=0.001,
    # epsilon_decay_period=250000,
    # eval_mode=configs.EVAL_MODE , # True for training
    # tf_device="/gpu:*",
    # summary_writer=summary_writer,
    # summary_writing_frequency=configs.SUMMARY_WRITING_FREQUENCY,
    # )

    AGENT = SnakeRainbowAgent(
        sess=sess,
        num_actions=env.action_space.n,
        observation_shape=env.observation_space.shape,
        stack_size=STACK_SIZE,
        network=RainbowNetwork,
        num_atoms=51,
        vmax=10.0,
        gamma=0.99,
        update_horizon=1,
        min_replay_history=configs.MIN_REPLAY_HISTORY,
        update_period=4,
        target_update_period=configs.TARGET_UPDATE_PERIOD,
        epsilon_fn=dqn_agent.linearly_decaying_epsilon,
        epsilon_train=0.01,
        epsilon_eval=0.001,
        epsilon_decay_period=250000,
        eval_mode=configs.EVAL_MODE , # True for training
        replay_scheme="prioritized",
        tf_device="/gpu:*",
        summary_writer=summary_writer,
        summary_writing_frequency=configs.SUMMARY_WRITING_FREQUENCY,
    )
    return AGENT



def _env_fn(*args):
    return env


runner = SnakeRunner(
    base_dir=configs.BASE_DIR,
    create_agent_fn=_agent_fn,
    create_environment_fn=_env_fn,
    checkpoint_file_prefix="ckpt",
    logging_file_prefix="log",
    log_every_n=10,
    num_iterations=2000,
    training_steps=25000,
    evaluation_steps=12500,
    max_steps_per_episode=10000,
)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    runner.run_experiment()
    # runner.visualize(record_path = configs.BASE_DIR+'visualize/',
    #                 num_global_steps=500)
