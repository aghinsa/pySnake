from gym.envs.registration import register
register(
    id='SnakeClassic-v0',
    entry_point='gym_snake_classic.envs:SnakeClassicEnv',
)