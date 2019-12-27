import gym
from dopamine.discrete_domains.gym_lib import create_gym_environment


def test_snake_classic(dopamine=False):
    if(dopamine):
        env = create_gym_environment(
                        environment_name="gym_snake_classic:SnakeClassic",
                        version = 'v0'
                        )
    else:
        env = gym.make("gym_snake_classic:SnakeClassic-v0")
    env.reset()
    if not dopamine:
        env.render()
    for _ in range(100):
        action = 3
        state, reward, done, _ = env.step(action)
        if not dopamine:
            env.render()
        if done:
            break
    print(f"Reward :{reward}")


if __name__ == "__main__":
    test_snake_classic()