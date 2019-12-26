import gym


def test_snake_classic():
    env = gym.make("gym_snake_classic:SnakeClassic-v0")
    env.reset()
    env.render()
    for _ in range(100):
        action = 3
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break
    print(f"Reward :{reward}")


if __name__ == "__main__":
    test_snake_classic()