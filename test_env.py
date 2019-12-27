import gym
import unittest
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

class TestSnakeEnv(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("gym_snake_classic:SnakeClassic-v0")


    def test_0_snake_classic(self,render=True):
        self.env.reset()
        if render:
            self.env.render()
        for _ in range(100):
            action = 3
            state, reward, done, _ = self.env.step(action)
            if render:
                self.env.render()
            if done:
                break
        self.assertEqual(reward,-89)

    def test_1_observe(self):
        img=self.env._observe()
        uni = np.unique(img)
        self.assertEqual(len(uni),157)


if __name__ == "__main__":
    t = TestSnakeEnv()
    t.setUp()
    t.test_0_snake_classic()
    t.test_1_observe()
    # unittest.main()