import os
import gym
import configs
import pygame
import numpy as np


from gym import spaces
from gym.utils import seeding
from PIL import Image
from gym_snake_classic.envs.src.assets import Snake,Food
from gym_snake_classic.envs.src.game import Game,GameConfig

WIDTH = 400
HEIGHT = 400

OBS_WIDTH=256
OBS_HEIGHT=256


class SnakeClassicEnv(gym.Env):
    """
    Gym Environment for classic snake game
    """

    metadata = {'render.modes':['human']}
    reward_range = (-np.inf, np.inf)

    ACTION_LOOKUP = {
            0 : 'UP',
            1 : 'DOWN',
            2 : 'LEFT',
            3 : 'RIGHT'
    }

    def __init__(self):
        self.temp_filename='_temp_window.jpg'
        width,height = (WIDTH,HEIGHT)
        self.action_space = spaces.Discrete(4)
        self.n_steps = 0
        self.reward = 0
        cfg = GameConfig(width = width,
                    height = height,
                    player = Snake,
                    food = Food,
                    player_size = (20,20),
                    food_size = (20,20),
                    )

        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (OBS_HEIGHT, OBS_WIDTH, 3))

        self.snake_game = Game(cfg)
        self.snake_game.on_init()
        self.prev_length = self.snake_game.player.length

    @property
    def env(self):
        return self

    def _observe(self):
        pygame.image.save(self.snake_game.window,self.temp_filename)
        obs = Image.open(self.temp_filename)
        obs=obs.resize((OBS_HEIGHT,OBS_WIDTH),Image.BILINEAR)
        obs=np.array(obs)
        return obs

    def step(self, action):
        
        self.take_action(action)
        self.snake_game.on_loop()
        self.snake_game.on_render(show=configs.SHOW)

        #observation
        #TODO figure out a faster way
        obs = self._observe()

        #done
        done = self.snake_game.done

        if done :
            self.reward -= 100
            reward = self.reward #reset changes reward
            self.reset()
        else:
            if(not self.prev_length == self.snake_game.player.length):
                self.reward += 1000
            reward = self.reward

        self.prev_length=self.snake_game.player.length
        #info
        info = {}
        return (obs,reward,done,info)


    def take_action(self, action):
        act = self.ACTION_LOOKUP[action]
        self.snake_game.take_action(act)

    def get_reward(self):
        return self.reward

    def reset(self):
        self.n_steps=0
        self.reward=0
        self.snake_game.reset()
        self.snake_game.on_loop()
        self.snake_game.on_render(show=configs.SHOW)

        return self._observe()


    def render(self,mode='human'):
        self.snake_game.on_render()
