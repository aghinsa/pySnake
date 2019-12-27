import os
import gym
import pygame
from PIL import Image
import numpy as np
from matplotlib.image import imread
from gym import error, spaces, utils
from gym.utils import seeding

from gym_snake_classic.envs.src.game import Game,GameConfig
from gym_snake_classic.envs.src.assets import Snake,Food

class SnakeClassicEnv(gym.Env):
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
        width,height = (800,600)
        self.action_space = spaces.Discrete(4)
        self.n_steps = 0
        self.reward = 0
        self.prev_reward = -1
        cfg = GameConfig(width = 800,
                    height = 600,
                    player = Snake,
                    food = Food,
                    player_size = (20,20),
                    food_size = (20,20),
                    )
        
        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (height, width, 3))

        self.snake_game = Game(cfg)
        self.snake_game.on_init()
    
    @property
    def env(self):
        return self
        
    def _observe(self):
        pygame.image.save(self.snake_game.window,self.temp_filename)
        obs = imread(self.temp_filename)
        return obs

    def step(self, action):
        
        self.take_action(action)
        self.snake_game.on_loop()
        self.snake_game.on_render(show=True)
        #observation
        #TODO figure out a faster way
        obs = self._observe()
        
        
        #done
        done = self.snake_game.done
        
        if done :
            self.reward -= 100
        else:
            if(self.prev_reward == self.reward):
                self.reward -= 1
            else:
                self.reward += 10
        self.prev_reward=self.reward

        #info
        info = {}
        return (obs,self.reward,done,info)


    def take_action(self, action):
        act = self.ACTION_LOOKUP[action]
        self.snake_game.take_action(act)
    
    def get_reward(self):
        return self.reward
        
    def reset(self):
        self.n_steps=0
        self.snake_game.reset()
        return self._observe()
        

    def render(self,mode='human'):
        self.snake_game.on_render()
    
        