import os
import gym
import pygame
import numpy as np
from matplotlib.image import imread
from gym import error, spaces, utils
from gym.utils import seeding

from gym_snake_classic.envs.src.game import Game,GameConfig
from gym_snake_classic.envs.src.assets import Snake,Food

class SnakeClassicEnv(gym.Env):
    metadata = {'render.modes':['human']}

    ACTION_LOOKUP = {
            0 : 'UP',
            1 : 'DOWN',
            2 : 'LEFT',
            3 : 'RIGHT'
    }

    def __init__(self):
        width,height = (800,600)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=
                    (height, width, 3))
        cfg = GameConfig(width = 800,
                    height = 600,
                    player = Snake,
                    food = Food,
                    player_size = (20,20),
                    food_size = (20,20),
                    render = True
                    )

        self.snake_game = Game(cfg)
        self.snake_game.on_init()
    
    def __del__(self):
        self.snake_game.on_cleanup()
        os.remove(self.temp_filename)
        
    def _observe(self):
        self.temp_filename='_temp_window.jpg'
        pygame.image.save(self.snake_game.window,self.temp_filename)
        obs = imread(self.temp_filename)
        return obs

    def step(self, action):
        
        self.take_action(action)
        self.snake_game.on_loop()
        #observation
        #TODO figure out a faster way
        obs = self._observe()
        
        #reward
        reward = self.get_reward()
        
        #done
        done = self.snake_game.done
        
        #info
        info = {}
        return (obs,reward,done,info)

    def take_action(self, action):
        act = self.ACTION_LOOKUP[action]
        self.snake_game.take_action(act)

    def get_reward(self):
        return self.snake_game.score
        
    def reset(self):
        self.snake_game.reset()
        return self._observe
        

    def render(self,mode='human',):
        self.snake_game.on_render()
    
        