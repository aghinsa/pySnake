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

    def __init__(self,rgb=True):
        self.temp_filename='_temp_window.jpg'
        width,height = (800,600)
        self.action_space = spaces.Discrete(4)
        
        cfg = GameConfig(width = 800,
                    height = 600,
                    player = Snake,
                    food = Food,
                    player_size = (20,20),
                    food_size = (20,20),
                    render = True,
                    rgb = rgb
                    )
        if rgb:
            self.observation_space = spaces.Box(low=0, high=255, shape=
                    (height, width, 3))
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=
                    (height, width))


        self.snake_game = Game(cfg)
        self.snake_game.on_init()
    
    @property
    def env(self):
        return self
        
    def _observe(self):
        pygame.image.save(self.snake_game.window,self.temp_filename)
        obs = Image.open(self.temp_filename)
        if not self.snake_game.config.rgb:
            obs=obs.convert('LA')
        # HACK to pass three channels to memory buffer in dopamine
        # TODO figure out how to pass rgb directly
        

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
        return self._observe()
        

    def render(self,mode='human'):
        self.snake_game.on_render()
    
        