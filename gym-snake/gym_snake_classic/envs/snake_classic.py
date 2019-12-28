import os
import gym
import configs
import pygame
import numpy as np


from gym import spaces
from gym.utils import seeding
from matplotlib.image import imread
from gym_snake_classic.envs.src.assets import Snake,Food
from gym_snake_classic.envs.src.game import Game,GameConfig



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
        width,height = (800,600)
        self.action_space = spaces.Discrete(4)
        self.n_steps = 0
        self.reward = 0
        self.prev_reward = -1
        cfg = GameConfig(width = width,
                    height = height,
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
        self.snake_game.on_render(show=configs.SHOW)
        
        #observation
        #TODO figure out a faster way
        obs = self._observe()
        
        #done
        done = self.snake_game.done
        
        if done :
            self.reward -= 10
            reward = self.reward #reset changes reward
            self.reset()
        else:
            if(not self.prev_reward == self.reward):
                self.reward += 100
            reward = self.reward
            
        self.prev_reward=reward
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
    
        