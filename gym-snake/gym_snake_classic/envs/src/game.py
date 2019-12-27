import pygame
import time

from pygame.locals import *
from dataclasses import dataclass
from typing import Any,Tuple
from random import randint

@dataclass
class GameConfig:
    height : int 
    width : int 
    player : Any
    food : Any
    player_size : Tuple[int]
    food_size : Tuple[int]
    render : bool = True
    rgb : bool = True
    
class Game:
    def __init__(self,config:GameConfig)->None:
        self.config = config
        self.window_size = (self.config.width,self.config.height)
        self.player = self.config.player(5,self.window_size)
        self._running = True
        self.food = self.config.food(5,5) # setting init position

    def reset(self):
        self.player.reset()
        self.on_loop()

    def on_init(self):
        pygame.init()
        self.display = pygame.display.set_mode(
            self.window_size,
            pygame.HWSURFACE
        )
        pygame.display.set_caption('Snake')
        self._running=True
        self.snake_body = pygame.Surface( self.config.player_size )
        
        self.food_img = pygame.Surface( self.config.food_size )

    
    def spawn_food(self):
        step = self.food.step
        nx=randint(2,10)*step
        ny=randint(2,10)*step
        while((nx,ny) in zip(self.player.x,self.player.y)):
            nx=randint(2,10)*step
            ny=randint(2,10)*step
        self.food.position=(nx,ny)

    @property
    def window(self):
        return self.display
    
    @property
    def score(self):
        return self.player.length
    @property
    def done(self):
        return not self._running

    def on_loop(self):
        self.player.update()

        # check collison with food
        head = self.snake_body.get_rect(topleft=self.player.position)
        food_pos = self.food_img.get_rect(topleft=self.food.position)
        
        
        if( head.colliderect(food_pos) ):
            self.player.length = self.player.length+1
            self.player.eat(food_pos)
            step = self.food.step
            self.spawn_food()

        #check collision with self
        for _pos in zip(self.player.x[1:],self.player.y[2:]) :
            head = pygame.Rect(self.player.position,(1,1))
            _pos = pygame.Rect(_pos,(1,1))

            if( head.colliderect(_pos) ):
                self._running = False

    def on_render(self):
        self.display.fill((255,255,255))
        self.player.draw(self.display,self.config.player_size)
        self.food.draw(self.display,self.config.food_size)
        pygame.display.flip()
        

    def on_cleanup(self):
        pygame.quit()

    def take_action(self,act):
        if(act=='UP'):
            self.player.moveUp() 
        if(act=='DOWN'):
            self.player.moveDown()  
        if(act=='LEFT'):
            self.player.moveLeft() 
        if(act=='RIGHT'):
            self.player.moveRight()



