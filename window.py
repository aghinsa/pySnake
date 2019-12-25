from pygame.locals import *
import pygame
from dataclasses import dataclass
from typing import Any
@dataclass
class WindowConfig:
    height : int 
    width : int 
    player : Any

class Window:
    def __init__(self,config:WindowConfig)->None:
        self.config = config
        self.player = self.config.player()
        self._running = True
        self.window_size = (self.config.width,self.config.height)

    def on_init(self):
        pygame.init()
        self.display = pygame.display.set_mode(
            self.window_size,
            pygame.HWSURFACE
        )
        pygame.display.set_caption('Snake')
        self._running=True
        obj_size = (10,10)
        self.snake_body = pygame.image.load("src/img/snake.jpg").convert()

    def on_event(self,event):
        if event.type == QUIT:
            self._running=False
    
    def on_loop(self):
        pass
    
    def on_render(self):
        self.display.fill((255,255,255))
        self.display.blit(self.snake_body,
                        self.player.position)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if (self.on_init() == False):
            self._running = False
        while(self._running):
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if(keys[K_RIGHT]):
                self.player.moveRight()
            
            if(keys[K_LEFT]):
                self.player.moveLeft() 
            
            if(keys[K_UP]):
                self.player.moveUp() 
            
            if(keys[K_DOWN]):
                self.player.moveDown() 
            
            if(keys[K_ESCAPE]):
                self._running=False

            self.on_loop()
            self.on_render()
        
        print("Exiting")
        self.on_cleanup() 


