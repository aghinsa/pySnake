import pygame
class Snake:
    x = []
    y = []
    step = 10
    direction = 0
    length = 3

    update_count = 0
    update_count_max = 1

    def __init__(self,length,window_size):
        self.length = length
        self.init_length = length
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.length=self.init_length
        self.x=[]
        self.y=[]
        for _ in range(self.length):
            self.x.append(0)
            self.y.append(0)


    def _update(self):
        for i in range(self.length-1,0,-1):
            self.x[i]=self.x[i-1]
            self.y[i]=self.y[i-1]

    def update(self):

        for i in range(self.length-1,0,-1):
            self.x[i]=self.x[i-1]
            self.y[i]=self.y[i-1]

        if self.direction == 0:
            self.x[0] += self.step
        elif self.direction == 1:
            self.x[0] -= self.step
        elif self.direction == 2:
            self.y[0] -= self.step
        elif self.direction == 3:
            self.y[0] += self.step
        
        self.x[0] %= self.window_size[0]
        self.y[0] %= self.window_size[1]


    def moveRight(self):
        self.direction = 0

    def moveLeft(self):
        self.direction = 1        

    def moveUp(self):
        self.direction = 2

    def moveDown(self):
        self.direction = 3        

    @property
    def position(self):
        return(self.x[0],self.y[0])
    
    def draw(self,surface,snake_size):
        pygame.draw.rect(surface,(180,160,255), 
                        pygame.Rect(
                            (self.x[0],self.y[0]),
                            snake_size)
                         )
        for i in range(1,self.length):
                pygame.draw.rect(surface,(51,153,255), 
                        pygame.Rect(
                            (self.x[i],self.y[i]),
                            snake_size)
                         )
            

    def eat(self,pos):
        self.x.insert(0,pos[0])
        self.y.insert(0,pos[1])


class Food:
    x,y=(0,0)
    step=44

    def __init__(self,x,y):
        self.x = x*self.step
        self.y = y*self.step
    @property
    def position(self):
        return (self.x,self.y)
    @position.setter
    def position(self,value):
        self.x=value[0]
        self.y=value[1]

    def draw(self,surface,food_size):
        pygame.draw.rect(surface,(255,153,51), 
                    pygame.Rect(
                        self.position,
                        food_size)
                         )
        
