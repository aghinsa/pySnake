import pygame
class Snake:
    x = []
    y = []
    step = 44
    direction = 0
    length = 3

    update_count = 0
    update_count_max = 1

    def __init__(self,length):
        self.length = length
        for _ in range(length):
            self.x.append(0)
            self.y.append(0)


    def update(self):

        # self.update_count+=1
        # if self.update_count > self.update_count_max:
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
        else:
            raise Exception

            # self.update_count = 0

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
        for i in range(self.length):
            pygame.draw.rect(surface,(51,153,255), 
                    pygame.Rect(
                        (self.x[i]/2,self.y[i]/2),
                        snake_size)
                         )


class Food:
    x,y=(0,0)
    step=44

    def __init__(self,x,y):
        self.x = x*self.step
        self.y = y*self.step
    @property
    def position(self):
        return (self.x,self.y)

    def draw(self,surface,food_size):
        pygame.draw.rect(surface,(255,153,51), 
                    pygame.Rect(
                        self.position,
                        food_size)
                         )
        
