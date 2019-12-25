
class Snake:
    x,y=(10,10)
    speed = 1

    def moveRight(self):
        self.x += self.speed
    def moveLeft(self):
        self.x -= self.speed

    def moveUp(self):
        self.y -= self.speed
    def moveDown(self):
        self.y += self.speed

    @property
    def position(self):
        return(self.x,self.y)