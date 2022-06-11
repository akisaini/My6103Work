#%%
import pygame
from pygame.locals import *
import time
import random
#%%

SIZE =  40

class Apple():
    def __init__(self, screen):
        self.image = pygame.image.load('resources/apple.jpg').convert()
        self.screen = screen
        self.x = 120
        self.y = 120
        
    def draw_apple(self):
        self.screen.blit(self.image, (self.x, self.y))
        pygame.display.flip()       

    def move(self):
        self.x =  random.randint(1,25)*SIZE
        self.y = random.randint(1,20)*SIZE
        
class Snake():
    # self and screen(surface)
    def __init__(self, screen, length):
        # setting up the block
        self.screen = screen
        self.length = length
        self.image = pygame.image.load('resources/block.jpg').convert()
        # self.surface equivalent
    # x and y coordinates
        self.block_x = [40]*length
        self.block_y = [40]*length
        self.direction = 'right'
        
        
    def draw_block(self):
        self.screen.fill((110,110,5))
        
        for i in range(self.length):
            self.screen.blit(self.image, (self.block_x[i],self.block_y[i]))
        pygame.display.flip()


    def move_down(self):
        self.direction = 'down'
         
    def move_up(self):
        self.direction = 'up'
        
    def move_left(self):
        self.direction = 'left'
              
    def move_right(self):
        self.direction = 'right'         
    
    def trail(self):
        # update body 
        for i in range(self.length-1, 0, -1):
            self.block_x[i] = self.block_x[i-1]
            self.block_y[i] = self.block_y[i-1]
            
        
        if self.direction == 'right':
            self.block_x[0] += SIZE
        if self.direction == 'left':
            self.block_x[0] -= SIZE
        if self.direction == 'up':
            self.block_y[0] -= SIZE
        if self.direction == 'down':
            self.block_y[0  ] += SIZE 
        
        self.draw_block()
        
    def increase_length(self):
        self.length += 1
        self.block_x.append(-1)
        self.block_y.append(-1)
    
    
class Game():
    def __init__(self):
        pygame.init()
        self.surface = pygame.display.set_mode((1000,800))
        self.snake = Snake(self.surface, 5)
        self.snake.draw_block()
        self.apple = Apple(self.surface)
        self.apple.draw_apple()
        
        
    def play(self):
        self.snake.trail()
        self.apple.draw_apple()
          
    
    def running(self):
        
        running_ = True
        
        while running_:
            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running_ = False
                        
                    if event.key == K_DOWN:
                        self.snake.move_down()
                        
                    if event.key == K_UP:
                        self.snake.move_up()
                        
                    if event.key == K_RIGHT:
                        self.snake.move_right()
                        
                    if event.key == K_LEFT:
                        self.snake.move_left()
                        
                elif event.type == QUIT:
                        running_ = False  
             
             
            self.play() 
            time.sleep(0.1)
            
if __name__ == '__main__':
    game = Game()
    game.running()

# %%
