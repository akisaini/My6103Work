#%%
import pygame
from pygame.locals import *
import time
import random
#%%
# 40 is the size of the block in pixels. it's a 40*40 square. 
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
        # randomizing x and y coordinates
        self.x = random.randint(1,24)*SIZE
        self.y = random.randint(1,19)*SIZE
        
class Snake():
    # self and screen(pygame surface) and initial length of the snake. 
    def __init__(self, screen):
        # setting up the block
        self.screen = screen
        self.length = 1
        self.image = pygame.image.load('resources/block.jpg').convert()
        # x and y coordinates
        self.block_x = [SIZE]
        self.block_y = [SIZE]
        self.direction = 'right'
        
        
    def draw_block(self):
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
            self.block_y[0] += SIZE 
        
        self.draw_block()
        
    def increase_length(self):
        self.length += 1
        self.block_x.append(-1)
        self.block_y.append(-1)
    
    
class Game():
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Snaky')
        
        
        # initializing the sound module. 
        pygame.mixer.init()
        self.play_background_music()
        self.surface = pygame.display.set_mode((1000,800))
        self.snake = Snake(self.surface)
        self.snake.draw_block()
        self.apple = Apple(self.surface)
        self.apple.draw_apple()
        
        
    def display_score(self):
        font = pygame.font.SysFont('arial', 30)
        score = font.render(f'Score: {self.snake.length}', True, (255,255,255))    
        self.surface.blit(score, (850,10))

    def is_collision(self, x1,y1, x2,y2):
        if x1 >= x2 and x1 < x2 + SIZE:
            if y1 >= y2 and y1 < y2 + SIZE:
                return True 
            
        return False         
    
    def render_background(self):
        bg = pygame.image.load('resources/background.jpg')  
        self.surface.blit(bg, (0,0))
        pygame.display.flip()
       
    def play_background_music(self):
        sound_bg = pygame.mixer.music.load('resources/bg_music_1.mp3')
        pygame.mixer.music.play(-1,0)
        
    def play(self):
        self.render_background()
        self.snake.trail()
        self.apple.draw_apple()
        self.display_score()
        pygame.display.flip()
        
        # Collision with apple:
        if self.is_collision(self.snake.block_x[0], self.snake.block_y[0], self.apple.x, self.apple.y):
            sound_ap = pygame.mixer.Sound('resources/1_snake_game_resources_ding.mp3')
            pygame.mixer.Sound.play(sound_ap)
            print('collision occured!')
            self.snake.increase_length()          
            self.apple.move()
        
        # collision with own body:
        for i in range(3, self.snake.length):
            if self.is_collision(self.snake.block_x[0], self.snake.block_y[0], self.snake.block_x[i], self.snake.block_y[i]):
                sound_cr = pygame.mixer.Sound('resources/1_snake_game_resources_crash.mp3')
                pygame.mixer.Sound.play(sound_cr)
                raise 'Game Over'
    
    def show_game_over(self):
        self.render_background()
        font = pygame.font.SysFont('arial', 30)
        line1 = font.render(f'Game Over! Score is {self.snake.length}', True, (255,255,255))
        self.surface.blit(line1, (200,300))
        line2 = font.render(f'To Play Again, Press enter! To Exit Press esc!', True, (255,255,255))
        self.surface.blit(line2, (200,350))
        pygame.display.flip()

        pygame.mixer.music.pause() 


    
    def running(self):
        
        running_ = True
        pause = False
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
        
        
            try:  
                if not pause: 
                    self.play()    
            except Exception as e:         
                self.show_game_over()
                pause = True
            
            
            # sleep for .2 seconds before executing the while loop again. Lowering the time will increase the speed of the snake. 
            time.sleep(.2)    
                
if __name__ == '__main__':
    game = Game()
    game.running()
# %%
