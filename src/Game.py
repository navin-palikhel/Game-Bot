import sys, pygame
import time
import random, math
from copy import deepcopy
import os

from NEAT import Pool

pygame.init()

SCREEN_HEIGHT = 400
SCREEN_WIDTH = 600
WINDOW_X = 100
WINDOW_Y = 100

GRAVITY = 0.10
BIRD_JUMP_VEL = 1
JUMP_DURATION = 5

BIRD_HEIGHT = 30
BIRD_WIDTH = 75
BIRD_INI_POS_X = 200
BIRD_INI_POS_Y = 100

PIPE_WIDTH = 30
PIPE_GAP = 200
PIPE_GAP_WIDTH_FACTOR = 0.3
FRAME_SLEEP_TIME = 0

# set window position
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (WINDOW_X, WINDOW_Y)


class Bird(object):
    BIRD_HEIGHT = 15
    BIRD_WIDTH = 17
    JUMP_DURATION = 5

    def __init__(self, pos_x, pos_y, genome):
        self.isAlive = True

        self.genome = genome

        self.birdImg = "bb_1_50x50.png"
        self.spriteNo = 0

        tempBird = pygame.image.load(self.birdImg)
        self.drawBird = pygame.transform.scale(tempBird, (BIRD_WIDTH, BIRD_HEIGHT))
        self.rect = self.drawBird.get_rect()

        self.iniPosX = pos_x
        self.iniPosY = pos_y
        self.rect.y = pos_y
        self.rect.x = pos_x

        self.speed = [0, 0]

        self.isJumping = True
        self.jumpCounter = 0

        self.height = None
        self.nextPipeHeight = None
        self.nextPipeDist = None

        self.pipesCrossed = []
        self.score = 0
        self.fitness = 0
        self.flaps = 0

    def setSprite(self):
        self.spriteNo = self.spriteNo + 1
        self.spriteNo = self.spriteNo % 4
        self.birdImg = "bg_600x400.png"
        tempBird = pygame.image.load(self.birdImg)
        self.drawBird = pygame.transform.scale(tempBird, (BIRD_WIDTH, BIRD_HEIGHT))

    # self.rect = self.drawBird.get_rect()

    def setRectSpeed(self, speed):
        self.rect = self.rect.move(speed)

    def checkJump(self):
        if self.isJumping:
            self.speed[1] = -(BIRD_JUMP_VEL * self.jumpCounter - 0.5 * GRAVITY * self.jumpCounter * self.jumpCounter)
            self.jumpCounter += 1
        else:
            self.speed[1] += GRAVITY

        if self.jumpCounter == JUMP_DURATION:
            self.jumpCounter = 0
            self.isJumping = False
            self.flaps += 1

    def checkCollision(self, screen_height, pipes):
        # Bird fell below screen
        if self.rect.y > screen_height:
            self.isAlive = False
            return False
        # Brid touching ceiling
        if self.rect.y < 0:
            self.rect.y = 0

        for pipe in pipes:
            if pygame.Rect.colliderect(self.rect, pipe.topRect):
                self.isAlive = False
                return False
            if pygame.Rect.colliderect(self.rect, pipe.bottomRect):
                self.isAlive = False
                return False

    def hasCrossed(self, pipe_id):
        if pipe_id in self.pipesCrossed:
            return True
        else:
            return False

    def setInputs(self, pipes):
        self.height = None
        self.nextPipeHeight = None
        self.nextPipeDist = None

        # Inputs for neural network
        self.height = self.rect.y / SCREEN_HEIGHT
        for pipe in pipes:
            if pipe.bottomRect.x <= self.rect.x:
                continue
            else:
                self.nextPipeHeight = (SCREEN_HEIGHT - pipe.bottomRect.y) / SCREEN_HEIGHT
                self.nextPipeDist = pipe.bottomRect.x / SCREEN_WIDTH
                break


class Pipe(object):
    def __init__(self, pos_x, pipeid):
        self.id = pipeid

        midpt = random.randint(math.floor(SCREEN_HEIGHT * 0.25), math.floor(SCREEN_HEIGHT * 0.75))
        temp1 = midpt - BIRD_HEIGHT / PIPE_GAP_WIDTH_FACTOR
        temp2 = midpt + BIRD_HEIGHT / PIPE_GAP_WIDTH_FACTOR

        self.topRect = pygame.Rect(pos_x, 0, PIPE_WIDTH, temp1)
        self.bottomRect = pygame.Rect(pos_x, temp2, PIPE_WIDTH, SCREEN_HEIGHT - temp2)

        self.color = (0, 128, 0)
        self.speed = [-1.8, 0]

    def setRectSpeed(self, speed):
        self.topRect.move_ip(speed)
        self.bottomRect.move_ip(speed)

    def drawPipe(self, screen):
        pygame.draw.rect(screen, self.color, self.topRect, 0)
        pygame.draw.rect(screen, self.color, self.bottomRect, 0)


def initializeGame():
    size = width, height = SCREEN_WIDTH, SCREEN_HEIGHT
    screen = pygame.display.set_mode(size)

    # bg = pygame.image.load("bg_600x400.png").convert_alpha()
    # screen.blit(bg, (0, 0))

    birds = []

    for species in Pool.species:
        for genome in species.genomes:
            genome.generateNetwork()
            birds.append(Bird(BIRD_INI_POS_X, BIRD_INI_POS_Y, genome))

    # Generate pipes
    pipes = []
    distance = SCREEN_WIDTH
    for i in range(0, 5):
        pipes.append(Pipe(distance + PIPE_GAP * i, i))

    ticks = 0
    allDead = False

    while not allDead:

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        # color screen full shite
        screen.fill((255, 255, 255))  # White background
        # bg = pygame.image.load("bg_600x400.png").convert_alpha()
        # screen.blit(bg, (0, 0))

        for bird in birds:
            if bird.isAlive == True:
                # if bird.score >= 5:
                # 	# set sprite
                # 	bird.setSprite()

                # set bird speed
                bird.setRectSpeed(bird.speed)
                # check if bird is jumping
                bird.checkJump()
                # Draw bird on screen
                screen.blit(bird.drawBird, bird.rect)

        # draw pipes
        for pipe in pipes:
            pipe.setRectSpeed(pipe.speed)
            pipe.drawPipe(screen)

        for bird in birds:
            if bird.isAlive == True:
                # check collision of bird
                bird.checkCollision(screen.get_rect().height, pipes)
                # generate inputs from bird
                bird.setInputs(pipes)

        for bird in birds:
            if bird.isAlive == True:
                # Calculate score
                for pipe in pipes:
                    if pipe.topRect.x <= bird.iniPosX and bird.hasCrossed(pipe.id) == False:
                        bird.pipesCrossed.append(pipe.id)
                        bird.score += 1

                # Calculate fitness
                bird.fitness = (ticks - 1.5 * bird.flaps)

        # Infinite pipes
        if pipes[0].topRect.x <= 0:
            pipes.remove(pipes[0])
            pipes.append(Pipe(pipes[len(pipes) - 1].topRect.x + PIPE_GAP, pipes[len(pipes) - 1].id + 1))

        for bird in birds:
            if bird.isAlive == True:
                output = bird.genome.evaluateNetwork([bird.height, bird.nextPipeHeight, bird.nextPipeDist, 1])
                if output[0] > 0.5:
                    bird.isJumping = True

        pygame.display.flip()
        ticks = ticks + 1
        time.sleep(FRAME_SLEEP_TIME)

        aliveStatus = []
        for bird in birds:
            aliveStatus.append(bird.isAlive)

        if all(isAlive == False for isAlive in aliveStatus):
            allDead = True
        else:
            allDead = False

        # Calculate maximum fitness
        max_fitness = 0
        max_score = 0
        mortality = 0
        for bird in birds:
            if bird.fitness > max_fitness:
                max_fitness = bird.fitness
            if bird.score > max_score:
                max_score = bird.score
            if bird.score < 1:
                mortality += 1

    return [birds, max_fitness, max_score, mortality]


# First generation *****************************************
generation_no = 0
print("**************************************")
print("Generation  : ", generation_no)

Pool.initializePool()
op = initializeGame()
birds = op[0]
max_fitness = op[1]
max_score = op[2]
mortality = op[3]

print("Maximum Fitness : ", max_fitness)
print("Maximum Score : ", max_score)
print("Mortality : ", mortality)

for bird in birds:
    bird.genome.fitness = bird.fitness

    if bird.fitness > Pool.maxFitness:
        Pool.maxFitness = bird.fitness

# Repeat generations ****************************************
while 1:

    generation_no += 1
    print("**************************************")
    print("Generation  : ", generation_no)

    Pool.newGeneration()
    op = initializeGame()
    birds = op[0]
    max_fitness = op[1]
    max_score = op[2]
    mortality = op[3]

    print("Maximum Fitness : ", max_fitness)
    print("Maximum Score : ", max_score)
    print("Mortality : ", mortality)

    for bird in birds:
        bird.genome.fitness = bird.fitness

        if bird.fitness > Pool.maxFitness:
            Pool.maxFitness = bird.fitness
