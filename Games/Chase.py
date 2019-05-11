from Utils import Vector
from Utils import Circle
import pygame
import random
import numpy as np
import sys

class Chase:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.playerVelocity = 5
        self.enemyVelocity =  3
        self.stepReward = 1
        self.first = True

    def reset(self):
        self.player = Circle(Vector(random.random() * self.width, random.random() * self.height), 5)
        self.enemy = Circle(Vector(random.random() * self.width, random.random() * self.height), 5)
        return self.getState()

    # 0, 1, 2, 3 -> left, up, right, down
    def step(self, action):
        #should return
        #returns observation, reward, done, _

        if action == 0:
            self.player.moveX(-self.playerVelocity)
        elif action == 1:
            self.player.moveY(self.playerVelocity)
        elif action == 2:
            self.player.moveX(self.playerVelocity)
        elif action == 3:
            self.player.moveY(-self.playerVelocity)

        currentEnemyLoc = self.enemy.getCenter()
        currentPlayerLoc = self.player.getCenter()

        distanceVector = currentPlayerLoc.sub(currentEnemyLoc)
        distanceVector = distanceVector.norm()
        distanceVector = distanceVector.scale(self.enemyVelocity)
        newEnemyLoc = currentEnemyLoc.add(distanceVector)
        self.enemy.setCenter(newEnemyLoc)

        self.player.bound(0,self.width, 0, self.height)
        self.enemy.bound(0,self.width, 0, self.height)

        done = False
        if(self.player.intersects(self.enemy)):
            done = True

        return self.getState(), self.stepReward, done, 0

    def getState(self):
        layer1 = np.zeros((self.width, self.height))
        layer2 = np.zeros((self.width, self.height))

        return [layer1, layer2]

    def inputShape(self):
        return (2,self.width,self.height)

    def outputShape(self):
        return 4

    def render(self):
        # print("STARTING RENDER")
        if self.first:
            pygame.init()
            self.window = pygame.display.set_mode((100,100))
            pygame.display.set_caption("Chase")
            self.first = False


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                 sys.exit(0)

        self.window.fill((0,0,0))
        pygame.draw.circle(self.window, (255, 0, 0), (int(self.player.center.x),int(self.player.center.y)), 5)
        pygame.draw.circle(self.window, (0, 255, 0), (int(self.enemy.center.x),int(self.enemy.center.y)), 5)
        pygame.display.update()
        # print(self.enemy.center.x, self.enemy.center.y)
        # print("ENDING")1




    def close(self):
        pass
