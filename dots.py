# Adapted from Code Bullet
# Smart Dots Genetic Algorithm
# https://github.com/Code-Bullet/Smart-Dots-Genetic-Algorithm-Tutorial
# Implemented for Python
# Lev Lewis 2019

from p5 import *
import numpy as np
import itertools

# Parameters
mutationRate = 0.01
children = 100
moves = 200
clampspeed = 6
window = 600
startposition = [300,550]
goal = [300,10]



class Obstacle:
    def __init__(self, pos, w, h):
        self.pos = pos
        self.h = h
        self.w = w

    def show(self):
        fill(0,0,255)
        rect(self.pos, self.w, self.h)


class Brain:
    def __init__(self, size):
        self.directions = []
        self.step = 0
        self.size = size
        self.randomize()

    def randomize(self):
        for _ in range(self.size):
            angle = np.random.uniform(0, 2.0*np.math.pi)
            vector = [np.math.cos(angle), np.math.sin(angle)]
            self.directions.append(vector)

    def clone(self):
        clone = Brain(self.size)
        clone.directions = self.directions.copy()
        return clone

    def mutate(self):
        for i in range(self.size):
            if(np.random.rand(1) < mutationRate):
                angle = np.random.uniform(0, 2.0*np.math.pi)
                vector = [np.math.cos(angle), np.math.sin(angle)]
                self.directions[i] = vector


class Dot:
    def __init__(self):
        self.brain = Brain(moves)
        self.clampspeed = clampspeed
        self.dead = False
        self.reachedGoal = False
        self.fitness = 0
        self.isBest = False
        self.hitObstacle = False

        # need to set type to float so self.vel += self.acc works
        # (self.acc is a float in directoins[])
        self.pos = np.array(startposition, dtype='f')
        self.vel = np.array([0,0], dtype='f')
        self.acc = np.array([0,0], dtype='f')

    def move(self):
        if (self.brain.size > self.brain.step):
            self.acc = self.brain.directions[self.brain.step]
            self.brain.step += 1
        else:
            self.dead = True
        
        self.vel += self.acc
        self.vel = np.clip(self.vel,-self.clampspeed,self.clampspeed)
        self.pos += self.vel

    def update(self):
        if (not self.dead and not self.reachedGoal):
            self.move()
            for obstacle in obstacles:
                if (0 < self.pos[0]-obstacle.pos[0] < obstacle.w and 0 < self.pos[1]-obstacle.pos[1] < obstacle.h):
                    self.hitObstacle = True

            if (self.pos[0] < 2 or self.pos[1] < 2 or self.pos[0] > (window-2) or self.pos[1] > window-2):
                self.dead = True

            elif (abs(self.pos[0]-goal[0]) < 5 and abs(self.pos[1]-goal[1]) < 5):
                self.reachedGoal = True
            
            elif (self.hitObstacle):
                self.dead = True

    def show(self):
        fill(0)
        if (self.dead):
            fill(255,255,0)
        if (self.reachedGoal):
            fill(0,255,0)
        if (self.isBest):
            fill(255,0,255)
        circle((self.pos[0], self.pos[1]),4)

    def calculateFitness(self):
        if (self.reachedGoal):
            self.fitness = (1.0/16) + (10000.0/float(self.brain.step*self.brain.step))
        else:
            distanceToGoal = np.linalg.norm(self.pos-goal)
            self.fitness = 1.0/(distanceToGoal * distanceToGoal)

    def getBaby(self):
        baby = Dot()
        baby.brain = self.brain.clone()
        return baby


class Population:
    def __init__(self, size):
        self.gen = 1
        self.steps = 0
        self.size = size
        self.dots = []
        self.bestStep = moves
        self.fitnessSum = 0
        self.bestDot = Dot()

        for _ in range(self.size):
            self.dots.append(Dot())

    def update(self):
        self.steps+=1

        for dot in self.dots:
            if(dot.brain.step > self.bestStep):
                dot.dead = True
            else:
                dot.update()

    def show(self):
        # Draw dots in reverse order so BEST is always on top (visible)
        for dot in reversed(self.dots):
            dot.show()
    
    def calculateFitness(self):
        for dot in self.dots:
            dot.calculateFitness()

    def getFitnessSum(self):
        self.fitnessSum = 0
        for dot in self.dots:
            self.fitnessSum += dot.fitness
        return self.fitnessSum

    def allDotsDead(self):
        for dot in self.dots:
            if(not dot.dead and not dot.reachedGoal):
                return False
        self.steps = 0 # Reset moves counter for debugging
        return True

    def naturalSelection(self):
        newDots = []

        self.bestDot = self.setBestDot()
        baby = self.bestDot.getBaby()
        baby.isBest = True
        newDots.append(baby)

        for _ in range(self.size-1):
            parent = self.getParent()
            baby = parent.getBaby()
            newDots.append(baby)
        self.dots = []
        self.dots = newDots
        self.gen += 1

    def mutateDots(self):
        for dot in self.dots:
            if (not dot.isBest): dot.brain.mutate()
    
    def getParent(self):
        fitnessSum = self.getFitnessSum()
        rand = np.random.uniform(0.0, fitnessSum)
        runningSum = 0.0

        for dot in self.dots:
            runningSum += dot.fitness
            if (runningSum > rand):
                return dot

    def setBestDot(self):
        maxFitness = 0
        self.bestDot = Dot()
        for dot in self.dots:
            if (dot.fitness > maxFitness):
                maxFitness = dot.fitness
                self.bestDot = dot
        if (self.bestDot.reachedGoal):
            self.bestStep = self.bestDot.brain.step
        return self.bestDot
        


# Initalize population of dots
dots = Population(children)

# Create obstacles
obstacles = []
obstacles.append(Obstacle((0,100),225,400))
obstacles.append(Obstacle((250,100),500,400))

def setup():
    size(window, window)
    
def draw():
    background(255)
    fill(255,0,0)
    circle(goal,10)

    for obstacle in obstacles:
        obstacle.show()
    
    if (dots.allDotsDead()):
        dots.calculateFitness()
        dots.naturalSelection()
        dots.mutateDots()

    else:
        # For debugging
        #print("Current Generation: {} ({}) \t Current Step: {}/{} \t Reached Goal? {} ({}/200) \t Score: {}".format(dots.gen, len(dots.dots), dots.steps, moves, dots.bestDot.reachedGoal, dots.bestStep, dots.bestDot.fitness))
        dots.update()
        dots.show()

def key_pressed(event):
    background(0)
    for i in range(moves):
        dots.update()

run(frame_rate=100)
