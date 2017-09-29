# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 13:11:05 2017

@author: Andrew
"""

Key = 1 # Rings
#Key = 2  # Tubes
#Key = 3  # Globes

if Key == 1:
	from integraleval import fitness_vector
elif Key == 2:
	from tubeseval import fitness_vector
elif Key == 3:
	from globeseval import fitness_vector
#import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import math
total_mass = 1000000
center_mass = 500000
ring_mass = total_mass - center_mass
initial_radius = 10000 

pop_size = 500
tournament_size = 8
crossover_rate = .9
mutation_rate = .9
target_fitness = -100.0

def generate_initial_distribution():
    p = [(center_mass, 0), (ring_mass, initial_radius)]
    return p
    
p = [generate_initial_distribution() for i in range(pop_size)]
total_momentum = fitness_vector(p[0])[-2]
print(total_momentum)
#total_momentum = .0000
def GA(p):
    best_fitness = 0
    while best_fitness >= target_fitness:
        fitness = []
        for j in p:
            fitness.append(fitness_vector(j)[-1])
        ng = []
        for j in range(pop_size):
            m = selection(p, fitness)
            f = selection(p, fitness)
            c = []
            if random.rand() < .7:
                c = crossover(m, f)
            else:
                c = m
            if random.rand() < .1:
 #               if (random.rand() < .05):
 #                   c = mutate_star(c)
 #               else:
                    c = mutation(c)
            c = fixChild(c)
            ng.append(c)
        best_fitness = sorted(fitness)[0]
        print(best_fitness)
        p = ng
    fitness = []
    for i in p:
        fitness.append(fitness_vector(i)[-1])
    best = 0
    for i in range(len(fitness)):
        if fitness[i] < fitness[best]:
            best = i
    return p[best]
def selection(p, fitness):
    best = random.randint(0, pop_size - 1)
    for i in range(7):
        r = random.randint(0, pop_size - 1)
        if fitness[r] < fitness[best]:
            best = r
    return p[best]
def mutation(p):
    ring = random.randint(1, len(p))
    mass_split = random.rand()
    random_distance = random.uniform(0, p[ring][1])
    ring1 = (p[ring][0] * mass_split, p[ring][1] - random_distance)
    ring2 = (p[ring][0] * (1 - mass_split), p[ring][1] + random_distance)
    child = []
    for i in range(len(p)):
        if i != ring:
            child.append(p[i])
    child = placeElement(child, ring1)
    child = placeElement(child, ring2)
    return child
    
def mutate_star(p):
    amount = random.rand() * p[0][0]
    distance = random.rand() * p[-1][1]
    child = []
    new_star = (p[0][0] - amount, 0)
    ring = (amount, distance)
    for i in range(len(p) - 1):
        child.append(p[i + 1])
    child = placeElement(child, new_star)
    child = placeElement(child, ring)
    return child
    
def mutation2(p):
    ring = random.randint(1, len(p))
    split = random.rand()
    distance = random.uniform(0, p[-1][1])
    newRing = (p[ring][0] * split, distance)
    oldRing = (p[ring][0] * (1 - split), p[ring][1])
    child = []
    for i in range(len(p)):
        if i != ring:
            child.append(p[i])
    child = placeElement(child, newRing)
    child = placeElement(child, oldRing)
    return child
    
def fixChild(p):
    vector = fitness_vector(p)
    fixedChild = []
    for i in range(len(vector) - 2): 
        if vector[i] < 0:
            p[i + 1] = (p[i][0] + p[i + 1][0], (p[i][1] + p[i + 1][1]) / 2)
            vector = fitness_vector(p)
        else:
            fixedChild.append(p[i])
    momentum = vector[-2]
    multiplier = total_momentum / momentum
    if Key != 2:  #Don't square it in the tubes case
    	multiplier = multiplier * multiplier 
    for i in range(len(fixedChild)):
        fixedChild[i] = (fixedChild[i][0], multiplier * fixedChild[i][1])
    return fixedChild
    
def placeElement(p, element):
    c = []
    placed = False
    for i in p:
        if element[1] == i[1]:
            c.append((i[0] + element[0]), i[1])
        else:
            if (not placed) and element[1] < i[1]:
                c.append(element)
                placed = True
            c.append(i)
    if not placed:
        c.append(element)
    return c
    
#computes elementwise weighted average between two parents both in terms of their mass and radius
def crossover(p1, p2):
    total_mass = 45
    i = 0
    m = 0
    m1 = 0
    m2 = 0
    w1 = random.rand()
    w2 = 1 - w1
    r = []
    child = []
    while i < len(p1) and i < len(p2):
        childM = p1[i][0] * w1 + p2[i][0] * w2
        m1 += p1[i][0]
        m2 += p2[i][0]
        m += childM
        childR = p1[i][1] * w1 + p2[i][1] * w2
        r.append(childR)
        child.append((childM, childR))
        i += 1
    remaining = total_mass - m
    if remaining > 0:
        if i < len(p1):
            px = p1
            remainingX = total_mass - m1
        else:
            px = p2
            remaining = total_mass - m2
        while i < len(px):
            mass_percent = px[i][0] / remainingX
            childM = mass_percent * remaining
            childR = px[i][1]
            child = placeElement(child, (childM, childR))
            i += 1
    return child  
r = GA(p)
x = []
y = []
# Drop last two values:
count = 0
for i in r[1:]:
		count = count + 1
#    x.append(i[1])
#    y.append(i[0])
		x.append(count)
		y.append(math.log10(i[1]))
plt.scatter(x, y)
plt.show()
print(r)
print(total_momentum)
#print(fitness_vector(r)[-2])
#print(fitness_vector(r))
