# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:25:34 2018

https://anaconda.org/pypi/aco-pants
http://aco-pants.readthedocs.io/en/latest/index.html#

@author: mhaa
"""

import pants
import math
import random
import numpy as np

import matplotlib.pyplot as plt

# random.seed(0)

nodes = []
for _ in range(5):   # 20
    x = random.uniform(-10, 10)
    y = random.uniform(-10, 10)
    nodes.append((x, y))


def euclidean(a, b):
    return math.sqrt(pow(a[1] - b[1], 2) + pow(a[0] - b[0], 2))
    
world = pants.World(nodes, euclidean)

solver = pants.Solver()

solution = solver.solve(world)

print(solution.distance)
print(solution.tour)    # Nodes visited in order
#print(solution.path)    # Edges taken in order

sol = np.array(solution.tour).reshape(-1, 2)

plnod = np.array(nodes)
fig=plt.figure()
plt.plot(plnod[:, 0], plnod[:, 1], 'o')   
plt.plot(sol[:, 0], sol[:, 1], '-') 

first_point = np.array(nodes[0])
print('First point:', first_point)

print('Last point:', x, y)

plt.plot(x, y, 'rx', ms=20)
plt.plot(first_point[0], first_point[1], 'k+', ms=20)

plt.show()

