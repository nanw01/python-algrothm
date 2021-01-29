# -*- coding: utf-8 -*-
"""
Modify the Monte_Carlo_pi_list example by adding a plot.
The list stores all of the approximations of pi
Use matplotlib and pyplot to actually make the graph

Created on Monday Jan 27 23:05:41 2020

@author: mroll
"""

import random  # import the random package to generat random numbers
from matplotlib import pyplot as plt  # I'm uusing plt as shorthand for pylot


def distance():
    X = random.random()
    Y = random.random()
    return X**2 + Y**2


# Initialize some stuff. You need this if you run it multiple times
inside_circle = 0
total_points = 0
all_approximations = []  # hard brackets tells python it's a list

# Create a for loop that runs 1000 times, making counter go from 0 to 999
for counter in range(0, 1000):
    thing_to_test = distance()
    # Test to see if our (X,Y) is in the circle or not (under the line)
    if(thing_to_test < 1):  # note that what you do when if is true is indented
        inside_circle = inside_circle+1
    total_points = total_points+1  # Always update total points, no indent
    all_approximations.append(4*inside_circle/total_points)
    # add our current approximation to all_approximations using append
    # be sure to do it in the loop

plt.plot(all_approximations)  # plot that list of approximations
# pyplot is a useful plotting library as it does smart stuff by default
# Notice that I didn't tell it anything and the graph looks pretty good!
plt.show()
print("pi approximation is", 4*inside_circle/total_points)
